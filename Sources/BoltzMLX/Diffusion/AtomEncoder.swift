import MLX

/// Deterministic mapping from atom query windows to overlapping key windows.
public struct AtomWindowing {
  public let atomCount: Int
  public let queryWindow: Int
  public let keyWindow: Int
  public let indexingMatrix: MLXArray

  public init(atomCount: Int, queryWindow: Int, keyWindow: Int) {
    precondition(queryWindow.isMultiple(of: 2))
    precondition(atomCount.isMultiple(of: queryWindow))
    precondition(keyWindow.isMultiple(of: queryWindow / 2))
    let windowCount = atomCount / queryWindow
    let halfWindowSpan = keyWindow / (queryWindow / 2)
    precondition(halfWindowSpan.isMultiple(of: 2))
    var values = Array(
      repeating: Float(0),
      count: 2 * windowCount * halfWindowSpan * windowCount
    )
    let columns = halfWindowSpan * windowCount
    for source in 0..<(2 * windowCount) {
      for window in 0..<windowCount {
        let bucket = max(0, min(halfWindowSpan + 1, source - 2 * window + halfWindowSpan / 2))
        if bucket > 0, bucket <= halfWindowSpan {
          values[source * columns + window * halfWindowSpan + bucket - 1] = 1
        }
      }
    }
    self.atomCount = atomCount
    self.queryWindow = queryWindow
    self.keyWindow = keyWindow
    self.indexingMatrix = MLXArray(values, [2 * windowCount, columns])
  }

  public func toKeys(_ input: MLXArray) -> MLXArray {
    let batch = input.shape[0]
    let width = input.shape[2]
    let windowCount = atomCount / queryWindow
    let halfQueries = queryWindow / 2
    let split = input.reshaped(batch, 2 * windowCount, halfQueries, width)
    return MLX.einsum("bjid,jk->bkid", split, indexingMatrix)
      .reshaped(batch, windowCount, keyWindow, width)
  }
}

/// Static atom reference encoder shared by the trunk and diffusion conditioning.
public struct AtomEncoder {
  public struct Output {
    public let query: MLXArray
    public let conditioning: MLXArray
    public let pair: MLXArray
    public let windowing: AtomWindowing
  }

  public let queryWindow: Int
  public let keyWindow: Int
  public let useNoAtomCharacters: Bool
  public let useBackboneFeature: Bool
  public let useResidueFeatures: Bool
  public let atomFeatures: AffineLinear
  public let referencePosition: AffineLinear
  public let referenceDistance: AffineLinear
  public let referenceMask: AffineLinear
  public let sequenceNorm: BoltzLayerNorm?
  public let sequenceToAtom: AffineLinear?
  public let pairNorm: BoltzLayerNorm?
  public let pairToAtom: AffineLinear?
  public let conditioningToPairKey: AffineLinear
  public let conditioningToPairQuery: AffineLinear
  public let pairMLP: [AffineLinear]

  public func callAsFunction(
    features: [String: MLXArray],
    sequence: MLXArray? = nil,
    pair: MLXArray? = nil
  ) throws -> Output {
    let reference = try requireFeature("ref_pos", from: features).asType(.float32)
    let batch = reference.shape[0]
    let atomCount = reference.shape[1]
    let atomMask = try requireFeature("atom_pad_mask", from: features)
    let atomUID = try requireFeature("ref_space_uid", from: features)
    var inputs = [
      reference,
      try requireFeature("ref_charge", from: features).expandedDimensions(axis: -1),
      try requireFeature("ref_element", from: features),
    ]
    if !useNoAtomCharacters {
      inputs.append(
        try requireFeature("ref_atom_name_chars", from: features).reshaped(batch, atomCount, -1)
      )
    }
    if useBackboneFeature {
      inputs.append(try requireFeature("atom_backbone_feat", from: features))
    }
    if useResidueFeatures {
      let residue = MLX.concatenated(
        [
          try requireFeature("res_type", from: features),
          try requireFeature("modified", from: features).expandedDimensions(axis: -1),
          oneHot(try requireFeature("mol_type", from: features), classes: 4),
        ],
        axis: -1
      )
      let atomToToken = try requireFeature("atom_to_token", from: features).asType(.float32)
      inputs.append(MLX.matmul(atomToToken, residue.asType(.float32)))
    }
    var c = atomFeatures(MLX.concatenated(inputs.map { $0.asType(.float32) }, axis: -1))
    let query = c
    let windowing = AtomWindowing(
      atomCount: atomCount,
      queryWindow: queryWindow,
      keyWindow: keyWindow
    )
    let windowCount = atomCount / queryWindow
    let referenceQueries = reference.reshaped(batch, windowCount, queryWindow, 1, 3)
    let referenceKeys = windowing.toKeys(reference).reshaped(batch, windowCount, 1, keyWindow, 3)
    let difference = referenceKeys - referenceQueries
    let reciprocalSquaredDistance =
      1 / (1 + (difference * difference).sum(axis: -1, keepDims: true))

    let maskQueries = atomMask.reshaped(batch, windowCount, queryWindow, 1)
    let maskKeys = windowing.toKeys(atomMask.expandedDimensions(axis: -1).asType(.float32))
      .reshaped(batch, windowCount, 1, keyWindow).asType(.bool)
    let uidQueries = atomUID.reshaped(batch, windowCount, queryWindow, 1)
    let uidKeys = windowing.toKeys(atomUID.expandedDimensions(axis: -1).asType(.float32))
      .reshaped(batch, windowCount, 1, keyWindow).asType(atomUID.dtype)
    let valid = MLX.logicalAnd(
      MLX.logicalAnd(maskQueries.asType(.bool), maskKeys),
      uidQueries .== uidKeys
    ).expandedDimensions(axis: -1).asType(.float32)
    var p = referencePosition(difference) * valid
    p = p + referenceDistance(reciprocalSquaredDistance) * valid
    p = p + referenceMask(valid) * valid

    if let sequenceNorm, let sequenceToAtom, let pairNorm, let pairToAtom {
      guard let sequence, let pair else {
        throw BoltzError.missingTensor("atom encoder trunk inputs")
      }
      let atomToToken = try requireFeature("atom_to_token", from: features).asType(.float32)
      c = c + MLX.matmul(atomToToken, sequenceToAtom(sequenceNorm(sequence.asType(.float32))))
      let tokenQueries = atomToToken.reshaped(
        batch,
        windowCount,
        queryWindow,
        atomToToken.shape[2]
      )
      let tokenKeys = windowing.toKeys(atomToToken)
      // Contract the two token indices sequentially (i, then j). A single 3-way
      // einsum lets MLX pick a contraction order that materializes an
      // atoms*tokens^2 intermediate (28 GB at 384 tokens) and OOMs; the two-step
      // form peaks at a few MB and is numerically identical.
      let atomPair = pairToAtom(pairNorm(pair.asType(.float32)))
      let queryContracted = MLX.einsum("bijd,bwki->bwkjd", atomPair, tokenQueries)
      p = p + MLX.einsum("bwkjd,bwlj->bwkld", queryContracted, tokenKeys)
    }

    // c_to_p_trans_q / c_to_p_trans_k are Sequential(ReLU, Linear) in Boltz; the
    // stored `.1` weight is only the Linear, so the ReLU must be applied here.
    p =
      p
      + conditioningToPairQuery(
        MLX.maximum(c.reshaped(batch, windowCount, queryWindow, 1, c.shape[2]), 0)
      )
    p =
      p
      + conditioningToPairKey(
        MLX.maximum(
          windowing.toKeys(c).reshaped(batch, windowCount, 1, keyWindow, c.shape[2]),
          0
        )
      )
    var mlp = p
    for linear in pairMLP {
      mlp = linear(MLX.maximum(mlp, 0))
    }
    p = p + mlp
    return Output(query: query, conditioning: c, pair: p, windowing: windowing)
  }
}

extension BoltzWeightStore {
  func atomEncoder(
    _ prefix: String,
    queryWindow: Int,
    keyWindow: Int,
    structurePrediction: Bool,
    useNoAtomCharacters: Bool = false,
    useBackboneFeature: Bool = false,
    useResidueFeatures: Bool = false
  ) throws -> AtomEncoder {
    try AtomEncoder(
      queryWindow: queryWindow,
      keyWindow: keyWindow,
      useNoAtomCharacters: useNoAtomCharacters,
      useBackboneFeature: useBackboneFeature,
      useResidueFeatures: useResidueFeatures,
      atomFeatures: linear("\(prefix).embed_atom_features"),
      referencePosition: linear("\(prefix).embed_atompair_ref_pos"),
      referenceDistance: linear("\(prefix).embed_atompair_ref_dist"),
      referenceMask: linear("\(prefix).embed_atompair_mask"),
      sequenceNorm: structurePrediction ? layerNorm("\(prefix).s_to_c_trans.0") : nil,
      sequenceToAtom: structurePrediction ? linear("\(prefix).s_to_c_trans.1") : nil,
      pairNorm: structurePrediction ? layerNorm("\(prefix).z_to_p_trans.0") : nil,
      pairToAtom: structurePrediction ? linear("\(prefix).z_to_p_trans.1") : nil,
      conditioningToPairKey: linear("\(prefix).c_to_p_trans_k.1"),
      conditioningToPairQuery: linear("\(prefix).c_to_p_trans_q.1"),
      pairMLP: [
        linear("\(prefix).p_mlp.1"),
        linear("\(prefix).p_mlp.3"),
        linear("\(prefix).p_mlp.5"),
      ]
    )
  }
}
