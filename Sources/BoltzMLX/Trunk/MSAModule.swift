import MLX

/// Pair-only block shared by the MSA and template stacks.
public struct PairStackLayer {
  public let outgoingMultiplication: TriangleMultiplication
  public let incomingMultiplication: TriangleMultiplication
  public let startingAttention: TriangleAttention
  public let endingAttention: TriangleAttention
  public let transition: Transition

  public func callAsFunction(_ pair: MLXArray, mask: MLXArray) -> MLXArray {
    var z = pair
    z = z + outgoingMultiplication(z, mask: mask)
    z = z + incomingMultiplication(z, mask: mask)
    z = z + startingAttention(z, mask: mask)
    z = z + endingAttention(z, mask: mask)
    z = z + transition(z)
    return z
  }
}

/// One MSA communication block followed by a pair-only update.
public struct MSALayer {
  public let pairWeightedAveraging: PairWeightedAveraging
  public let msaTransition: Transition
  public let outerProduct: OuterProductMean
  public let pairStack: PairStackLayer

  public func callAsFunction(
    pair: MLXArray,
    msa: MLXArray,
    pairMask: MLXArray,
    msaMask: MLXArray
  ) -> (pair: MLXArray, msa: MLXArray) {
    var m = msa + pairWeightedAveraging(msa, pair: pair, mask: pairMask)
    m = m + msaTransition(m)
    var z = pair + outerProduct(m, mask: msaMask)
    z = pairStack(z, mask: pairMask)
    return (z, m)
  }
}

/// Boltz-2 MSA stack consuming the precomputed MSA feature arrays.
public struct MSAModule {
  public let tokenClasses: Int
  public let includePaired: Bool
  public let msaProjection: AffineLinear
  public let sequenceProjection: AffineLinear
  public let layers: [MSALayer]

  public init(
    tokenClasses: Int = 33,
    includePaired: Bool,
    msaProjection: AffineLinear,
    sequenceProjection: AffineLinear,
    layers: [MSALayer]
  ) {
    self.tokenClasses = tokenClasses
    self.includePaired = includePaired
    self.msaProjection = msaProjection
    self.sequenceProjection = sequenceProjection
    self.layers = layers
  }

  public func callAsFunction(
    pair: MLXArray,
    sequenceInput: MLXArray,
    features: [String: MLXArray]
  ) throws -> MLXArray {
    let msaMask = try requireFeature("msa_mask", from: features)
    let tokenMask = try requireFeature("token_pad_mask", from: features).asType(.float32)
    let pairMask =
      tokenMask.expandedDimensions(axis: 2) * tokenMask.expandedDimensions(axis: 1)
    var m = msaProjection(
      try msaInputFeatures(
        features,
        tokenClasses: tokenClasses,
        includePaired: includePaired
      )
    )
    m = m + sequenceProjection(sequenceInput).expandedDimensions(axis: 1)
    var z = pair
    for layer in layers {
      (z, m) = layer(pair: z, msa: m, pairMask: pairMask, msaMask: msaMask)
    }
    return z
  }
}

func msaInputFeatures(
  _ features: [String: MLXArray],
  tokenClasses: Int,
  includePaired: Bool
) throws -> MLXArray {
  let msa = oneHot(try requireFeature("msa", from: features), classes: tokenClasses)
  var inputs = [
    msa,
    try requireFeature("has_deletion", from: features).expandedDimensions(axis: -1),
    try requireFeature("deletion_value", from: features).expandedDimensions(axis: -1),
  ]
  if includePaired {
    inputs.append(
      try requireFeature("msa_paired", from: features).expandedDimensions(axis: -1)
    )
  }
  return MLX.concatenated(inputs.map { $0.asType(.float32) }, axis: -1)
}

extension BoltzWeightStore {
  func pairStackLayer(
    _ prefix: String,
    pairWidth: Int,
    headWidth: Int,
    headCount: Int
  ) throws -> PairStackLayer {
    try PairStackLayer(
      outgoingMultiplication: triangleMultiplication(
        "\(prefix).tri_mul_out",
        direction: .outgoing
      ),
      incomingMultiplication: triangleMultiplication(
        "\(prefix).tri_mul_in",
        direction: .incoming
      ),
      startingAttention: triangleAttention(
        "\(prefix).tri_att_start",
        startingNode: true,
        pairWidth: pairWidth,
        headWidth: headWidth,
        headCount: headCount
      ),
      endingAttention: triangleAttention(
        "\(prefix).tri_att_end",
        startingNode: false,
        pairWidth: pairWidth,
        headWidth: headWidth,
        headCount: headCount
      ),
      transition: transition("\(prefix).transition_z")
    )
  }

  func pairWeightedAveraging(
    _ prefix: String,
    hiddenWidth: Int = 32,
    headCount: Int = 8
  ) throws -> PairWeightedAveraging {
    try PairWeightedAveraging(
      hiddenWidth: hiddenWidth,
      headCount: headCount,
      msaNorm: layerNorm("\(prefix).norm_m"),
      pairNorm: layerNorm("\(prefix).norm_z"),
      value: linear("\(prefix).proj_m"),
      gate: linear("\(prefix).proj_g"),
      pairBias: linear("\(prefix).proj_z"),
      output: linear("\(prefix).proj_o")
    )
  }

  func outerProductMean(_ prefix: String) throws -> OuterProductMean {
    try OuterProductMean(
      norm: layerNorm("\(prefix).norm"),
      first: linear("\(prefix).proj_a"),
      second: linear("\(prefix).proj_b"),
      output: linear("\(prefix).proj_o")
    )
  }

  func msaModule(
    configuration: MSAConfiguration,
    pairWidth: Int
  ) throws -> MSAModule {
    var layers: [MSALayer] = []
    layers.reserveCapacity(configuration.msaBlocks)
    for index in 0..<configuration.msaBlocks {
      let prefix = "msa_module.layers.\(index)"
      layers.append(
        try MSALayer(
          pairWeightedAveraging: pairWeightedAveraging(
            "\(prefix).pair_weighted_averaging"
          ),
          msaTransition: transition("\(prefix).msa_transition"),
          outerProduct: outerProductMean("\(prefix).outer_product_mean"),
          pairStack: pairStackLayer(
            "\(prefix).pairformer_layer",
            pairWidth: pairWidth,
            headWidth: configuration.pairwiseHeadWidth,
            headCount: configuration.pairwiseNumHeads
          )
        )
      )
    }
    return try MSAModule(
      includePaired: configuration.usePairedFeature,
      msaProjection: linear("msa_module.msa_proj"),
      sequenceProjection: linear("msa_module.s_proj"),
      layers: layers
    )
  }
}
