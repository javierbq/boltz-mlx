import MLX

/// Windowed atom transformer wrapping the generic conditioned transformer stack.
public struct AtomTransformer {
  public let queryWindow: Int
  public let keyWindow: Int
  public let transformer: DiffusionTransformer

  public func callAsFunction(
    query: MLXArray,
    mask: MLXArray,
    conditioning: MLXArray,
    bias: MLXArray,
    multiplicity: Int,
    windowing: AtomWindowing
  ) -> MLXArray {
    let batch = query.shape[0]
    let atomCount = query.shape[1]
    let width = query.shape[2]
    let windowCount = atomCount / queryWindow
    let windowedQuery = query.reshaped(batch * windowCount, queryWindow, width)
    let windowedConditioning = conditioning.reshaped(
      batch * windowCount,
      queryWindow,
      conditioning.shape[2]
    )
    let windowedMask = mask.reshaped(batch * windowCount, queryWindow)
    let windowedBias = MLX.repeated(bias, count: multiplicity, axis: 0)
      .reshaped(batch * windowCount, queryWindow, keyWindow, -1)
    let toKeys: (MLXArray) -> MLXArray = { input in
      let featureWidth = input.shape[2]
      return windowing.toKeys(input.reshaped(batch, atomCount, featureWidth))
        .reshaped(batch * windowCount, self.keyWindow, featureWidth)
    }
    return transformer(
      activation: windowedQuery,
      conditioning: windowedConditioning,
      bias: windowedBias,
      mask: windowedMask,
      toKeys: toKeys
    ).reshaped(batch, atomCount, width)
  }
}

/// Atom attention plus atom-to-token aggregation used by both trunk input and score model.
public struct AtomAttentionEncoder {
  public struct Output {
    public let token: MLXArray
    public let query: MLXArray
    public let conditioning: MLXArray
    public let windowing: AtomWindowing
  }

  public let structurePrediction: Bool
  public let coordinateToQuery: AffineLinear?
  public let atomTransformer: AtomTransformer
  public let atomToToken: AffineLinear

  public func callAsFunction(
    features: [String: MLXArray],
    query: MLXArray,
    conditioning: MLXArray,
    bias: MLXArray,
    windowing: AtomWindowing,
    coordinates: MLXArray? = nil,
    multiplicity: Int = 1
  ) throws -> Output {
    var q = query
    if structurePrediction {
      guard let coordinates, let coordinateToQuery else {
        throw BoltzError.missingTensor("atom attention coordinates")
      }
      q = MLX.repeated(q, count: multiplicity, axis: 0) + coordinateToQuery(coordinates)
    }
    let c = MLX.repeated(conditioning, count: multiplicity, axis: 0)
    let atomMask = MLX.repeated(
      try requireFeature("atom_pad_mask", from: features),
      count: multiplicity,
      axis: 0
    )
    q = atomTransformer(
      query: q,
      mask: atomMask,
      conditioning: c,
      bias: bias,
      multiplicity: multiplicity,
      windowing: windowing
    )
    let projected = MLX.maximum(atomToToken(q), 0).asType(.float32)
    var atomToTokenWeights = try requireFeature("atom_to_token", from: features)
      .asType(.float32)
    atomToTokenWeights = MLX.repeated(atomToTokenWeights, count: multiplicity, axis: 0)
    let meanWeights =
      atomToTokenWeights
      / (atomToTokenWeights.sum(axis: 1, keepDims: true) + 1e-6)
    let token = MLX.matmul(meanWeights.transposed(0, 2, 1), projected).asType(q.dtype)
    return Output(token: token, query: q, conditioning: c, windowing: windowing)
  }
}

extension BoltzWeightStore {
  func atomTransformer(
    _ prefix: String,
    depth: Int,
    width: Int,
    headCount: Int,
    queryWindow: Int,
    keyWindow: Int,
    postLayerNorm: Bool = false
  ) throws -> AtomTransformer {
    try AtomTransformer(
      queryWindow: queryWindow,
      keyWindow: keyWindow,
      transformer: diffusionTransformer(
        "\(prefix).diffusion_transformer",
        depth: depth,
        width: width,
        headCount: headCount,
        postLayerNorm: postLayerNorm
      )
    )
  }

  func atomAttentionEncoder(
    _ prefix: String,
    depth: Int,
    headCount: Int,
    atomWidth: Int,
    queryWindow: Int,
    keyWindow: Int,
    structurePrediction: Bool
  ) throws -> AtomAttentionEncoder {
    try AtomAttentionEncoder(
      structurePrediction: structurePrediction,
      coordinateToQuery: structurePrediction ? linear("\(prefix).r_to_q_trans") : nil,
      atomTransformer: atomTransformer(
        "\(prefix).atom_encoder",
        depth: depth,
        width: atomWidth,
        headCount: headCount,
        queryWindow: queryWindow,
        keyWindow: keyWindow
      ),
      atomToToken: linear("\(prefix).atom_to_token_trans.0")
    )
  }
}
