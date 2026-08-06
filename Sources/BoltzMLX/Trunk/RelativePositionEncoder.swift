import MLX

/// Boltz-2 relative-position buckets followed by the learned int8 projection.
public struct RelativePositionEncoder {
  public let projection: AffineLinear
  public let fixSymCheck: Bool
  public let cyclicPositionEncoding: Bool

  public init(
    projection: AffineLinear,
    fixSymCheck: Bool,
    cyclicPositionEncoding: Bool
  ) {
    self.projection = projection
    self.fixSymCheck = fixSymCheck
    self.cyclicPositionEncoding = cyclicPositionEncoding
  }

  public func callAsFunction(_ features: [String: MLXArray]) throws -> MLXArray {
    try projection(
      relativePositionFeatures(
        features,
        fixSymCheck: fixSymCheck,
        cyclicPositionEncoding: cyclicPositionEncoding
      )
    )
  }
}

func relativePositionFeatures(
  _ features: [String: MLXArray],
  fixSymCheck: Bool,
  cyclicPositionEncoding: Bool,
  residueMaximum: Int = 32,
  symmetryMaximum: Int = 2
) throws -> MLXArray {
  let asymID = try requireFeature("asym_id", from: features)
  let residueIndex = try requireFeature("residue_index", from: features)
  let entityID = try requireFeature("entity_id", from: features)
  let tokenIndex = try requireFeature("token_index", from: features)
  let symmetryID = try requireFeature("sym_id", from: features)

  let sameChain = asymID.expandedDimensions(axis: 2) .== asymID.expandedDimensions(axis: 1)
  let sameResidue =
    residueIndex.expandedDimensions(axis: 2) .== residueIndex.expandedDimensions(axis: 1)
  let sameEntity =
    entityID.expandedDimensions(axis: 2) .== entityID.expandedDimensions(axis: 1)

  var residueDelta =
    residueIndex.expandedDimensions(axis: 2) - residueIndex.expandedDimensions(axis: 1)
  if cyclicPositionEncoding {
    let cyclicPeriod = try requireFeature("cyclic_period", from: features)
    let period = MLX.which(
      cyclicPeriod .> 0,
      cyclicPeriod,
      MLX.full(cyclicPeriod.shape, values: 10_000)
    )
    let pairPeriod = period.expandedDimensions(axis: 2)
    let wrapped = MLX.round(residueDelta.asType(.float32) / pairPeriod.asType(.float32))
    residueDelta = residueDelta - pairPeriod * wrapped.asType(residueDelta.dtype)
  }

  let residueBucket = MLX.which(
    sameChain,
    MLX.clip(residueDelta + residueMaximum, min: 0, max: 2 * residueMaximum),
    MLX.full(residueDelta.shape, values: 2 * residueMaximum + 1)
  )
  let relativeResidue = oneHot(residueBucket, classes: 2 * residueMaximum + 2)

  let tokenDelta =
    tokenIndex.expandedDimensions(axis: 2) - tokenIndex.expandedDimensions(axis: 1)
  let tokenBucket = MLX.which(
    MLX.logicalAnd(sameChain, sameResidue),
    MLX.clip(tokenDelta + residueMaximum, min: 0, max: 2 * residueMaximum),
    MLX.full(tokenDelta.shape, values: 2 * residueMaximum + 1)
  )
  let relativeToken = oneHot(tokenBucket, classes: 2 * residueMaximum + 2)

  let chainDelta =
    symmetryID.expandedDimensions(axis: 2) - symmetryID.expandedDimensions(axis: 1)
  let invalidChain = fixSymCheck ? MLX.logicalNot(sameEntity) : sameChain
  let chainBucket = MLX.which(
    invalidChain,
    MLX.full(chainDelta.shape, values: 2 * symmetryMaximum + 1),
    MLX.clip(chainDelta + symmetryMaximum, min: 0, max: 2 * symmetryMaximum)
  )
  let relativeChain = oneHot(chainBucket, classes: 2 * symmetryMaximum + 2)

  return MLX.concatenated(
    [
      relativeResidue,
      relativeToken,
      sameEntity.expandedDimensions(axis: -1).asType(.float32),
      relativeChain,
    ],
    axis: -1
  )
}

func requireFeature(_ name: String, from features: [String: MLXArray]) throws -> MLXArray {
  guard let value = features[name] else {
    throw BoltzError.missingTensor(name)
  }
  return value
}

func oneHot(_ indices: MLXArray, classes: Int) -> MLXArray {
  let categories = MLXArray(0..<classes)
  return (indices.expandedDimensions(axis: -1) .== categories).asType(.float32)
}
