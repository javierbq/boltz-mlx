import MLX

/// Repeated pair-only stack used inside each template projection.
public struct PairStack {
  public let layers: [PairStackLayer]

  public func callAsFunction(_ pair: MLXArray, mask: MLXArray) -> MLXArray {
    var z = pair
    for layer in layers {
      z = layer(z, mask: mask)
    }
    return z
  }
}

/// Boltz-2 template encoder, including visibility-aware v2 masking.
public struct TemplateModule {
  public let useVisibilityMask: Bool
  public let minimumDistance: Float
  public let maximumDistance: Float
  public let distanceBins: Int
  public let pairNorm: BoltzLayerNorm
  public let templateNorm: BoltzLayerNorm
  public let pairProjection: AffineLinear
  public let featureProjection: AffineLinear
  public let outputProjection: AffineLinear
  public let pairStack: PairStack

  public func callAsFunction(
    pair: MLXArray,
    features: [String: MLXArray],
    pairMask: MLXArray
  ) throws -> MLXArray {
    let residueType = try requireFeature("template_restype", from: features).asType(.float32)
    let frameRotation = try requireFeature("template_frame_rot", from: features)
    let frameTranslation = try requireFeature("template_frame_t", from: features)
    let frameMask = try requireFeature("template_mask_frame", from: features)
    let betaCoordinates = try requireFeature("template_cb", from: features)
    let alphaCoordinates = try requireFeature("template_ca", from: features)
    let betaMask = try requireFeature("template_mask_cb", from: features)
    let rawTemplateMask = try requireFeature("template_mask", from: features)
    let templateMask = rawTemplateMask.any(axis: 2).asType(.float32)
    let templateCount = MLX.maximum(templateMask.sum(axis: 1), 1)

    let betaPairMask =
      betaMask.expandedDimensions(axis: 3) * betaMask.expandedDimensions(axis: 2)
    let framePairMask =
      frameMask.expandedDimensions(axis: 3) * frameMask.expandedDimensions(axis: 2)
    let templatePairMask: MLXArray
    if useVisibilityMask {
      let visibility = try requireFeature("visibility_ids", from: features)
      templatePairMask =
        (visibility.expandedDimensions(axis: 3) .== visibility.expandedDimensions(axis: 2))
        .asType(.float32)
    } else {
      let asymmetry = try requireFeature("asym_id", from: features)
      templatePairMask =
        (asymmetry.expandedDimensions(axis: 2) .== asymmetry.expandedDimensions(axis: 1))
        .expandedDimensions(axis: 1).asType(.float32)
    }

    let distanceFeatures = templateDistanceBuckets(
      betaCoordinates,
      minimum: minimumDistance,
      maximum: maximumDistance,
      count: distanceBins
    )
    let rotation = frameRotation.expandedDimensions(axis: 2)
      .transposed(0, 1, 2, 3, 5, 4)
    let translation = frameTranslation.expandedDimensions(axis: 2)
      .expandedDimensions(axis: -1)
    let alpha = alphaCoordinates.expandedDimensions(axis: 3)
      .expandedDimensions(axis: -1)
    let vector = MLX.matmul(rotation, alpha - translation).squeezed(axis: -1)
    let vectorNorm = MLX.sqrt((vector * vector).sum(axis: -1, keepDims: true))
    let unitVector = MLX.which(vectorNorm .> 0, vector / vectorNorm, MLX.zeros(vector.shape))

    var templateFeatures = MLX.concatenated(
      [
        distanceFeatures,
        betaPairMask.expandedDimensions(axis: -1),
        unitVector,
        framePairMask.expandedDimensions(axis: -1),
      ],
      axis: -1
    )
    templateFeatures = templateFeatures * templatePairMask.expandedDimensions(axis: -1)
    let tokenCount = residueType.shape[2]
    let residueI = MLX.broadcast(
      residueType.expandedDimensions(axis: 3),
      to: Array(residueType.shape.prefix(3)) + [tokenCount, residueType.shape[3]]
    )
    let residueJ = MLX.broadcast(
      residueType.expandedDimensions(axis: 2),
      to: Array(residueType.shape.prefix(3)) + [tokenCount, residueType.shape[3]]
    )
    templateFeatures = featureProjection(
      MLX.concatenated([templateFeatures, residueI, residueJ], axis: -1)
    )

    let batch = pair.shape[0]
    let templateTotal = residueType.shape[1]
    var v = pairProjection(pairNorm(pair).expandedDimensions(axis: 1)) + templateFeatures
    v = v.reshaped(batch * templateTotal, tokenCount, tokenCount, -1)
    let expandedPairMask = MLX.broadcast(
      pairMask.expandedDimensions(axis: 1),
      to: [batch, templateTotal, tokenCount, tokenCount]
    ).reshaped(batch * templateTotal, tokenCount, tokenCount)
    v = v + pairStack(v, mask: expandedPairMask)
    v = templateNorm(v).reshaped(batch, templateTotal, tokenCount, tokenCount, -1)

    let weighted = v * templateMask.expandedDimensions(axes: [2, 3, 4])
    let averaged = weighted.sum(axis: 1) / templateCount.expandedDimensions(axes: [1, 2, 3])
    return outputProjection(MLX.maximum(averaged, 0))
  }
}

func templateDistanceBuckets(
  _ coordinates: MLXArray,
  minimum: Float,
  maximum: Float,
  count: Int
) -> MLXArray {
  let difference =
    coordinates.expandedDimensions(axis: 3) - coordinates.expandedDimensions(axis: 2)
  let distances = MLX.sqrt((difference * difference).sum(axis: -1))
  let boundaries = MLX.linspace(minimum, maximum, count: count - 1)
  let bucket = (distances.expandedDimensions(axis: -1) .> boundaries)
    .asType(.int32).sum(axis: -1)
  return oneHot(bucket, classes: count)
}

extension BoltzWeightStore {
  func pairStack(
    _ prefix: String,
    blockCount: Int,
    pairWidth: Int,
    headWidth: Int,
    headCount: Int
  ) throws -> PairStack {
    var layers: [PairStackLayer] = []
    for index in 0..<blockCount {
      layers.append(
        try pairStackLayer(
          "\(prefix).layers.\(index)",
          pairWidth: pairWidth,
          headWidth: headWidth,
          headCount: headCount
        )
      )
    }
    return PairStack(layers: layers)
  }

  func templateModule(
    configuration: TemplateConfiguration,
    pairWidth: Int,
    useV2: Bool
  ) throws -> TemplateModule {
    try TemplateModule(
      useVisibilityMask: useV2,
      minimumDistance: 3.25,
      maximumDistance: 50.75,
      distanceBins: 38,
      pairNorm: layerNorm("template_module.z_norm"),
      templateNorm: layerNorm("template_module.v_norm"),
      pairProjection: linear("template_module.z_proj"),
      featureProjection: linear("template_module.a_proj"),
      outputProjection: linear("template_module.u_proj"),
      pairStack: pairStack(
        "template_module.pairformer",
        blockCount: configuration.templateBlocks,
        pairWidth: configuration.templateDim,
        headWidth: configuration.pairwiseHeadWidth ?? 32,
        headCount: configuration.pairwiseNumHeads ?? 4
      )
    )
  }
}
