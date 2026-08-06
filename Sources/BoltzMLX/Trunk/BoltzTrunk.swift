import MLX

/// Outputs retained for diffusion conditioning and optional parity inspection.
public struct TrunkOutput {
  public let sequence: MLXArray
  public let pair: MLXArray
  public let sequenceInput: MLXArray
  public let relativePosition: MLXArray
}

/// Native MLX implementation of the Boltz-2 input and Pairformer trunk.
public struct BoltzTrunk {
  public let inputEmbedder: InputEmbedder
  public let sequenceInitial: AffineLinear
  public let pairInitialFirst: AffineLinear
  public let pairInitialSecond: AffineLinear
  public let relativePosition: RelativePositionEncoder
  public let tokenBonds: AffineLinear
  public let tokenBondType: AffineEmbedding?
  public let contactConditioning: ContactConditioning
  public let sequenceNorm: BoltzLayerNorm
  public let pairNorm: BoltzLayerNorm
  public let sequenceRecycle: AffineLinear
  public let pairRecycle: AffineLinear
  public let templateModule: TemplateModule?
  public let msaModule: MSAModule
  public let pairformer: Pairformer

  public func callAsFunction(
    features: [String: MLXArray],
    recyclingSteps: Int,
    clearCacheBetweenRecycles: Bool = true
  ) throws -> TrunkOutput {
    let sequenceInput = try inputEmbedder(features)
    let initialSequence = sequenceInitial(sequenceInput)
    let relative = try relativePosition(features)
    var initialPair =
      pairInitialFirst(sequenceInput).expandedDimensions(axis: 2)
      + pairInitialSecond(sequenceInput).expandedDimensions(axis: 1)
    initialPair = initialPair + relative
    initialPair =
      initialPair
      + tokenBonds(try requireFeature("token_bonds", from: features).asType(.float32))
    if let tokenBondType {
      initialPair =
        initialPair
        + tokenBondType(try requireFeature("type_bonds", from: features))
    }
    initialPair = initialPair + (try contactConditioning(features))

    var sequence = MLX.zeros(initialSequence.shape, dtype: initialSequence.dtype)
    var pair = MLX.zeros(initialPair.shape, dtype: initialPair.dtype)
    let mask = try requireFeature("token_pad_mask", from: features).asType(.float32)
    let pairMask = mask.expandedDimensions(axis: 2) * mask.expandedDimensions(axis: 1)
    for _ in 0...recyclingSteps {
      sequence = initialSequence + sequenceRecycle(sequenceNorm(sequence))
      pair = initialPair + pairRecycle(pairNorm(pair))
      if let templateModule {
        pair =
          pair
          + (try templateModule(pair: pair, features: features, pairMask: pairMask))
      }
      pair =
        pair
        + (try msaModule(pair: pair, sequenceInput: sequenceInput, features: features))
      (sequence, pair) = pairformer(
        sequence: sequence,
        pair: pair,
        mask: mask,
        pairMask: pairMask
      )
      MLX.eval(sequence, pair)
      if clearCacheBetweenRecycles {
        Memory.clearCache()
      }
    }
    return TrunkOutput(
      sequence: sequence,
      pair: pair,
      sequenceInput: sequenceInput,
      relativePosition: relative
    )
  }
}

extension BoltzWeightStore {
  func trunk() throws -> BoltzTrunk {
    guard let configuration = artifact.configuration else {
      throw BoltzError.missingFile("config.json")
    }
    let unspecifiedName = "contact_conditioning.encoding_unspecified"
    let unselectedName = "contact_conditioning.encoding_unselected"
    guard let unspecified = artifact.arrays[unspecifiedName] else {
      throw BoltzError.missingTensor(unspecifiedName)
    }
    guard let unselected = artifact.arrays[unselectedName] else {
      throw BoltzError.missingTensor(unselectedName)
    }
    let template: TemplateModule?
    if configuration.useTemplates {
      guard let templateConfiguration = configuration.template else {
        throw BoltzError.missingFile("template_args in config.json")
      }
      template = try templateModule(
        configuration: templateConfiguration,
        pairWidth: configuration.tokenZ,
        useV2: configuration.useTemplatesV2
      )
    } else {
      template = nil
    }
    return try BoltzTrunk(
      inputEmbedder: inputEmbedder(configuration: configuration),
      sequenceInitial: linear("s_init"),
      pairInitialFirst: linear("z_init_1"),
      pairInitialSecond: linear("z_init_2"),
      relativePosition: RelativePositionEncoder(
        projection: linear("rel_pos.linear_layer"),
        fixSymCheck: configuration.fixSymCheck,
        cyclicPositionEncoding: configuration.cyclicPosEnc
      ),
      tokenBonds: linear("token_bonds"),
      tokenBondType: configuration.bondTypeFeature ? embedding("token_bonds_type") : nil,
      contactConditioning: ContactConditioning(
        fourierProjection: linear("contact_conditioning.fourier_embedding.proj"),
        encoder: linear("contact_conditioning.encoder"),
        unspecifiedEncoding: unspecified,
        unselectedEncoding: unselected,
        cutoffMinimum: configuration.conditioningCutoffMin,
        cutoffMaximum: configuration.conditioningCutoffMax
      ),
      sequenceNorm: layerNorm("s_norm"),
      pairNorm: layerNorm("z_norm"),
      sequenceRecycle: linear("s_recycle"),
      pairRecycle: linear("z_recycle"),
      templateModule: template,
      msaModule: msaModule(
        configuration: configuration.msa,
        pairWidth: configuration.tokenZ
      ),
      pairformer: pairformer(
        configuration: configuration.pairformer,
        sequenceWidth: configuration.tokenS,
        pairWidth: configuration.tokenZ
      )
    )
  }
}
