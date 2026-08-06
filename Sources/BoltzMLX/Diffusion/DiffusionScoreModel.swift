import MLX

/// Fixed random Fourier features stored in the Boltz checkpoint.
public struct FourierEmbedding {
  public let projection: AffineLinear

  public func callAsFunction(_ values: MLXArray) -> MLXArray {
    MLX.cos(Float.pi * 2 * projection(values.expandedDimensions(axis: -1)))
  }
}

/// Noise- and trunk-conditioned token representation for one score evaluation.
public struct SingleConditioning {
  public let inputNorm: BoltzLayerNorm
  public let inputProjection: AffineLinear
  public let fourier: FourierEmbedding
  public let fourierNorm: BoltzLayerNorm
  public let fourierProjection: AffineLinear
  public let transitions: [Transition]

  public func callAsFunction(
    noise: MLXArray,
    trunkSequence: MLXArray,
    inputSequence: MLXArray
  ) -> MLXArray {
    var sequence = inputProjection(
      inputNorm(MLX.concatenated([trunkSequence, inputSequence], axis: -1))
    )
    let noiseEmbedding = fourierProjection(fourierNorm(fourier(noise)))
    sequence = sequence + noiseEmbedding.expandedDimensions(axis: 1)
    for transition in transitions {
      sequence = sequence + transition(sequence)
    }
    return sequence
  }
}

/// Broadcasts token activations to atoms and predicts coordinate updates.
public struct AtomAttentionDecoder {
  public let tokenToAtom: AffineLinear
  public let atomTransformer: AtomTransformer
  public let outputNorm: BoltzLayerNorm
  public let positionUpdate: AffineLinear

  public func callAsFunction(
    token: MLXArray,
    query: MLXArray,
    conditioning: MLXArray,
    bias: MLXArray,
    features: [String: MLXArray],
    windowing: AtomWindowing,
    multiplicity: Int
  ) throws -> MLXArray {
    var atomToToken = try requireFeature("atom_to_token", from: features).asType(.float32)
    atomToToken = MLX.repeated(atomToToken, count: multiplicity, axis: 0)
    var q = query + MLX.matmul(atomToToken, tokenToAtom(token.asType(.float32)))
    let mask = MLX.repeated(
      try requireFeature("atom_pad_mask", from: features),
      count: multiplicity,
      axis: 0
    )
    q = atomTransformer(
      query: q,
      mask: mask,
      conditioning: conditioning,
      bias: bias,
      multiplicity: multiplicity,
      windowing: windowing
    )
    return positionUpdate(outputNorm(q))
  }
}

/// Complete Boltz-2 diffusion score network for one noise level.
public struct DiffusionScoreModel {
  public let singleConditioning: SingleConditioning
  public let atomEncoder: AtomAttentionEncoder
  public let sequenceToTokenNorm: BoltzLayerNorm
  public let sequenceToToken: AffineLinear
  public let tokenTransformer: DiffusionTransformer
  public let tokenNorm: BoltzLayerNorm
  public let atomDecoder: AtomAttentionDecoder

  public func callAsFunction(
    noisyCoordinates: MLXArray,
    noise: MLXArray,
    trunk: TrunkOutput,
    features: [String: MLXArray],
    conditioning: DiffusionConditioningOutput,
    multiplicity: Int
  ) throws -> MLXArray {
    let repeatedTrunk = MLX.repeated(trunk.sequence, count: multiplicity, axis: 0)
    let repeatedInput = MLX.repeated(trunk.sequenceInput, count: multiplicity, axis: 0)
    let sequenceConditioning = singleConditioning(
      noise: noise,
      trunkSequence: repeatedTrunk,
      inputSequence: repeatedInput
    )
    let atoms = try atomEncoder(
      features: features,
      query: conditioning.query.asType(.float32),
      conditioning: conditioning.conditioning.asType(.float32),
      bias: conditioning.atomEncoderBias.asType(.float32),
      windowing: conditioning.windowing,
      coordinates: noisyCoordinates,
      multiplicity: multiplicity
    )
    var token = atoms.token + sequenceToToken(sequenceToTokenNorm(sequenceConditioning))
    let tokenMask = MLX.repeated(
      try requireFeature("token_pad_mask", from: features),
      count: multiplicity,
      axis: 0
    ).asType(.float32)
    token = tokenTransformer(
      activation: token,
      conditioning: sequenceConditioning,
      bias: conditioning.tokenTransformerBias.asType(.float32),
      mask: tokenMask,
      multiplicity: multiplicity
    )
    token = tokenNorm(token)
    return try atomDecoder(
      token: token,
      query: atoms.query,
      conditioning: atoms.conditioning,
      bias: conditioning.atomDecoderBias.asType(.float32),
      features: features,
      windowing: atoms.windowing,
      multiplicity: multiplicity
    )
  }
}

extension BoltzWeightStore {
  func singleConditioning(
    _ prefix: String,
    transitionCount: Int
  ) throws -> SingleConditioning {
    var transitions: [Transition] = []
    for index in 0..<transitionCount {
      transitions.append(try transition("\(prefix).transitions.\(index)"))
    }
    return try SingleConditioning(
      inputNorm: layerNorm("\(prefix).norm_single"),
      inputProjection: linear("\(prefix).single_embed"),
      fourier: FourierEmbedding(projection: linear("\(prefix).fourier_embed.proj")),
      fourierNorm: layerNorm("\(prefix).norm_fourier"),
      fourierProjection: linear("\(prefix).fourier_to_single"),
      transitions: transitions
    )
  }

  func atomAttentionDecoder(
    _ prefix: String,
    depth: Int,
    headCount: Int,
    atomWidth: Int,
    queryWindow: Int,
    keyWindow: Int
  ) throws -> AtomAttentionDecoder {
    try AtomAttentionDecoder(
      tokenToAtom: linear("\(prefix).a_to_q_trans"),
      atomTransformer: atomTransformer(
        "\(prefix).atom_decoder",
        depth: depth,
        width: atomWidth,
        headCount: headCount,
        queryWindow: queryWindow,
        keyWindow: keyWindow
      ),
      outputNorm: layerNorm("\(prefix).atom_feat_to_atom_pos_update.0"),
      positionUpdate: linear("\(prefix).atom_feat_to_atom_pos_update.1")
    )
  }

  func diffusionScoreModel(
    configuration: BoltzModelConfiguration
  ) throws -> DiffusionScoreModel {
    let prefix = "structure_module.score_model"
    let score = configuration.scoreModel
    return try DiffusionScoreModel(
      singleConditioning: singleConditioning(
        "\(prefix).single_conditioner",
        transitionCount: score.conditioningTransitionLayers
      ),
      atomEncoder: atomAttentionEncoder(
        "\(prefix).atom_attention_encoder",
        depth: score.atomEncoderDepth,
        headCount: score.atomEncoderHeads,
        atomWidth: configuration.atomS,
        queryWindow: configuration.atomsPerWindowQueries,
        keyWindow: configuration.atomsPerWindowKeys,
        structurePrediction: true
      ),
      sequenceToTokenNorm: layerNorm("\(prefix).s_to_a_linear.0"),
      sequenceToToken: linear("\(prefix).s_to_a_linear.1"),
      tokenTransformer: diffusionTransformer(
        "\(prefix).token_transformer",
        depth: score.tokenTransformerDepth,
        width: configuration.tokenS * 2,
        headCount: score.tokenTransformerHeads
      ),
      tokenNorm: layerNorm("\(prefix).a_norm"),
      atomDecoder: atomAttentionDecoder(
        "\(prefix).atom_attention_decoder",
        depth: score.atomDecoderDepth,
        headCount: score.atomDecoderHeads,
        atomWidth: configuration.atomS,
        queryWindow: configuration.atomsPerWindowQueries,
        keyWindow: configuration.atomsPerWindowKeys
      )
    )
  }
}
