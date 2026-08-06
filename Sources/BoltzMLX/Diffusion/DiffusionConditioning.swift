import MLX

/// Residual pair conditioning applied before the score network.
public struct PairwiseConditioning {
  public let norm: BoltzLayerNorm
  public let projection: AffineLinear
  public let transitions: [Transition]

  public func callAsFunction(pair: MLXArray, relativePosition: MLXArray) -> MLXArray {
    var output = projection(norm(MLX.concatenated([pair, relativePosition], axis: -1)))
    for transition in transitions {
      output = output + transition(output)
    }
    return output
  }
}

/// Values reused at every diffusion step for a fixed trunk result.
public struct DiffusionConditioningOutput {
  public let query: MLXArray
  public let conditioning: MLXArray
  public let windowing: AtomWindowing
  public let atomEncoderBias: MLXArray
  public let atomDecoderBias: MLXArray
  public let tokenTransformerBias: MLXArray
}

/// Precomputes the atom and token attention biases consumed by the score model.
public struct DiffusionConditioning {
  public let pairwise: PairwiseConditioning
  public let atomEncoder: AtomEncoder
  public let atomEncoderBiasNorms: [BoltzLayerNorm]
  public let atomEncoderBiasProjections: [AffineLinear]
  public let atomDecoderBiasNorms: [BoltzLayerNorm]
  public let atomDecoderBiasProjections: [AffineLinear]
  public let tokenBiasNorms: [BoltzLayerNorm]
  public let tokenBiasProjections: [AffineLinear]

  public func callAsFunction(
    trunk: TrunkOutput,
    features: [String: MLXArray]
  ) throws -> DiffusionConditioningOutput {
    let pair = pairwise(pair: trunk.pair, relativePosition: trunk.relativePosition)
    let atom = try atomEncoder(
      features: features,
      sequence: trunk.sequence,
      pair: pair
    )
    return DiffusionConditioningOutput(
      query: atom.query,
      conditioning: atom.conditioning,
      windowing: atom.windowing,
      atomEncoderBias: projectAndConcatenate(
        atom.pair,
        norms: atomEncoderBiasNorms,
        projections: atomEncoderBiasProjections
      ),
      atomDecoderBias: projectAndConcatenate(
        atom.pair,
        norms: atomDecoderBiasNorms,
        projections: atomDecoderBiasProjections
      ),
      tokenTransformerBias: projectAndConcatenate(
        pair,
        norms: tokenBiasNorms,
        projections: tokenBiasProjections
      )
    )
  }
}

private func projectAndConcatenate(
  _ input: MLXArray,
  norms: [BoltzLayerNorm],
  projections: [AffineLinear]
) -> MLXArray {
  precondition(norms.count == projections.count)
  return MLX.concatenated(
    zip(norms, projections).map { norm, projection in
      projection(norm(input))
    },
    axis: -1
  )
}

extension BoltzWeightStore {
  func pairwiseConditioning(
    _ prefix: String,
    transitionCount: Int
  ) throws -> PairwiseConditioning {
    var transitions: [Transition] = []
    for index in 0..<transitionCount {
      transitions.append(try transition("\(prefix).transitions.\(index)"))
    }
    return try PairwiseConditioning(
      norm: layerNorm("\(prefix).dim_pairwise_init_proj.0"),
      projection: linear("\(prefix).dim_pairwise_init_proj.1"),
      transitions: transitions
    )
  }

  func normProjectionLists(
    _ prefix: String,
    count: Int
  ) throws -> (norms: [BoltzLayerNorm], projections: [AffineLinear]) {
    var norms: [BoltzLayerNorm] = []
    var projections: [AffineLinear] = []
    for index in 0..<count {
      norms.append(try layerNorm("\(prefix).\(index).0"))
      projections.append(try linear("\(prefix).\(index).1"))
    }
    return (norms, projections)
  }

  func diffusionConditioning(
    configuration: BoltzModelConfiguration
  ) throws -> DiffusionConditioning {
    let prefix = "diffusion_conditioning"
    let atomEncoderBias = try normProjectionLists(
      "\(prefix).atom_enc_proj_z",
      count: configuration.scoreModel.atomEncoderDepth
    )
    let atomDecoderBias = try normProjectionLists(
      "\(prefix).atom_dec_proj_z",
      count: configuration.scoreModel.atomDecoderDepth
    )
    let tokenBias = try normProjectionLists(
      "\(prefix).token_trans_proj_z",
      count: configuration.scoreModel.tokenTransformerDepth
    )
    return try DiffusionConditioning(
      pairwise: pairwiseConditioning(
        "\(prefix).pairwise_conditioner",
        transitionCount: configuration.scoreModel.conditioningTransitionLayers
      ),
      atomEncoder: atomEncoder(
        "\(prefix).atom_encoder",
        queryWindow: configuration.atomsPerWindowQueries,
        keyWindow: configuration.atomsPerWindowKeys,
        structurePrediction: true
      ),
      atomEncoderBiasNorms: atomEncoderBias.norms,
      atomEncoderBiasProjections: atomEncoderBias.projections,
      atomDecoderBiasNorms: atomDecoderBias.norms,
      atomDecoderBiasProjections: atomDecoderBias.projections,
      tokenBiasNorms: tokenBias.norms,
      tokenBiasProjections: tokenBias.projections
    )
  }
}
