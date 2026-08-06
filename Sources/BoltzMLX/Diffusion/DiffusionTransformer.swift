import MLX
import MLXNN

/// Adaptive layer normalization used in all conditioned diffusion blocks.
public struct AdaptiveLayerNorm {
  public let conditioningNorm: BoltzLayerNorm
  public let scale: AffineLinear
  public let shift: AffineLinear

  public func callAsFunction(
    activation: MLXArray,
    conditioning: MLXArray
  ) -> MLXArray {
    let normalizedConditioning = conditioningNorm(conditioning)
    return MLX.sigmoid(scale(normalizedConditioning)) * layerNormalized(activation)
      + shift(normalizedConditioning)
  }
}

/// SwiGLU transition whose residual strength is conditioned per token.
public struct ConditionedTransition {
  public let adaptiveNorm: AdaptiveLayerNorm
  public let swishGate: AffineLinear
  public let value: AffineLinear
  public let output: AffineLinear
  public let outputGate: AffineLinear

  public func callAsFunction(
    activation: MLXArray,
    conditioning: MLXArray
  ) -> MLXArray {
    let normalized = adaptiveNorm(activation: activation, conditioning: conditioning)
    let gateParts = swishGate(normalized).split(parts: 2, axis: -1)
    let gated = gateParts[0] * MLXNN.silu(gateParts[1]) * value(normalized)
    return MLX.sigmoid(outputGate(conditioning)) * output(gated)
  }
}

/// One token- or atom-level conditioned transformer layer.
public struct DiffusionTransformerLayer {
  public let adaptiveNorm: AdaptiveLayerNorm
  public let attention: AttentionPairBias
  public let outputGate: AffineLinear
  public let transition: ConditionedTransition
  public let postNorm: BoltzLayerNorm?

  public func callAsFunction(
    activation: MLXArray,
    conditioning: MLXArray,
    bias: MLXArray,
    mask: MLXArray,
    multiplicity: Int = 1,
    toKeys: ((MLXArray) -> MLXArray)? = nil
  ) -> MLXArray {
    let normalized = adaptiveNorm(activation: activation, conditioning: conditioning)
    let keyInput: MLXArray
    let keyMask: MLXArray
    if let toKeys {
      keyInput = toKeys(normalized)
      keyMask = toKeys(mask.expandedDimensions(axis: -1)).squeezed(axis: -1)
    } else {
      keyInput = normalized
      keyMask = mask
    }
    let attended = attention(
      sequence: normalized,
      pair: bias,
      mask: keyMask,
      keyInput: keyInput,
      multiplicity: multiplicity
    )
    var output = activation + MLX.sigmoid(outputGate(conditioning)) * attended
    output = output + transition(activation: output, conditioning: conditioning)
    if let postNorm {
      output = postNorm(output)
    }
    return output
  }
}

/// Repeated conditioned transformer with one pair-bias slice per layer.
public struct DiffusionTransformer {
  public let layers: [DiffusionTransformerLayer]

  public func callAsFunction(
    activation: MLXArray,
    conditioning: MLXArray,
    bias: MLXArray,
    mask: MLXArray,
    multiplicity: Int = 1,
    toKeys: ((MLXArray) -> MLXArray)? = nil
  ) -> MLXArray {
    let batch = bias.shape[0]
    let queries = bias.shape[1]
    let keys = bias.shape[2]
    let splitBias = bias.reshaped(batch, queries, keys, layers.count, -1)
    var output = activation
    for (index, layer) in layers.enumerated() {
      output = layer(
        activation: output,
        conditioning: conditioning,
        bias: splitBias[0..., 0..., 0..., index, 0...],
        mask: mask,
        multiplicity: multiplicity,
        toKeys: toKeys
      )
    }
    return output
  }
}

extension BoltzWeightStore {
  func biaslessLayerNorm(_ prefix: String) throws -> BoltzLayerNorm {
    let weightName = "\(prefix).weight"
    guard let weight = artifact.arrays[weightName] else {
      throw BoltzError.missingTensor(weightName)
    }
    return BoltzLayerNorm(weight: weight, bias: nil)
  }

  func adaptiveLayerNorm(_ prefix: String) throws -> AdaptiveLayerNorm {
    try AdaptiveLayerNorm(
      conditioningNorm: biaslessLayerNorm("\(prefix).s_norm"),
      scale: linear("\(prefix).s_scale"),
      shift: linear("\(prefix).s_bias")
    )
  }

  func conditionedTransition(_ prefix: String) throws -> ConditionedTransition {
    try ConditionedTransition(
      adaptiveNorm: adaptiveLayerNorm("\(prefix).adaln"),
      swishGate: linear("\(prefix).swish_gate.0"),
      value: linear("\(prefix).a_to_b"),
      output: linear("\(prefix).b_to_a"),
      outputGate: linear("\(prefix).output_projection.0")
    )
  }

  func diffusionAttention(
    _ prefix: String,
    width: Int,
    headCount: Int
  ) throws -> AttentionPairBias {
    try AttentionPairBias(
      sequenceWidth: width,
      headCount: headCount,
      query: linear("\(prefix).proj_q"),
      key: linear("\(prefix).proj_k"),
      value: linear("\(prefix).proj_v"),
      gate: linear("\(prefix).proj_g"),
      pairNorm: nil,
      pairBias: nil,
      output: linear("\(prefix).proj_o")
    )
  }

  func diffusionTransformer(
    _ prefix: String,
    depth: Int,
    width: Int,
    headCount: Int,
    postLayerNorm: Bool = false
  ) throws -> DiffusionTransformer {
    var layers: [DiffusionTransformerLayer] = []
    for index in 0..<depth {
      let layerPrefix = "\(prefix).layers.\(index)"
      layers.append(
        try DiffusionTransformerLayer(
          adaptiveNorm: adaptiveLayerNorm("\(layerPrefix).adaln"),
          attention: diffusionAttention(
            "\(layerPrefix).pair_bias_attn",
            width: width,
            headCount: headCount
          ),
          outputGate: linear("\(layerPrefix).output_projection_linear"),
          transition: conditionedTransition("\(layerPrefix).transition"),
          postNorm: postLayerNorm ? layerNorm("\(layerPrefix).post_lnorm") : nil
        )
      )
    }
    return DiffusionTransformer(layers: layers)
  }
}
