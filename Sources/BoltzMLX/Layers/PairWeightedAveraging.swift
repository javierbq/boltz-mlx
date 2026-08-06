import MLX

/// Pair-biased averaging of MSA rows with learned gating.
public struct PairWeightedAveraging {
  public let hiddenWidth: Int
  public let headCount: Int
  public let msaNorm: BoltzLayerNorm
  public let pairNorm: BoltzLayerNorm
  public let value: AffineLinear
  public let gate: AffineLinear
  public let pairBias: AffineLinear
  public let output: AffineLinear

  public init(
    hiddenWidth: Int,
    headCount: Int,
    msaNorm: BoltzLayerNorm,
    pairNorm: BoltzLayerNorm,
    value: AffineLinear,
    gate: AffineLinear,
    pairBias: AffineLinear,
    output: AffineLinear
  ) {
    self.hiddenWidth = hiddenWidth
    self.headCount = headCount
    self.msaNorm = msaNorm
    self.pairNorm = pairNorm
    self.value = value
    self.gate = gate
    self.pairBias = pairBias
    self.output = output
  }

  public func callAsFunction(_ msa: MLXArray, pair: MLXArray, mask: MLXArray) -> MLXArray {
    let normalizedMSA = msaNorm(msa)
    let normalizedPair = pairNorm(pair)
    let batch = msa.shape[0]
    let sequences = msa.shape[1]
    let tokens = msa.shape[2]
    var projectedValue = value(normalizedMSA).reshaped(
      batch, sequences, tokens, headCount, hiddenWidth
    )
    projectedValue = projectedValue.transposed(0, 3, 1, 2, 4)
    var bias = pairBias(normalizedPair).transposed(0, 3, 1, 2)
    bias = bias + (1 - mask.expandedDimensions(axis: 1)) * -1e6
    let weights = MLX.softmax(bias, axis: -1)
    var averaged = MLX.einsum("bhij,bhsjd->bhsid", weights, projectedValue)
    averaged = averaged.transposed(0, 2, 3, 1, 4)
      .reshaped(batch, sequences, tokens, headCount * hiddenWidth)
    return output(MLX.sigmoid(gate(normalizedMSA)) * averaged)
  }
}
