import MLX

/// Multi-head attention with a learned pairwise bias and output gate.
public struct AttentionPairBias {
  public let sequenceWidth: Int
  public let headCount: Int
  public let query: AffineLinear
  public let key: AffineLinear
  public let value: AffineLinear
  public let gate: AffineLinear
  public let pairNorm: BoltzLayerNorm?
  public let pairBias: AffineLinear?
  public let output: AffineLinear

  public init(
    sequenceWidth: Int,
    headCount: Int,
    query: AffineLinear,
    key: AffineLinear,
    value: AffineLinear,
    gate: AffineLinear,
    pairNorm: BoltzLayerNorm?,
    pairBias: AffineLinear?,
    output: AffineLinear
  ) {
    precondition(sequenceWidth.isMultiple(of: headCount))
    self.sequenceWidth = sequenceWidth
    self.headCount = headCount
    self.query = query
    self.key = key
    self.value = value
    self.gate = gate
    self.pairNorm = pairNorm
    self.pairBias = pairBias
    self.output = output
  }

  public func callAsFunction(
    sequence: MLXArray,
    pair: MLXArray,
    mask: MLXArray,
    keyInput: MLXArray,
    multiplicity: Int = 1
  ) -> MLXArray {
    let batch = sequence.shape[0]
    let headWidth = sequenceWidth / headCount
    let q = query(sequence).reshaped(batch, -1, headCount, headWidth)
    let k = key(keyInput).reshaped(batch, -1, headCount, headWidth)
    let v = value(keyInput).reshaped(batch, -1, headCount, headWidth)
    let projectedPair: MLXArray
    if let pairNorm, let pairBias {
      projectedPair = pairBias(pairNorm(pair))
    } else {
      projectedPair = pair
    }
    var bias = projectedPair.transposed(0, 3, 1, 2)
    if multiplicity > 1 {
      bias = MLX.repeated(bias, count: multiplicity, axis: 0)
    }
    let attended = attentionWithPairBias(
      query: q,
      key: k,
      value: v,
      bias: bias,
      keyMask: mask,
      headDimension: headWidth
    ).reshaped(batch, -1, sequenceWidth)
    return output(MLX.sigmoid(gate(sequence)) * attended)
  }
}
