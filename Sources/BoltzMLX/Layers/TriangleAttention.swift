import MLX

/// Starting- or ending-node triangular attention from the Boltz pair stack.
public struct TriangleAttention {
  public let startingNode: Bool
  public let pairWidth: Int
  public let headWidth: Int
  public let headCount: Int
  public let inputNorm: BoltzLayerNorm
  public let triangleBias: AffineLinear
  public let query: AffineLinear
  public let key: AffineLinear
  public let value: AffineLinear
  public let gate: AffineLinear
  public let output: AffineLinear

  public init(
    startingNode: Bool,
    pairWidth: Int,
    headWidth: Int,
    headCount: Int,
    inputNorm: BoltzLayerNorm,
    triangleBias: AffineLinear,
    query: AffineLinear,
    key: AffineLinear,
    value: AffineLinear,
    gate: AffineLinear,
    output: AffineLinear
  ) {
    self.startingNode = startingNode
    self.pairWidth = pairWidth
    self.headWidth = headWidth
    self.headCount = headCount
    self.inputNorm = inputNorm
    self.triangleBias = triangleBias
    self.query = query
    self.key = key
    self.value = value
    self.gate = gate
    self.output = output
  }

  public func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
    let orientedInput = startingNode ? input : input.transposed(0, 2, 1, 3)
    let orientedMask = startingNode ? mask : mask.transposed(0, 2, 1)
    let normalized = inputNorm(orientedInput)
    let batch = normalized.shape[0]
    let rows = normalized.shape[1]
    let columns = normalized.shape[2]

    let q = query(normalized).reshaped(batch, rows, columns, headCount, headWidth)
    let k = key(normalized).reshaped(batch, rows, columns, headCount, headWidth)
    let v = value(normalized).reshaped(batch, rows, columns, headCount, headWidth)
    let bias = triangleBias(normalized).transposed(0, 3, 1, 2)
      .expandedDimensions(axis: 1)
    var attended = triangleAttentionCore(
      query: q,
      key: k,
      value: v,
      bias: bias,
      keyMask: orientedMask,
      headDimension: headWidth
    )
    let gated = MLX.sigmoid(gate(normalized)).reshaped(
      batch,
      rows,
      columns,
      headCount,
      headWidth
    )
    // `attended * gated` is [batch, rows, columns, headCount, headWidth]; the
    // output projection maps headCount * headWidth back to pairWidth, so flatten
    // to the head-hidden width (which need not equal pairWidth, e.g. the
    // template pair stack where token_z=64 but headCount*headWidth=128).
    attended = output((attended * gated).reshaped(batch, rows, columns, headCount * headWidth))
    return startingNode ? attended : attended.transposed(0, 2, 1, 3)
  }
}

func triangleAttentionCore(
  query: MLXArray,
  key: MLXArray,
  value: MLXArray,
  bias: MLXArray,
  keyMask: MLXArray,
  headDimension: Int,
  infinity: Float = 1e9
) -> MLXArray {
  var scores = MLX.einsum(
    "brihd,brjhd->brhij",
    query.asType(.float32),
    key.asType(.float32)
  )
  scores = scores / Float(headDimension).squareRoot() + bias.asType(.float32)
  let mask = keyMask.expandedDimensions(axes: [2, 3]).asType(.float32)
  scores = scores + (1 - mask) * -infinity
  let probabilities = MLX.softmax(scores, axis: -1)
  return MLX.einsum(
    "brhij,brjhd->brihd",
    probabilities,
    value.asType(.float32)
  ).asType(value.dtype)
}
