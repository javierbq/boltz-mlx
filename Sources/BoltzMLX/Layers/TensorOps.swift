import MLX

/// PyTorch-compatible feature-last layer normalization with learned affine terms.
public struct BoltzLayerNorm {
  public let weight: MLXArray
  public let bias: MLXArray?
  public let epsilon: Float

  public init(weight: MLXArray, bias: MLXArray?, epsilon: Float = 1e-5) {
    self.weight = weight
    self.bias = bias
    self.epsilon = epsilon
  }

  public func callAsFunction(_ input: MLXArray) -> MLXArray {
    let mean = input.mean(axis: -1, keepDims: true)
    let centered = input - mean
    let variance = (centered * centered).mean(axis: -1, keepDims: true)
    var output = centered * MLX.rsqrt(variance + epsilon) * weight
    if let bias {
      output = output + bias
    }
    return output
  }
}

func layerNormalized(_ input: MLXArray, epsilon: Float = 1e-5) -> MLXArray {
  let mean = input.mean(axis: -1, keepDims: true)
  let centered = input - mean
  let variance = (centered * centered).mean(axis: -1, keepDims: true)
  return centered * MLX.rsqrt(variance + epsilon)
}

func attentionWithPairBias(
  query: MLXArray,
  key: MLXArray,
  value: MLXArray,
  bias: MLXArray,
  keyMask: MLXArray,
  headDimension: Int,
  infinity: Float = 1e6
) -> MLXArray {
  var attention = MLX.einsum(
    "bihd,bjhd->bhij",
    query.asType(.float32),
    key.asType(.float32)
  )
  let scale = MLXArray(Float(headDimension).squareRoot())
  attention = attention / scale + bias.asType(.float32)
  let expandedMask = keyMask.expandedDimensions(axes: [1, 2]).asType(.float32)
  attention = attention + (1 - expandedMask) * -infinity
  attention = MLX.softmax(attention, axis: -1)
  return MLX.einsum(
    "bhij,bjhd->bihd",
    attention,
    value.asType(.float32)
  ).asType(value.dtype)
}

/// Direction of the pairwise triangular contraction.
public enum TriangleDirection: Sendable {
  case outgoing
  case incoming
}

func triangleProjection(
  _ a: MLXArray,
  _ b: MLXArray,
  direction: TriangleDirection
) -> MLXArray {
  switch direction {
  case .outgoing:
    MLX.einsum("bikd,bjkd->bijd", a, b)
  case .incoming:
    MLX.einsum("bkid,bkjd->bijd", a, b)
  }
}

func outerProductMean(_ a: MLXArray, _ b: MLXArray, mask: MLXArray) -> MLXArray {
  let expandedMask = mask.expandedDimensions(axis: -1).asType(a.dtype)
  let maskedA = a * expandedMask
  let maskedB = b * expandedMask
  let pairMask =
    expandedMask.expandedDimensions(axis: 2)
    * expandedMask.expandedDimensions(axis: 3)
  let count = MLX.maximum(pairMask.sum(axis: 1), 1)
  var output = MLX.einsum(
    "bsic,bsjd->bijcd",
    maskedA.asType(.float32),
    maskedB.asType(.float32)
  )
  let shape = output.shape
  output = output.reshaped(shape[0], shape[1], shape[2], shape[3] * shape[4])
  return output / count
}
