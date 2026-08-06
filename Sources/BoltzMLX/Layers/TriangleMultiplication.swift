import MLX

/// Incoming or outgoing triangular multiplicative update for pair features.
public struct TriangleMultiplication {
  public let direction: TriangleDirection
  public let inputNorm: BoltzLayerNorm
  public let inputProjection: AffineLinear
  public let inputGate: AffineLinear
  public let outputNorm: BoltzLayerNorm
  public let outputProjection: AffineLinear
  public let outputGate: AffineLinear

  public init(
    direction: TriangleDirection,
    inputNorm: BoltzLayerNorm,
    inputProjection: AffineLinear,
    inputGate: AffineLinear,
    outputNorm: BoltzLayerNorm,
    outputProjection: AffineLinear,
    outputGate: AffineLinear
  ) {
    self.direction = direction
    self.inputNorm = inputNorm
    self.inputProjection = inputProjection
    self.inputGate = inputGate
    self.outputNorm = outputNorm
    self.outputProjection = outputProjection
    self.outputGate = outputGate
  }

  public func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
    let normalized = inputNorm(input)
    var projected = inputProjection(normalized) * MLX.sigmoid(inputGate(normalized))
    projected = projected * mask.expandedDimensions(axis: -1)
    let parts = projected.asType(.float32).split(parts: 2, axis: -1)
    let triangle = triangleProjection(parts[0], parts[1], direction: direction)
    return outputProjection(outputNorm(triangle)) * MLX.sigmoid(outputGate(normalized))
  }
}
