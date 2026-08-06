import MLX

/// MSA outer-product mean projected into pair features.
public struct OuterProductMean {
  public let norm: BoltzLayerNorm
  public let first: AffineLinear
  public let second: AffineLinear
  public let output: AffineLinear

  public init(
    norm: BoltzLayerNorm,
    first: AffineLinear,
    second: AffineLinear,
    output: AffineLinear
  ) {
    self.norm = norm
    self.first = first
    self.second = second
    self.output = output
  }

  public func callAsFunction(_ msa: MLXArray, mask: MLXArray) -> MLXArray {
    let normalized = norm(msa)
    return output(outerProductMean(first(normalized), second(normalized), mask: mask))
  }
}
