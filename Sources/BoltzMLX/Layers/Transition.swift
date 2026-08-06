import MLX
import MLXNN

/// Gated two-branch transition used throughout the Boltz trunk.
public struct Transition {
  public let norm: BoltzLayerNorm
  public let first: AffineLinear
  public let gate: AffineLinear
  public let output: AffineLinear

  public init(
    norm: BoltzLayerNorm,
    first: AffineLinear,
    gate: AffineLinear,
    output: AffineLinear
  ) {
    self.norm = norm
    self.first = first
    self.gate = gate
    self.output = output
  }

  public func callAsFunction(_ input: MLXArray) -> MLXArray {
    let normalized = norm(input)
    return output(MLXNN.silu(first(normalized)) * gate(normalized))
  }
}
