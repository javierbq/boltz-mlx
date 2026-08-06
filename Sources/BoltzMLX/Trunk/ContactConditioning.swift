import MLX

/// Encodes user-supplied Boltz contact constraints into the pair representation.
public struct ContactConditioning {
  public let fourierProjection: AffineLinear
  public let encoder: AffineLinear
  public let unspecifiedEncoding: MLXArray
  public let unselectedEncoding: MLXArray
  public let cutoffMinimum: Float
  public let cutoffMaximum: Float

  public init(
    fourierProjection: AffineLinear,
    encoder: AffineLinear,
    unspecifiedEncoding: MLXArray,
    unselectedEncoding: MLXArray,
    cutoffMinimum: Float,
    cutoffMaximum: Float
  ) {
    self.fourierProjection = fourierProjection
    self.encoder = encoder
    self.unspecifiedEncoding = unspecifiedEncoding
    self.unselectedEncoding = unselectedEncoding
    self.cutoffMinimum = cutoffMinimum
    self.cutoffMaximum = cutoffMaximum
  }

  public func callAsFunction(_ features: [String: MLXArray]) throws -> MLXArray {
    let flags = try requireFeature("contact_conditioning", from: features)
    let threshold = try requireFeature("contact_threshold", from: features)
    let normalizedThreshold =
      (threshold - cutoffMinimum) / (cutoffMaximum - cutoffMinimum)
    let fourier = MLX.cos(
      Float.pi * 2
        * fourierProjection(normalizedThreshold.expandedDimensions(axis: -1))
    )
    let learned = encoder(
      MLX.concatenated(
        [
          flags[.ellipsis, 2...],
          normalizedThreshold.expandedDimensions(axis: -1),
          fourier,
        ],
        axis: -1
      )
    )
    return applyContactSelections(
      encoded: learned,
      contactFlags: flags,
      unspecified: unspecifiedEncoding,
      unselected: unselectedEncoding
    )
  }
}

func applyContactSelections(
  encoded: MLXArray,
  contactFlags: MLXArray,
  unspecified: MLXArray,
  unselected: MLXArray
) -> MLXArray {
  let firstFlags = contactFlags[.ellipsis, ..<2]
  let learnedMask = 1 - firstFlags.sum(axis: -1, keepDims: true)
  return encoded * learnedMask
    + unspecified * contactFlags[.ellipsis, 0..<1]
    + unselected * contactFlags[.ellipsis, 1..<2]
}
