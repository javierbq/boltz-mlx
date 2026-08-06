import MLX

/// Conservative preflight and MLX cache policy for memory-constrained iOS inference.
public struct MemoryPlanner: Sendable {
  public let limits: BoltzInputLimits
  public let memoryLimit: Int
  public let cacheLimit: Int

  public init(
    limits: BoltzInputLimits = BoltzInputLimits(
      maximumTokens: 256,
      maximumAtoms: 2_048,
      maximumMSADepth: 1_024
    ),
    memoryLimit: Int = 6 * 1_024 * 1_024 * 1_024,
    cacheLimit: Int = 64 * 1_024 * 1_024
  ) {
    self.limits = limits
    self.memoryLimit = memoryLimit
    self.cacheLimit = cacheLimit
  }

  public func validate(metadata: FeatureMetadata) throws {
    try check(
      "tokens",
      found: metadata.tokenCount,
      maximum: limits.maximumTokens
    )
    try check(
      "atoms",
      found: metadata.atomCount,
      maximum: limits.maximumAtoms
    )
    try check(
      "msa_depth",
      found: metadata.msaDepth,
      maximum: limits.maximumMSADepth
    )
    let estimate = estimatedActivationBytes(metadata: metadata)
    if estimate > memoryLimit {
      throw BoltzError.inputLimitExceeded(
        dimension: "estimated_activation_bytes",
        found: estimate,
        maximum: memoryLimit
      )
    }
  }

  public func apply() {
    Memory.cacheLimit = cacheLimit
    Memory.memoryLimit = memoryLimit
  }

  public func estimatedActivationBytes(metadata: FeatureMetadata) -> Int {
    let token = metadata.tokenCount
    let atom = metadata.atomCount
    let msa = metadata.msaDepth
    let pairWorkingSet = token * token * 128 * 4 * 8
    let msaWorkingSet = msa * token * 64 * 2 * 4
    let atomWorkingSet = atom * 128 * 4 * 12
    return pairWorkingSet + msaWorkingSet + atomWorkingSet
  }

  private func check(_ dimension: String, found: Int, maximum: Int) throws {
    if found > maximum {
      throw BoltzError.inputLimitExceeded(
        dimension: dimension,
        found: found,
        maximum: maximum
      )
    }
  }
}
