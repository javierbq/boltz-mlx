import Foundation

/// Sampling controls shared by the macOS and iOS runtimes.
public struct BoltzPredictionOptions: Sendable, Equatable {
  public var recyclingSteps: Int
  public var diffusionSteps: Int
  public var seed: UInt64

  public init(recyclingSteps: Int = 0, diffusionSteps: Int = 20, seed: UInt64 = 0) {
    self.recyclingSteps = recyclingSteps
    self.diffusionSteps = diffusionSteps
    self.seed = seed
  }
}

/// A progress report emitted from inside a prediction.
///
/// Reported ONLY from loops whose iteration ends in an `MLX.eval`, because MLX is
/// lazy: a callback fired where nothing has been materialised would race ahead of
/// the real work and then stall. That is exactly two places -- the trunk's
/// recycling loop and the diffusion sampler -- and deliberately NOT the 64-block
/// Pairformer loop, which has no eval and would emit 64 reports in milliseconds
/// before going quiet for the rest of the pass.
///
/// `completed` counts finished iterations, so the first report is 1, not 0, and
/// the last equals `total`.
public struct BoltzProgress: Sendable, Equatable {
  public enum Stage: String, Sendable, Equatable {
    /// Trunk recycling. `total` is `recyclingSteps + 1` -- the loop runs
    /// `0...recyclingSteps`, so a request for 3 recycles is 4 passes.
    case trunk
    /// Diffusion sampling. `total` is the resolved step count.
    case diffusion
  }

  public let stage: Stage
  public let completed: Int
  public let total: Int

  public init(stage: Stage, completed: Int, total: Int) {
    self.stage = stage
    self.completed = completed
    self.total = total
  }

  /// Completion within this stage, 0...1. Zero when `total` is not positive,
  /// rather than a division by zero.
  public var fraction: Double {
    total > 0 ? Double(completed) / Double(total) : 0
  }
}

/// Allocation limits applied before pairwise tensors are constructed.
public struct BoltzInputLimits: Sendable, Equatable {
  public var maximumTokens: Int
  public var maximumAtoms: Int
  public var maximumMSADepth: Int

  public init(maximumTokens: Int, maximumAtoms: Int, maximumMSADepth: Int) {
    self.maximumTokens = maximumTokens
    self.maximumAtoms = maximumAtoms
    self.maximumMSADepth = maximumMSADepth
  }

  /// Desktop limits: a real target with a real alignment.
  ///
  /// `MemoryPlanner`'s own default is deliberately phone-sized (256 tokens, 1024 MSA rows) and would
  /// refuse a typical alignment — a jackhmmer or ColabFold a3m for a 400-residue target routinely
  /// runs to thousands of rows. `maximumMSADepth` here is upstream's `const.max_msa_seqs`, which is
  /// also the cap `MSAPairing` applies, so this admits any alignment upstream itself would accept
  /// and no more.
  ///
  /// The estimated-activation check in `MemoryPlanner` still applies, and it scales with
  /// depth × tokens — so a pathologically deep alignment is refused on the memory estimate rather
  /// than on the depth, which is the honest place to refuse it.
  public static let desktop = BoltzInputLimits(
    maximumTokens: 1_024, maximumAtoms: 16_384, maximumMSADepth: 16_384)
}

/// Unpadded atom coordinates returned by the structure-only model.
public struct BoltzStructure: Sendable, Equatable {
  public let coordinates: [SIMD3<Float>]
  public let atomMask: [Bool]

  public init(coordinates: [SIMD3<Float>], atomMask: [Bool]) {
    self.coordinates = coordinates
    self.atomMask = atomMask
  }
}
