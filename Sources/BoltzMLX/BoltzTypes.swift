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
