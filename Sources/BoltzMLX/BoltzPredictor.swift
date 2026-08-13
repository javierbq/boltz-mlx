import Foundation
import MLX

/// A folded structure together with its interface confidence.
public struct ScoredStructure: @unchecked Sendable {
  public let structure: BoltzStructure
  /// Row-major [tokenCount * tokenCount] expected aligned error, Angstroms.
  public let pae: [Double]
  /// [tokens] pLDDT in 0...100. Per TOKEN, which for a protein chain is per residue.
  public let plddt: [Double]
  public let tokenCount: Int
  public let chainTokenRanges: [(chain: String, range: Range<Int>)]

  /// min_ipSAE between the first two chains — the design-loop gate. nil for a single chain, since
  /// an interface score is undefined without two.
  public func interfaceScores(cutoff: Double = InterfaceScoring.defaultPAECutoff) -> InterfaceScores? {
    guard chainTokenRanges.count >= 2 else { return nil }
    return InterfaceScoring.score(paeMatrix: pae, tokenCount: tokenCount,
                                  a: chainTokenRanges[0].range, b: chainTokenRanges[1].range,
                                  cutoff: cutoff)
  }
}

/// Serializes access to the large native MLX graph and its mutable memory policy.
public actor BoltzPredictor {
  private let trunk: BoltzTrunk
  private let conditioning: DiffusionConditioning
  private let diffusion: AtomDiffusion
  private let confidence: ConfidenceModule?
  private let memoryPlanner: MemoryPlanner

  public init(
    modelDirectory: URL,
    memoryPlanner: MemoryPlanner = MemoryPlanner()
  ) throws {
    let artifact = try BoltzArtifact.load(from: modelDirectory)
    guard let configuration = artifact.configuration else {
      throw BoltzError.missingFile(modelDirectory.appending(path: "config.json").path)
    }
    let weights = BoltzWeightStore(artifact: artifact)
    self.trunk = try weights.trunk()
    self.conditioning = try weights.diffusionConditioning(configuration: configuration)
    self.diffusion = try weights.atomDiffusion(configuration: configuration)
    // nil for a structure-only pack, which stays valid — scoring is then unavailable rather than
    // the load failing.
    self.confidence = try weights.confidenceModule(configuration: configuration)
    self.memoryPlanner = memoryPlanner
  }

  public func predict(
    features: FeatureBundle,
    options: BoltzPredictionOptions = BoltzPredictionOptions()
  ) async throws -> BoltzStructure {
    try await predict(arrays: features.arrays, metadata: features.metadata, options: options)
  }

  /// Predict from features built in-process by `BoltzFeaturizer` — no Python, no torch, and no
  /// feature bundle on disk. This is the path a design loop uses: sequences in, structure out,
  /// entirely in Swift/MLX.
  public func predict(
    featurized: BoltzFeaturizer.Output,
    options: BoltzPredictionOptions = BoltzPredictionOptions(),
    matchedNoise: MatchedNoise? = nil
  ) async throws -> BoltzStructure {
    try await predict(arrays: featurized.features, metadata: featurized.metadata,
                      options: options, matchedNoise: matchedNoise)
  }

  /// Externally supplied reverse-diffusion noise, so a run can be made trajectory-comparable with
  /// another backend. Rotations/translations default to identity, matching an upstream run whose
  /// `compute_random_augmentation` has been patched out.
  public struct MatchedNoise: @unchecked Sendable {
    public let initial: MLXArray
    public let steps: [MLXArray]
    public init(initial: MLXArray, steps: [MLXArray]) {
      self.initial = initial
      self.steps = steps
    }

    /// Identity rotation per step, matching a Python side patched to identity.
    var identityRotations: [MLXArray] {
      let eye = MLXArray(
        [1, 0, 0, 0, 1, 0, 0, 0, 1].map { Float($0) }, [1, 3, 3])
      return Array(repeating: eye, count: steps.count)
    }

    /// Zero translation per step.
    var identityTranslations: [MLXArray] {
      Array(repeating: MLXArray([Float](repeating: 0, count: 3), [1, 1, 3]), count: steps.count)
    }
  }

  private func predict(
    arrays features: [String: MLXArray],
    metadata: FeatureMetadata,
    options: BoltzPredictionOptions,
    matchedNoise: MatchedNoise? = nil
  ) async throws -> BoltzStructure {
    try memoryPlanner.validate(metadata: metadata)
    memoryPlanner.apply()
    try Task.checkCancellation()
    let trunkOutput = try trunk(
      features: features,
      recyclingSteps: options.recyclingSteps
    )
    try Task.checkCancellation()
    let diffusionConditioning = try conditioning(
      trunk: trunkOutput,
      features: features
    )
    let coordinates = try diffusion.sample(
      trunk: trunkOutput,
      features: features,
      conditioning: diffusionConditioning,
      steps: options.diffusionSteps,
      seed: options.seed,
      initialNoise: matchedNoise?.initial,
      stepNoises: matchedNoise?.steps,
      // Under matched noise the per-step coordinate augmentation must also be pinned. Left nil the
      // sampler DRAWS its own rotation/translation per step, which would diverge from an upstream
      // run whose compute_random_augmentation is patched to identity — the trajectories would share
      // an initial condition and then separate immediately.
      rotations: matchedNoise?.identityRotations,
      translations: matchedNoise?.identityTranslations
    )
    return try structure(
      coordinates: coordinates,
      atomMask: requireFeature("atom_pad_mask", from: features)
    )
  }

  /// Whether this pack can score an interface. False for a structure-only pack.
  public var canScoreInterfaces: Bool { confidence != nil }

  /// Fold AND score: returns the structure plus the PAE matrix, from which min_ipSAE is computed.
  /// Throws if the pack has no confidence head, rather than silently returning an unscored result —
  /// a design loop must never mistake "not scored" for "scored badly".
  public func predictScored(
    featurized: BoltzFeaturizer.Output,
    options: BoltzPredictionOptions = BoltzPredictionOptions()
  ) async throws -> ScoredStructure {
    guard let confidence else { throw BoltzError.missingTensor("confidence_module") }
    let features = featurized.features
    try memoryPlanner.validate(metadata: featurized.metadata)
    memoryPlanner.apply()
    try Task.checkCancellation()
    let trunkOutput = try trunk(features: features, recyclingSteps: options.recyclingSteps)
    try Task.checkCancellation()
    let conditioning = try self.conditioning(trunk: trunkOutput, features: features)
    let coordinates = try diffusion.sample(
      trunk: trunkOutput, features: features, conditioning: conditioning,
      steps: options.diffusionSteps, seed: options.seed)
    try Task.checkCancellation()
    // PADDED coordinates, deliberately: token_to_rep_atom is a one-hot over the padded atom axis.
    let scored = try confidence(
      sequenceInputs: trunkOutput.sequenceInput,
      sequence: trunkOutput.sequence,
      pair: trunkOutput.pair,
      predictedCoordinates: coordinates[0].expandedDimensions(axis: 0),
      features: features)
    let structure = try self.structure(
      coordinates: coordinates,
      atomMask: requireFeature("atom_pad_mask", from: features))
    return ScoredStructure(
      structure: structure, pae: scored.pae, plddt: scored.plddt,
      tokenCount: scored.tokenCount,
      chainTokenRanges: featurized.layout.chainTokenRanges)
  }

  public func memorySnapshot() -> Memory.Snapshot {
    Memory.snapshot()
  }

  /// Resets the MLX peak-memory high-water mark so a subsequent prediction can be
  /// measured independently (used by the multi-size on-device benchmark).
  public func resetPeakMemory() {
    GPU.resetPeakMemory()
  }

  private func structure(
    coordinates: MLXArray,
    atomMask: MLXArray
  ) throws -> BoltzStructure {
    let firstSample = coordinates[0].asType(.float32)
    let mask = atomMask[0].asType(.bool)
    MLX.eval(firstSample, mask)
    let coordinateValues = firstSample.asArray(Float.self)
    let maskValues = mask.asArray(Bool.self)
    guard coordinateValues.count == maskValues.count * 3 else {
      throw BoltzError.tensorShapeMismatch(
        name: "sample_atom_coords",
        expected: [maskValues.count, 3],
        actual: firstSample.shape
      )
    }
    var unpadded: [SIMD3<Float>] = []
    unpadded.reserveCapacity(maskValues.filter { $0 }.count)
    for (index, included) in maskValues.enumerated() where included {
      unpadded.append(
        SIMD3(
          coordinateValues[index * 3],
          coordinateValues[index * 3 + 1],
          coordinateValues[index * 3 + 2]
        )
      )
    }
    return BoltzStructure(
      coordinates: unpadded,
      atomMask: Array(repeating: true, count: unpadded.count)
    )
  }
}
