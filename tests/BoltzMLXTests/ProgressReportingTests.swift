import MLX
import XCTest

@testable import BoltzMLX

/// Serialises reports arriving on the actor's executor. All locking is scoped in
/// here so the async test body never calls NSLock.lock()/unlock(), which Swift 6
/// makes unavailable from an async context.
private final class ProgressCollector: @unchecked Sendable {
  private let lock = NSLock()
  private var reports: [BoltzProgress] = []

  func append(_ report: BoltzProgress) {
    lock.withLock { reports.append(report) }
  }

  func snapshot() -> [BoltzProgress] {
    lock.withLock { reports }
  }
}

/// The progress channel's contract, pinned without loading a model.
///
/// The callbacks fire from loops that end in an `MLX.eval`, so the only honest
/// place to assert their *timing* is an end-to-end run; what is asserted here is
/// the shape a consumer depends on -- counts start at 1, end at `total`, are
/// monotone, and never divide by zero.
final class ProgressReportingTests: XCTestCase {

  func testFractionIsCompletedOverTotal() {
    let p = BoltzProgress(stage: .diffusion, completed: 84, total: 200)
    XCTAssertEqual(p.fraction, 0.42, accuracy: 1e-9)
  }

  /// A caller dividing by `total` directly would trap here; the property must not.
  func testFractionIsZeroRatherThanDividingByZero() {
    XCTAssertEqual(BoltzProgress(stage: .trunk, completed: 0, total: 0).fraction, 0)
    XCTAssertEqual(BoltzProgress(stage: .trunk, completed: 3, total: -1).fraction, 0)
  }

  func testStagesAreDistinctAndStable() {
    // The raw values cross a wire into RayMol's phase table; renaming one would
    // silently stop matching its band.
    XCTAssertEqual(BoltzProgress.Stage.trunk.rawValue, "trunk")
    XCTAssertEqual(BoltzProgress.Stage.diffusion.rawValue, "diffusion")
  }

  /// The diffusion sampler reports `index + 1` of `stepCount`, so a consumer can
  /// treat the last report as "sampling finished" without an off-by-one.
  func testReportSequenceStartsAtOneAndEndsAtTotal() {
    let total = 200
    let reports = (0..<total).map {
      BoltzProgress(stage: .diffusion, completed: $0 + 1, total: total)
    }
    XCTAssertEqual(reports.first?.completed, 1)
    XCTAssertEqual(reports.last?.completed, total)
    XCTAssertEqual(reports.last?.fraction, 1.0)
    for (a, b) in zip(reports, reports.dropFirst()) {
      XCTAssertLessThan(a.completed, b.completed)
    }
  }

  /// The trunk loop runs `0...recyclingSteps`, so 3 recycles is 4 passes. Getting
  /// this wrong would make the trunk bar stop short of its band's end.
  func testTrunkTotalIsRecyclingStepsPlusOne() {
    let recyclingSteps = 3
    let p = BoltzProgress(
      stage: .trunk, completed: recyclingSteps + 1, total: recyclingSteps + 1)
    XCTAssertEqual(p.total, 4)
    XCTAssertEqual(p.fraction, 1.0)
  }

  /// The one that matters: the callbacks actually FIRE from a real run.
  ///
  /// Everything above pins the value type; only this proves the emit sites are
  /// inside the loops and reached. Deliberately tiny -- 2 recycles / 4 diffusion
  /// steps on one short chain -- because it is a wiring gate, not a quality gate.
  ///
  ///   BOLTZ_MODEL=~/Library/Application\ Support/RayMol/weights/boltz2-mlx-int8
  func testProgressIsReportedFromARealRun() async throws {
    guard let dir = ProcessInfo.processInfo.environment["BOLTZ_MODEL"] else {
      throw XCTSkip("set BOLTZ_MODEL to an exported model directory")
    }
    let limits = BoltzInputLimits(
      maximumTokens: 512, maximumAtoms: 4_096, maximumMSADepth: 1_024)
    let predictor = try BoltzPredictor(
      modelDirectory: URL(fileURLWithPath: dir),
      memoryPlanner: MemoryPlanner(limits: limits))

    let canonical = try CanonicalStructure.fromSequences(
      [("A", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")])
    XCTAssertTrue(canonical.diagnostics.isEmpty, "\(canonical.diagnostics)")
    let featurized = try BoltzFeaturizer().featurize(canonical)

    // The callback runs on the actor's executor while this test awaits, so a
    // plain array would be a concurrent write. NSLock's lock()/unlock() are
    // unavailable from an async context in Swift 6, so the locking is scoped
    // inside a collector and never touched directly from the async body.
    let collector = ProgressCollector()

    _ = try await predictor.predictScored(
      featurized: featurized,
      options: BoltzPredictionOptions(
        recyclingSteps: 2, diffusionSteps: 4, seed: 0),
      onProgress: { collector.append($0) })

    let reports = collector.snapshot()
    XCTAssertFalse(reports.isEmpty, "no progress was reported from a real run")

    let trunk = reports.filter { $0.stage == .trunk }
    let diffusion = reports.filter { $0.stage == .diffusion }

    // recyclingSteps 2 -> the loop runs 0...2 -> 3 passes.
    XCTAssertEqual(trunk.map(\.completed), [1, 2, 3])
    XCTAssertEqual(trunk.map(\.total), [3, 3, 3])
    XCTAssertEqual(diffusion.map(\.completed), [1, 2, 3, 4])
    XCTAssertEqual(diffusion.map(\.total), [4, 4, 4, 4])

    // Ordering: every trunk report precedes every diffusion report, so a consumer
    // mapping stages onto an ordered band table never sees the bar go backwards.
    let firstDiffusion = try XCTUnwrap(reports.firstIndex { $0.stage == .diffusion })
    XCTAssertFalse(reports[firstDiffusion...].contains { $0.stage == .trunk })
    XCTAssertEqual(reports.last?.fraction, 1.0)
  }

  /// Every new parameter is defaulted and appended last, so pre-existing call
  /// sites keep compiling. If this stops compiling, the change is source-breaking.
  func testExistingCallSitesStillCompileWithoutProgress() {
    let sampleWithoutProgress: (AtomDiffusion, TrunkOutput, [String: MLXArray],
                                DiffusionConditioningOutput) throws -> MLXArray = {
      diffusion, trunk, features, conditioning in
      try diffusion.sample(
        trunk: trunk, features: features, conditioning: conditioning,
        steps: 20, seed: 0)
    }
    XCTAssertNotNil(sampleWithoutProgress)
  }
}
