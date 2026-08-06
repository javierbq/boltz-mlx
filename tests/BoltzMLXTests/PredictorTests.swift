import XCTest

@testable import BoltzMLX

final class PredictorTests: XCTestCase {
  func testMemoryPlannerRejectsTokenLimitBeforeInference() throws {
    let planner = MemoryPlanner(
      limits: BoltzInputLimits(
        maximumTokens: 16,
        maximumAtoms: 128,
        maximumMSADepth: 32
      ),
      memoryLimit: 1_000_000_000,
      cacheLimit: 64_000_000
    )
    let metadata = FeatureMetadata(
      schemaVersion: 1,
      sampleID: "too-large",
      tokenCount: 17,
      atomCount: 64,
      msaDepth: 8
    )

    XCTAssertThrowsError(try planner.validate(metadata: metadata)) { error in
      XCTAssertEqual(
        error as? BoltzError,
        .inputLimitExceeded(dimension: "tokens", found: 17, maximum: 16)
      )
    }
  }
}
