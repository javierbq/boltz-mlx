import MLX
import XCTest

@testable import BoltzMLX

final class SamplerParityTests: XCTestCase {
  func testWeightedRigidAlignRecoversTranslation() throws {
    let coordinates = MLXArray(
      [
        Float(0), 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
      ],
      [1, 4, 3]
    )
    let target = coordinates + MLXArray([Float(2), -3, 4])

    let aligned = weightedRigidAlign(
      coordinates: coordinates,
      target: target,
      weights: MLX.ones([1, 4]),
      mask: MLX.ones([1, 4])
    )

    MLX.eval(aligned)
    let actual = aligned.asArray(Float.self)
    let expected = target.asArray(Float.self)
    for (actualValue, expectedValue) in zip(actual, expected) {
      XCTAssertEqual(actualValue, expectedValue, accuracy: 1e-4)
    }
  }

  func testDiffusionScheduleIncludesScaledEndpointsAndTerminalZero() throws {
    let schedule = diffusionSchedule(
      steps: 3,
      sigmaMinimum: 0.0001,
      sigmaMaximum: 160,
      sigmaData: 16,
      rho: 7
    )

    MLX.eval(schedule)
    let values = schedule.asArray(Float.self)
    XCTAssertEqual(values.count, 4)
    XCTAssertEqual(values[0], 2_560, accuracy: 1e-3)
    XCTAssertEqual(values[2], 0.0016, accuracy: 1e-5)
    XCTAssertEqual(values[3], 0, accuracy: 1e-8)
  }
}
