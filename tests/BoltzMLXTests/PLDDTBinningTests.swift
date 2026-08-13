import XCTest
@testable import BoltzMLX

/// The reduction is where binning bugs hide: once [tokens, bins] logits collapse to one
/// scalar per token, a wrong head and wrong bin centres are indistinguishable. These pin
/// the arithmetic against hand-computed values, independent of any checkpoint.
///
/// pLDDT bins tile the LDDT FRACTION 0...1 with centres (i + 0.5)/bins, scaled to a
/// percentage — a different quantity from PAE's Angstrom bins, not just a different scale.
final class PLDDTBinningTests: XCTestCase {

    /// All mass in the top bin of 50 => centre 49.5/50 = 0.99 => 99.
    func testAllMassInTopBinGivesNinetyNine() {
        var logits = [Double](repeating: -1000, count: 50)
        logits[49] = 0
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)
        XCTAssertEqual(out[0], 99.0, accuracy: 1e-6)
    }

    /// All mass in the bottom bin => centre 0.5/50 = 0.01 => 1.
    func testAllMassInBottomBinGivesOne() {
        var logits = [Double](repeating: -1000, count: 50)
        logits[0] = 0
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)
        XCTAssertEqual(out[0], 1.0, accuracy: 1e-6)
    }

    /// A uniform distribution has expectation at the midpoint, 50.
    func testUniformGivesFifty() {
        let logits = [Double](repeating: 0, count: 50)
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)
        XCTAssertEqual(out[0], 50.0, accuracy: 1e-6)
    }

    /// Softmax is shift-invariant; a constant offset must not move the answer.
    func testShiftInvariance() {
        let a = [Double](repeating: 0, count: 50)
        let b = a.map { $0 + 137.0 }
        XCTAssertEqual(
            InterfaceScoring.expectedPLDDT(logits: a, tokenCount: 1, binCount: 50)[0],
            InterfaceScoring.expectedPLDDT(logits: b, tokenCount: 1, binCount: 50)[0],
            accuracy: 1e-9)
    }

    /// Two bins with equal mass average their centres: (0.01 + 0.99)/2 = 0.5 => 50.
    func testTwoEqualExtremesAverage() {
        var logits = [Double](repeating: -1000, count: 50)
        logits[0] = 0; logits[49] = 0
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)
        XCTAssertEqual(out[0], 50.0, accuracy: 1e-6)
    }

    func testPerTokenIndependence() {
        var logits = [Double](repeating: -1000, count: 100)
        logits[49] = 0            // token 0 -> top bin
        logits[50 + 0] = 0        // token 1 -> bottom bin
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 2, binCount: 50)
        XCTAssertEqual(out[0], 99.0, accuracy: 1e-6)
        XCTAssertEqual(out[1], 1.0, accuracy: 1e-6)
    }

    /// Always a percentage, never a fraction — the units a B-factor column expects.
    func testResultIsAPercentageNotAFraction() {
        let logits = (0 ..< 50).map { Double($0) / 10.0 }
        let out = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)
        XCTAssertGreaterThan(out[0], 1.0)
        XCTAssertLessThanOrEqual(out[0], 100.0)
    }

    /// PAE and pLDDT must not drift into one another: same logits, different quantities.
    func testPLDDTIsNotJustPAEWithADifferentMaximum() {
        var logits = [Double](repeating: -1000, count: 50)
        logits[49] = 0
        let plddt = InterfaceScoring.expectedPLDDT(logits: logits, tokenCount: 1, binCount: 50)[0]
        let asPAE = InterfaceScoring.expectedError(
            logits: logits, tokenCount: 1, binCount: 50, maximumError: 1.0)[0]
        XCTAssertEqual(asPAE, 0.99, accuracy: 1e-6)
        XCTAssertEqual(plddt, 99.0, accuracy: 1e-6)
    }
}
