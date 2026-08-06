import XCTest
import Foundation
@testable import BoltzMLX

/// PARITY FOR THE GATE METRIC.
///
/// min_ipSAE is what a design loop accepts or rejects a candidate on, so it is worth pinning exactly
/// against the reference implementation before the model code that produces PAE exists. These tests
/// need no weights, no GPU and no features — they are pure arithmetic over PAE matrices — so they run
/// as an ordinary unit test and would catch a regression long before an end-to-end run could.
///
/// Reference: agent-smith/skills/boltz-predict/scripts/compute_ipsae.py, with expected values
/// generated from it into tests/fixtures/ipsae_cases.json. Cases deliberately span a confident
/// interface, no interface at all, a few-good-pairs interface, everything exactly at the cutoff, and
/// a deliberately asymmetric matrix where min != max (which is the whole reason min_ipSAE is the gate
/// rather than ipSAE).
final class InterfaceScoresTests: XCTestCase {

    private struct Case: Decodable {
        let name: String
        let n_a: Int
        let n_b: Int
        let pae: [Double]
        let expected: Expected
        struct Expected: Decodable {
            let ipsae: Double
            let min_ipsae: Double
            let ipsae_ab: Double
            let ipsae_ba: Double
            let ipae: Double
        }
    }

    private func cases() throws -> [Case] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "fixtures/ipsae_cases.json")
        return try JSONDecoder().decode([Case].self, from: Data(contentsOf: url))
    }

    func testMatchesTheReferenceImplementation() throws {
        let all = try cases()
        XCTAssertGreaterThanOrEqual(all.count, 5, "fixture set looks truncated")
        for c in all {
            let n = c.n_a + c.n_b
            let s = InterfaceScoring.score(paeMatrix: c.pae, tokenCount: n,
                                           a: 0 ..< c.n_a, b: c.n_a ..< n)
            XCTAssertEqual(s.minIPSAE, c.expected.min_ipsae, accuracy: 1e-9, "min_ipsae [\(c.name)]")
            XCTAssertEqual(s.ipsae, c.expected.ipsae, accuracy: 1e-9, "ipsae [\(c.name)]")
            XCTAssertEqual(s.ipsaeAB, c.expected.ipsae_ab, accuracy: 1e-9, "ipsae_ab [\(c.name)]")
            XCTAssertEqual(s.ipsaeBA, c.expected.ipsae_ba, accuracy: 1e-9, "ipsae_ba [\(c.name)]")
            // Looser only for ipae: numpy sums pairwise while this sums sequentially, so ~1e3 terms
            // differ in the last bits. The ipSAE values are maxima of small means and unaffected.
            XCTAssertEqual(s.ipae, c.expected.ipae, accuracy: 1e-7, "ipae [\(c.name)]")
        }
    }

    /// The directional score is a MAX over per-residue means, not a mean of means. Getting that wrong
    /// yields plausible but systematically low numbers, so assert the distinction directly: one
    /// excellent residue must carry the score even when every other residue is at the cutoff.
    func testDirectionalScoreIsMaxOverResiduesNotMean() {
        let nA = 20, nB = 20, n = nA + nB
        var pae = [Double](repeating: 9.99, count: n * n)   // just inside the cutoff everywhere
        for j in nA ..< n { pae[0 * n + j] = 0.1 }          // residue 0 is excellent
        let s = InterfaceScoring.score(paeMatrix: pae, tokenCount: n, a: 0 ..< nA, b: nA ..< n)
        let d = InterfaceScoring.d0(combinedLength: nA + nB)
        let best = 1.0 / (1.0 + pow(0.1 / d, 2))
        XCTAssertEqual(s.ipsaeAB, best, accuracy: 1e-9,
                       "A->B should equal the single best residue's score, not an average")
    }

    /// Pairs at or above the cutoff are excluded entirely. `boundary` in the fixtures covers exactly
    /// 10.0; this asserts the comparison is strict (<), not inclusive (<=).
    func testCutoffIsStrict() {
        let nA = 10, nB = 10, n = nA + nB
        let atCutoff = [Double](repeating: InterfaceScoring.defaultPAECutoff, count: n * n)
        XCTAssertEqual(
            InterfaceScoring.score(paeMatrix: atCutoff, tokenCount: n, a: 0 ..< nA, b: nA ..< n).minIPSAE,
            0, accuracy: 0, "PAE exactly at the cutoff must be excluded")
        var justBelow = atCutoff
        for i in 0 ..< (n * n) { justBelow[i] = InterfaceScoring.defaultPAECutoff - 1e-6 }
        XCTAssertGreaterThan(
            InterfaceScoring.score(paeMatrix: justBelow, tokenCount: n, a: 0 ..< nA, b: nA ..< n).minIPSAE,
            0, "PAE just below the cutoff must count")
    }

    /// The reference CRASHES here — for combined length < 15 it evaluates a cube root of a negative
    /// number, gets a complex value in Python, and `max(complex, 1.0)` raises TypeError. Swift would
    /// instead produce NaN and silently poison the gate. Both are wrong; the floor of 1.0 is the
    /// evident intent, so short chains must return a real number.
    func testShortChainsUseTheD0FloorInsteadOfProducingNaN() {
        XCTAssertEqual(InterfaceScoring.d0(combinedLength: 5), 1.0, accuracy: 0)
        XCTAssertEqual(InterfaceScoring.d0(combinedLength: 15), 1.0, accuracy: 0)
        XCTAssertGreaterThan(InterfaceScoring.d0(combinedLength: 40), 1.0)

        let nA = 3, nB = 2, n = nA + nB
        let pae = [Double](repeating: 1.0, count: n * n)
        let s = InterfaceScoring.score(paeMatrix: pae, tokenCount: n, a: 0 ..< nA, b: nA ..< n)
        XCTAssertFalse(s.minIPSAE.isNaN, "short chains produced NaN")
        XCTAssertEqual(s.minIPSAE, 1.0 / (1.0 + 1.0), accuracy: 1e-9, "d0 = 1 gives ptm = 1/(1+pae^2)")
    }

    /// A designed 12-18 residue macrocycle against a large target is the realistic small case; it must
    /// land in the regime where d0 is COMPUTED rather than floored at 1.0.
    /// d0(162) = 1.24 * 147^(1/3) - 1.8 = 4.744.
    func testMacrocycleAgainstLargeTargetUsesComputedD0() {
        let d = InterfaceScoring.d0(combinedLength: 12 + 150)
        XCTAssertEqual(d, 4.744263789, accuracy: 1e-6)
        XCTAssertGreaterThan(d, 1.0, "must not be floored")
    }

    // MARK: PAE from binned logits

    /// Boltz emits PAE as 64 bins over 32 A; the scalar is the expectation over bin CENTRES.
    /// Using edges instead biases every value low by half a bin (0.25 A here), which would make every
    /// interface look better than it is.
    func testExpectedErrorUsesBinCentres() {
        let bins = 64, n = 1
        // All probability on bin 0 -> the first centre, 0.25 A.
        var logits = [Double](repeating: -50, count: bins)
        logits[0] = 50
        XCTAssertEqual(InterfaceScoring.expectedError(logits: logits, tokenCount: n, binCount: bins)[0],
                       0.25, accuracy: 1e-6)
        // All probability on the last bin -> 31.75 A.
        logits = [Double](repeating: -50, count: bins)
        logits[bins - 1] = 50
        XCTAssertEqual(InterfaceScoring.expectedError(logits: logits, tokenCount: n, binCount: bins)[0],
                       31.75, accuracy: 1e-6)
        // Uniform -> the mean of the centres, 16.0 A.
        logits = [Double](repeating: 0, count: bins)
        XCTAssertEqual(InterfaceScoring.expectedError(logits: logits, tokenCount: n, binCount: bins)[0],
                       16.0, accuracy: 1e-6)
    }

    func testExpectedErrorIsNumericallyStableForLargeLogits() {
        let bins = 64
        let logits = [Double](repeating: 800, count: bins)   // would overflow a naive exp()
        let e = InterfaceScoring.expectedError(logits: logits, tokenCount: 1, binCount: bins)[0]
        XCTAssertFalse(e.isNaN)
        XCTAssertEqual(e, 16.0, accuracy: 1e-6)
    }
}
