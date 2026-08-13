import XCTest
@testable import BoltzMLX

/// Env-gated probe against the REAL checkpoint. The binning is unit-tested independently;
/// this checks the head itself — a wrong input tensor or weight lookup yields numbers that
/// look plausible but mean nothing, which no shape assertion would catch.
final class PLDDTLiveProbeTests: XCTestCase {
    func testPLDDTOnARealPrediction() async throws {
        guard let pack = ProcessInfo.processInfo.environment["BOLTZ_PACK"] else {
            throw XCTSkip("set BOLTZ_PACK to the model directory")
        }
        let seq = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAP"
        let canonical = try CanonicalStructure.fromSequences([("A", seq)])
        let features = try BoltzFeaturizer().featurize(canonical, alignments: [:])
        let predictor = try BoltzPredictor(
            modelDirectory: URL(fileURLWithPath: pack),
            memoryPlanner: MemoryPlanner(limits: .desktop))
        var options = BoltzPredictionOptions()
        options.recyclingSteps = 3
        options.diffusionSteps = 200
        let scored = try await predictor.predictScored(featurized: features, options: options)

        let p = scored.plddt
        XCTAssertEqual(p.count, seq.count, "one pLDDT per token")
        let mean = p.reduce(0, +) / Double(p.count)
        let distinct = Set(p.map { Int($0 * 100) }).count
        print("PLDDT n=\(p.count) min=\(p.min()!) max=\(p.max()!) mean=\(mean) distinct=\(distinct)")
        print("PLDDT first8=\(p.prefix(8).map { ($0 * 10).rounded() / 10 })")
        for v in p { XCTAssertTrue(v >= 0 && v <= 100, "out of range: \(v)") }
        // A uniform ~50 everywhere is the signature of a dead head fed uniform logits.
        XCTAssertGreaterThan(distinct, p.count / 4, "pLDDT is suspiciously flat")
        XCTAssertGreaterThan(mean, 50.0, "a confidently folded monomer should not average <50")

        // ORIENTATION. A reversed bin order maps x to 100-x, which stays mid-range and is
        // therefore invisible to any range or spread check. Chain termini are universally
        // the least confident part of a predicted structure, so termini-below-core is a
        // property that DOES distinguish the two, independent of any reference.
        let termini = (p.prefix(3) + p.suffix(3)).reduce(0, +) / 6.0
        let core = p.dropFirst(3).dropLast(3).reduce(0, +) / Double(p.count - 6)
        print("PLDDT termini=\(termini) core=\(core)")
        XCTAssertLessThan(termini, core,
                          "termini should be less confident than the core; if they are not, "
                          + "the bin order is probably reversed")

        // And write the PDB so the B-factor column can be inspected directly.
        let text = try StructureWriter.pdb(structure: scored.structure, canonical: canonical,
                                          plddt: p)
        if let out = ProcessInfo.processInfo.environment["PLDDT_PDB_OUT"] {
            try text.write(to: URL(fileURLWithPath: out), atomically: true, encoding: .utf8)
            print("PLDDT wrote \(out)")
        }
    }
}
