import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// DOES THE ALIGNMENT ACTUALLY REACH THE NETWORK?
///
/// The parity tests prove the seven MSA tensors are bitwise correct. They cannot prove the tensors
/// are CONSUMED. A featurizer that emits a perfect 249-row alignment into a trunk that ignores
/// everything past row 0 passes every one of them, and the only symptom would be a target that folds
/// as though it had no MSA — which is the exact problem the alignment was added to solve, and is
/// invisible without a reference structure.
///
/// So this folds the same target twice, with and without its alignment, and requires the predictions
/// to differ. It is deliberately a weak numerical claim (the structures differ) rather than a strong
/// one (they differ in the right direction): asserting a specific improvement needs an experimental
/// reference structure, which belongs in the metric work, not in a featurizer test.
///
///   BOLTZ_MODEL=.artifacts/boltz2-mlx-int8 swift test --filter MSAEndToEndTests
final class MSAEndToEndTests: XCTestCase {

    /// 384 tokens and 2961 atoms, well past the phone-sized `MemoryPlanner` defaults. Uses the
    /// `.desktop` preset rather than ad-hoc numbers so this exercises the limits a macOS caller would
    /// actually pass.
    private static let limits = BoltzInputLimits.desktop

    /// Small enough to run as a gate. Not the production regime.
    private let options = BoltzPredictionOptions(recyclingSteps: 0, diffusionSteps: 20, seed: 0)

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func predictor(_ variable: String = "BOLTZ_MODEL") throws -> BoltzPredictor {
        guard let path = ProcessInfo.processInfo.environment[variable] else {
            throw XCTSkip("set \(variable) to an exported int8 model directory")
        }
        return try BoltzPredictor(modelDirectory: URL(fileURLWithPath: path),
                                  memoryPlanner: MemoryPlanner(limits: Self.limits))
    }

    private func pairFixtureChains() throws -> [(chain: String, sequence: String)] {
        try fixtureChains("tests/fixtures/msapair.yaml")
    }

    private func targetSequence() throws -> String {
        let url = repositoryRoot.appending(path: ".artifacts/prot_msa_monomer.yaml")
        for raw in try String(contentsOf: url, encoding: .utf8).split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("sequence:") {
                return String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)
            }
        }
        throw XCTSkip("no sequence in the fixture")
    }

    private func alignment() throws -> MSAAlignment {
        try alignment("examples/msa/seq1.a3m")
    }

    /// Skips rather than fails when the alignment is absent — the barnase/barstar a3m files are not
    /// committed. Regenerate with `scripts/generate_msa.py`.
    private func alignment(_ relative: String) throws -> MSAAlignment {
        let url = repositoryRoot.appending(path: relative)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("missing \(relative); regenerate with scripts/generate_msa.py")
        }
        return try MSAAlignment.a3m(try String(contentsOf: url, encoding: .utf8))
    }

    private func radiusOfGyration(_ points: [SIMD3<Float>]) -> Double {
        var centre = SIMD3<Double>()
        for p in points { centre += SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)) }
        centre /= Double(points.count)
        var total = 0.0
        for p in points {
            let d = SIMD3<Double>(Double(p.x), Double(p.y), Double(p.z)) - centre
            total += (d * d).sum()
        }
        return (total / Double(points.count)).squareRoot()
    }

    /// Both folds use the SAME seed, so the initial noise and the augmentation are identical and any
    /// coordinate difference is attributable to the MSA features alone. That is what makes a weak
    /// threshold sufficient here: an inert MSA would give exactly 0.000 A, not a small number.
    ///
    /// DO NOT READ THE PRINTED RMSD AS AN IMPROVEMENT. At recycling 0 / 20 steps two runs of this
    /// target differing only in seed disagree by ~14.5 A (see EndToEndNoPythonTests), so a difference
    /// of this size says nothing about which fold is better — only that the features are consumed.
    /// Judging the alignment's benefit needs an experimental reference structure.
    func testTheAlignmentChangesThePredictedStructure() async throws {
        let predictor = try predictor()
        let sequence = try targetSequence()
        let structure = try CanonicalStructure.fromSequences([("A", sequence)])

        let withoutMSA = try BoltzFeaturizer().featurize(structure)
        let withMSA = try BoltzFeaturizer().featurize(structure, alignments: ["A": try alignment()])
        XCTAssertEqual(withoutMSA.layout.msaDepth, 1)
        XCTAssertEqual(withMSA.layout.msaDepth, 249)

        let bare = try await predictor.predict(featurized: withoutMSA, options: options)
        let aligned = try await predictor.predict(featurized: withMSA, options: options)

        XCTAssertEqual(bare.coordinates.count, aligned.coordinates.count,
                       "same sequence must yield the same atom count")

        var maximum = 0.0, total = 0.0
        for (a, b) in zip(bare.coordinates, aligned.coordinates) {
            let d = SIMD3<Double>(Double(a.x - b.x), Double(a.y - b.y), Double(a.z - b.z))
            let distance = (d * d).sum().squareRoot()
            maximum = max(maximum, distance)
            total += distance * distance
        }
        let rmsd = (total / Double(bare.coordinates.count)).squareRoot()

        print(String(format: "MSA A/B: RMSD %.3f A, max %.3f A, Rg %.2f -> %.2f A (depth 1 -> 249)",
                     rmsd, maximum, radiusOfGyration(bare.coordinates),
                     radiusOfGyration(aligned.coordinates)))

        XCTAssertGreaterThan(rmsd, 0.1,
                             "the 249-row alignment did not change the prediction; the MSA tensors "
                             + "are correct but are not reaching the trunk")
        // Both must still be a folded protein, not an exploded one — a plumbing error that scrambled
        // the MSA could satisfy the assertion above while destroying the fold.
        for (label, structure) in [("no MSA", bare), ("MSA", aligned)] {
            let rg = radiusOfGyration(structure.coordinates)
            XCTAssertTrue((10.0 ... 40.0).contains(rg),
                          "\(label): radius of gyration \(rg) A is not a folded 384-residue protein")
        }
    }

    /// The design shape: a binder with no alignment beside a target with one, through the network.
    /// Guards against a shape or masking error that only appears when the two chains have different
    /// alignment depths.
    func testABinderAgainstAnAlignedTargetFolds() async throws {
        let predictor = try predictor()
        let featurized = try BoltzFeaturizer().featurize(
            try CanonicalStructure.fromSequences(try pairFixtureChains()),
            alignments: ["B": try alignment()])
        XCTAssertEqual(featurized.layout.tokenCount, 424)
        XCTAssertEqual(featurized.layout.msaDepth, 249)

        let structure = try await predictor.predict(featurized: featurized, options: options)
        XCTAssertEqual(structure.coordinates.count, featurized.layout.atomCount)
        XCTAssertFalse(structure.coordinates.contains { $0.x.isNaN || $0.y.isNaN || $0.z.isNaN },
                       "prediction produced NaN coordinates")
        print(String(format: "binder+aligned target: %d atoms, Rg %.2f A",
                     structure.coordinates.count, radiusOfGyration(structure.coordinates)))
    }

    // MARK: A real binder against a really-aligned target

    /// Barnase + barstar: a natural femtomolar-affinity complex (PDB 1BRS), with a real 6628-row
    /// ColabFold alignment for the target and none for the binder.
    ///
    /// WHY THIS AND NOT THE POLY-ALANINE FIXTURE. That one only showed the gate rejecting garbage,
    /// which a broken pipeline also does — min_ipSAE 0.0 is the answer you get from a correct
    /// implementation on a non-binder AND from an implementation that has destroyed the interface.
    /// Separating those needs a pair the model should score HIGH.
    ///
    /// Runs at recycling 3 / 200 steps by default: that is the regime the confidence head was
    /// trained at, and the gate value is meaningless outside it.
    ///
    /// CAVEAT, stated because it limits the claim: 1BRS is in the PDB and therefore in training, so a
    /// good score here is not evidence of generalisation. It is evidence that a real interface
    /// survives the alignment path intact, which is what is being tested.
    func testARealBinderAgainstADeeplyAlignedTargetScoresAsABinder() async throws {
        let predictor = try predictor("BOLTZ_CONF_MODEL")
        let environment = ProcessInfo.processInfo.environment
        let production = BoltzPredictionOptions(
            recyclingSteps: Int(environment["BOLTZ_RECYCLING"] ?? "") ?? 3,
            diffusionSteps: Int(environment["BOLTZ_STEPS"] ?? "") ?? 200,
            seed: 0)

        let chains = try fixtureChains("tests/fixtures/realbinder.yaml")
        let structure = try CanonicalStructure.fromSequences(chains)
        let barnaseMSA = try alignment(".artifacts/barnase.a3m")
        let barstarMSA = try alignment(".artifacts/barstar.a3m")

        // Three variants. "target only" is the design shape; "both" is what you would actually do for
        // a natural complex and is the fair test of whether the pipeline can reproduce 1BRS; "none"
        // is the control.
        let targetOnly = try BoltzFeaturizer().featurize(structure, alignments: ["A": barnaseMSA])
        let both = try BoltzFeaturizer().featurize(
            structure, alignments: ["A": barnaseMSA, "B": barstarMSA])
        let withoutMSA = try BoltzFeaturizer().featurize(structure)
        XCTAssertEqual(targetOnly.layout.msaDepth, 6_628)
        XCTAssertEqual(both.layout.msaDepth, 6_628, "depth follows the deeper chain")
        XCTAssertEqual(targetOnly.layout.tokenCount, 199)

        var results: [(label: String, scores: InterfaceScores, structure: BoltzStructure)] = []
        for (label, features) in [("target only", targetOnly), ("both", both),
                                  ("no MSA", withoutMSA)] {
            let scored = try await predictor.predictScored(featurized: features, options: production)
            guard let scores = scored.interfaceScores() else {
                return XCTFail("no interface scores; the pack has no confidence head")
            }
            // NOTE: %s with a Swift String segfaults in CFStringAppendFormatCore — it expects a C
            // string pointer. Interpolate instead of formatting the label.
            print("barnase/barstar \(label.padding(toLength: 9, withPad: " ", startingAt: 0))"
                  + " depth \(features.layout.msaDepth)"
                  + String(format: "  min_ipSAE %.4f  ipSAE %.4f  iPAE %6.2f A",
                           scores.minIPSAE, scores.ipsae, scores.ipae))
            results.append((label, scores, scored.structure))
            if let directory = environment["BOLTZ_CA_OUT"] {
                let stem = label.replacingOccurrences(of: " ", with: "_")
                let base = URL(fileURLWithPath: directory)
                try writeAlphaCarbons(scored.structure, chains: chains,
                                      to: base.appending(path: "\(stem).csv"))
                try StructureWriter.pdb(structure: scored.structure,
                                        canonical: structure)
                    .write(to: base.appending(path: "\(stem).pdb"),
                           atomically: true, encoding: .utf8)
            }
        }

        let both_ = results[1].scores, none = results[2].scores
        // A real, high-affinity interface must clear the floor the poly-alanine fixture sat on. A
        // deliberately loose bound: this separates "scored as an interface at all" from "scored 0",
        // and asserts no calibrated cutoff (see InterfaceScores.minIPSAE).
        XCTAssertGreaterThan(both_.minIPSAE, 0.5,
                             "a femtomolar natural complex scored like a non-binder; the interface "
                             + "did not survive the alignment path")
        XCTAssertLessThan(both_.ipae, 10.0, "interface PAE should be confident for a real complex")
        XCTAssertEqual(both_.minIPSAE, min(both_.ipsaeAB, both_.ipsaeBA), accuracy: 1e-12)
        // The control must NOT score as a binder. Without this, an implementation that returned a
        // confident gate for everything would pass the assertion above.
        XCTAssertLessThan(none.minIPSAE, 0.1,
                          "the single-sequence control scored as a binder; the gate is not "
                          + "discriminating and the comparison above proves nothing")
    }

    /// CA coordinates labelled by chain and residue ordinal, so a comparison to the crystal can pair
    /// by identity rather than by array position.
    private func writeAlphaCarbons(
        _ structure: BoltzStructure,
        chains: [(chain: String, sequence: String)],
        to url: URL
    ) throws {
        var lines = ["chain,residue,x,y,z"]
        var atom = 0
        for (chain, sequence) in chains {
            for (ordinal, letter) in sequence.enumerated() {
                guard let template = AAResidueTemplates.template(oneLetter: letter) else { continue }
                if let offset = template.atoms.firstIndex(where: { $0.name == "CA" }),
                   atom + offset < structure.coordinates.count {
                    let p = structure.coordinates[atom + offset]
                    lines.append("\(chain),\(ordinal + 1),\(p.x),\(p.y),\(p.z)")
                }
                atom += template.atoms.count
            }
        }
        try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    private func fixtureChains(_ path: String) throws -> [(chain: String, sequence: String)] {
        let text = try String(contentsOf: repositoryRoot.appending(path: path), encoding: .utf8)
        var chains: [(String, String)] = []
        var pending: String?
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("id:") {
                pending = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("sequence:") {
                chains.append((pending ?? "A",
                               String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)))
                pending = nil
            }
        }
        return chains
    }

    /// THE WHOLE LOOP: aligned target -> fold -> min_ipSAE. This is the shape a design run gates on,
    /// so it is worth knowing that the alignment path and the confidence path compose — the MSA
    /// changes the trunk representations the confidence head reads, and a masking error in the deep
    /// MSA could surface as a NaN gate rather than a bad fold.
    ///
    /// Asserts only that the gate is a well-formed number in range. The VALUE at recycling 0 / 20
    /// steps carries no information about whether this binder is real, and no threshold is applied
    /// here for the reason documented on `InterfaceScores.minIPSAE`.
    func testTheGateMetricIsWellFormedWithADeepAlignment() async throws {
        let predictor = try predictor("BOLTZ_CONF_MODEL")
        let featurized = try BoltzFeaturizer().featurize(
            try CanonicalStructure.fromSequences(try pairFixtureChains()),
            alignments: ["B": try alignment()])

        let scored = try await predictor.predictScored(featurized: featurized, options: options)
        guard let scores = scored.interfaceScores() else {
            return XCTFail("no interface scores; the pack has no confidence head")
        }
        print(String(format: "deep-MSA gate: min_ipSAE %.4f  ipSAE %.4f  iPAE %.2f A",
                     scores.minIPSAE, scores.ipsae, scores.ipae))

        XCTAssertFalse(scores.minIPSAE.isNaN, "the gate is NaN with a 249-row alignment")
        XCTAssertTrue((0.0 ... 1.0).contains(scores.minIPSAE), "min_ipSAE out of range")
        XCTAssertLessThanOrEqual(scores.minIPSAE, scores.ipsae, "min must not exceed max")
        XCTAssertTrue(scores.ipae.isFinite && scores.ipae > 0, "iPAE must be a positive distance")
    }
}
