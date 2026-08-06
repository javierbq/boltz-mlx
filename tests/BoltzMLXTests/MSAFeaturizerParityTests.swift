import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// BITWISE PARITY FOR THE MSA TENSORS against Python-exported bundles.
///
/// The unit tests next door pin parsing and pairing in isolation; this pins the part that cannot be
/// checked without the reference — the expansion of a row plan into token space, and the exact
/// content of the seven tensors the trunk reads.
///
/// TWO FIXTURES, and the second is the one that matters:
///
///   * `prot_msa_monomer` — one chain, `examples/msa/seq1.a3m`, 249 rows over 384 columns with 217
///     rows carrying insertions. That last property is what makes the deletion assertions
///     meaningful: an alignment with no insertions could not tell a correct implementation from one
///     that silently dropped them.
///   * `msapair` — a 40-residue designed binder with NO alignment against that same 384-residue
///     target WITH its alignment. This is the production shape of a design run, and the only fixture
///     that exercises gap expansion and the paired/unpaired distinction across a real interface.
///
///   BOLTZ_MSA_REF_DIR=.artifacts/features swift test --filter MSAFeaturizerParityTests
final class MSAFeaturizerParityTests: XCTestCase {

    private struct Case {
        /// Reference bundle directory name under BOLTZ_MSA_REF_DIR.
        let name: String
        /// Fixture YAML, repository-relative — the same file the export was run on, so the sequences
        /// here can never drift from the ones the reference was built from.
        let yaml: String
        /// Chain -> a3m path. A chain absent from this map is featurized with no alignment.
        let alignments: [String: String]
        let depth: Int
        let tokenCount: Int
    }

    private static let monomer = Case(
        name: "prot_msa_monomer", yaml: ".artifacts/prot_msa_monomer.yaml",
        alignments: ["A": "examples/msa/seq1.a3m"], depth: 249, tokenCount: 384)

    private static let pair = Case(
        name: "msapair", yaml: "tests/fixtures/msapair.yaml",
        alignments: ["B": "examples/msa/seq1.a3m"], depth: 249, tokenCount: 424)

    /// Barnase (target, real 6628-row alignment from the internal ColabFold server) + barstar
    /// (its natural femtomolar-affinity inhibitor, no alignment). A genuine binder, and 27x deeper
    /// than the other fixtures — deep enough that anything quadratic or accumulating in the depth
    /// axis shows up.
    private static let realBinder = Case(
        name: "realbinder", yaml: "tests/fixtures/realbinder.yaml",
        alignments: ["A": ".artifacts/barnase.a3m"], depth: 6_628, tokenCount: 199)

    /// The same complex with BOTH chains aligned, at DIFFERENT real depths (6628 and 5735). This is
    /// the only fixture exercising the unpaired fill when one chain runs out of sequences before the
    /// other — the final depth is the deeper chain's, and barstar must be gapped past its own 5735.
    private static let realBinderBoth = Case(
        name: "realbinder_both", yaml: "tests/fixtures/realbinder_both.yaml",
        alignments: ["A": ".artifacts/barnase.a3m", "B": ".artifacts/barstar.a3m"],
        depth: 6_628, tokenCount: 199)

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func chains(_ c: Case) throws -> [(chain: String, sequence: String)] {
        let text = try String(contentsOf: repositoryRoot.appending(path: c.yaml), encoding: .utf8)
        var out: [(String, String)] = []
        var pendingID: String?
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("id:") {
                pendingID = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("sequence:") {
                // A single-chain fixture may omit `id:`; default it to the reference's chain A.
                out.append((pendingID ?? "A",
                            String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)))
                pendingID = nil
            }
        }
        XCTAssertFalse(out.isEmpty, "fixture \(c.yaml) parsed no chains")
        return out
    }

    /// Skips rather than fails when an alignment is absent: the barnase/barstar a3m files live in
    /// `.artifacts/` beside the reference bundles they pair with, and neither is committed. Regenerate
    /// both with `scripts/generate_msa.py` (see validation/msa_real_binder/report.md).
    private func alignments(_ c: Case) throws -> [String: MSAAlignment] {
        try c.alignments.mapValues { relative in
            let url = repositoryRoot.appending(path: relative)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("missing \(relative); regenerate with scripts/generate_msa.py")
            }
            return try MSAAlignment.a3m(try String(contentsOf: url, encoding: .utf8))
        }
    }

    private func reference(_ c: Case) throws -> FeatureBundle {
        guard let root = ProcessInfo.processInfo.environment["BOLTZ_MSA_REF_DIR"] else {
            throw XCTSkip("set BOLTZ_MSA_REF_DIR to the directory holding \(c.name)/")
        }
        var base = URL(fileURLWithPath: root)
        if !root.hasPrefix("/") { base = repositoryRoot.appending(path: root) }
        return try FeatureBundle.load(from: base.appending(path: c.name))
    }

    private func featurize(_ c: Case) throws -> BoltzFeaturizer.Output {
        try BoltzFeaturizer().featurize(try CanonicalStructure.fromSequences(try chains(c)),
                                       alignments: try alignments(c))
    }

    /// Every tensor the MSA module reads. `profile` is included because it is DERIVED from the msa
    /// matrix — a correct msa with a profile still computed from the query one-hot would otherwise
    /// pass unnoticed.
    private static let msaTensors = [
        "msa", "msa_mask", "msa_paired", "deletion_value", "has_deletion", "deletion_mean", "profile",
    ]

    private func assertBitwiseParity(_ c: Case) throws {
        let ref = try reference(c)
        let out = try featurize(c)
        XCTAssertEqual(ref.metadata.msaDepth, c.depth, "reference bundle is not the expected depth")
        XCTAssertEqual(out.layout.tokenCount, c.tokenCount)
        XCTAssertEqual(out.layout.msaDepth, c.depth)

        var mismatches: [String] = []
        for name in Self.msaTensors {
            guard let expected = ref.arrays[name] else {
                mismatches.append("\(name): absent from reference"); continue
            }
            guard let actual = out.features[name] else {
                mismatches.append("\(name): not emitted"); continue
            }
            guard actual.shape == expected.shape else {
                mismatches.append("\(name): shape \(actual.shape) vs \(expected.shape)"); continue
            }
            let delta = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32)))
                .item(Float.self)
            if delta != 0 { mismatches.append("\(name): max|delta| = \(delta)") }
        }
        XCTAssertTrue(mismatches.isEmpty,
                      "\(c.name): \(mismatches.count) of \(Self.msaTensors.count) MSA tensors differ:\n"
                      + mismatches.joined(separator: "\n"))
    }

    // MARK: The gate

    func testTheParsedAlignmentHasTheReferenceDepth() throws {
        let alignment = try alignments(Self.monomer)["A"]!
        XCTAssertEqual(alignment.depth, Self.monomer.depth,
                       "de-duplication removed a different number of rows than upstream")
        XCTAssertEqual(alignment.queryLength, 384)
    }

    func testBitwiseParityOnADeepMonomerAlignment() throws {
        try assertBitwiseParity(Self.monomer)
    }

    /// THE DESIGN CASE. Binder without an alignment, target with one.
    func testBitwiseParityOnABinderAgainstAnAlignedTarget() throws {
        try assertBitwiseParity(Self.pair)
    }

    /// A REAL BINDER against a real 6628-row alignment. Depth is where a subtle port error hides:
    /// the row cap, the unpaired fill and the profile denominator are all depth-dependent and all
    /// look correct at depth 249.
    func testBitwiseParityOnARealBinderAgainstADeeplyAlignedTarget() throws {
        try assertBitwiseParity(Self.realBinder)
    }

    /// Two real alignments of unequal depth through the pairing code.
    func testBitwiseParityOnTwoChainsBothDeeplyAligned() throws {
        try assertBitwiseParity(Self.realBinderBoth)
    }

    /// The unequal-depth gap fill, on real data: past barstar's own 5735 sequences its tokens must be
    /// gaps while barnase keeps advancing to 6628. Getting this wrong (repeating barstar's last
    /// sequence, or truncating the whole MSA to the shallower chain) still produces a well-formed
    /// tensor of plausible shape.
    func testTheShallowerChainIsGappedPastItsOwnDepth() throws {
        let out = try featurize(Self.realBinderBoth)
        let binder = out.layout.chainTokenRanges.first { $0.chain == "B" }!.range
        let target = out.layout.chainTokenRanges.first { $0.chain == "A" }!.range
        let depth = out.layout.msaDepth, tokens = out.layout.tokenCount
        let msa = out.features["msa"]!.reshaped([depth, tokens]).asType(.int32).asArray(Int32.self)

        let barstarDepth = try alignments(Self.realBinderBoth)["B"]!.depth
        XCTAssertEqual(barstarDepth, 5_735)
        XCTAssertLessThan(barstarDepth, depth, "fixture no longer has unequal depths")

        // Just inside barstar's depth: real residues, not gaps.
        let lastReal = (barstarDepth - 1) * tokens
        XCTAssertFalse(binder.allSatisfy { msa[lastReal + $0] == MSAAlignment.gapToken },
                       "row \(barstarDepth - 1) should still carry barstar sequence")
        // Past it: every binder token is a gap, while the target still advances.
        for r in barstarDepth ..< depth {
            for t in binder {
                XCTAssertEqual(msa[r * tokens + t], MSAAlignment.gapToken,
                               "row \(r) token \(t): barstar is exhausted and must be gapped")
            }
        }
        XCTAssertFalse(target.allSatisfy { msa[(depth - 1) * tokens + $0] == MSAAlignment.gapToken },
                       "the deeper chain must still carry sequence in the final row")
    }

    /// Dedup is the one place the Swift parser could silently disagree about depth — and if it did,
    /// every row after the divergence would be shifted while still looking like a plausible MSA.
    /// 6823 raw rows collapsing to exactly the reference's 6628 pins it.
    func testDeduplicationAgreesWithUpstreamOnARealAlignment() throws {
        let alignment = try alignments(Self.realBinder)["A"]!
        XCTAssertEqual(alignment.depth, 6_628)
        XCTAssertEqual(alignment.queryLength, 199 - 89, "barnase is 110 residues")
        XCTAssertEqual(try reference(Self.realBinder).metadata.msaDepth, alignment.depth)
    }

    /// The featurizer must report the real depth so `MemoryPlanner` can refuse an alignment that
    /// would not fit. Hardcoding 1 (as the single-sequence path did) makes the check vacuous.
    func testMetadataReportsTheRealMSADepth() throws {
        let out = try featurize(Self.monomer)
        XCTAssertEqual(out.metadata.msaDepth, Self.monomer.depth)
        XCTAssertEqual(out.layout.msaDepth, Self.monomer.depth)
    }

    // MARK: Direction of the pairing, asserted independently of the reference

    /// Stated directly rather than left implicit in the bitwise comparison, because getting it
    /// backwards does not fail loudly — it asserts co-evolution across the interface min_ipSAE
    /// scores and makes the gate read HIGH. Both halves are checked: the binder's tokens are gaps in
    /// every homolog row, and nothing past row 0 is marked paired.
    func testTheBinderIsGappedAndUnpairedInEveryHomologRow() throws {
        let out = try featurize(Self.pair)
        let range = out.layout.chainTokenRanges.first { $0.chain == "A" }!.range
        XCTAssertEqual(range.count, 40, "chain A should be the 40-residue binder")

        let depth = out.layout.msaDepth, tokens = out.layout.tokenCount
        let msa = out.features["msa"]!.reshaped([depth, tokens]).asType(.int32).asArray(Int32.self)
        let paired = out.features["msa_paired"]!.reshaped([depth, tokens]).asArray(Float.self)

        // Row 0 is the binder's own sequence, and is the one paired row.
        for t in range {
            XCTAssertNotEqual(msa[t], MSAAlignment.gapToken, "row 0 token \(t) must be the query")
            XCTAssertEqual(paired[t], 1, "row 0 must be paired")
        }
        for r in 1 ..< depth {
            for t in range {
                XCTAssertEqual(msa[r * tokens + t], MSAAlignment.gapToken,
                               "row \(r) token \(t): the binder must be a gap, not a repeat")
                XCTAssertEqual(paired[r * tokens + t], 0,
                               "row \(r): an unannotated target homolog is not paired to the binder")
            }
        }
    }

    /// The target half of the same rows must actually vary — otherwise the test above would pass on
    /// an implementation that emitted gaps everywhere.
    func testTheTargetHalfActuallyCarriesTheAlignment() throws {
        let out = try featurize(Self.pair)
        let range = out.layout.chainTokenRanges.first { $0.chain == "B" }!.range
        let depth = out.layout.msaDepth, tokens = out.layout.tokenCount
        let msa = out.features["msa"]!.reshaped([depth, tokens]).asType(.int32).asArray(Int32.self)

        var distinctRows = Set<[Int32]>()
        for r in 0 ..< depth {
            distinctRows.insert(range.map { msa[r * tokens + $0] })
        }
        XCTAssertEqual(distinctRows.count, depth, "every target row should be a distinct homolog")
    }

    // MARK: The deletion features

    /// UPSTREAM BUG, pinned. 217 of this fixture's 249 rows carry insertions, yet the reference
    /// emits deletion features that are identically zero: `construct_paired_msa` rebinds
    /// `chain_deletions` to a slice INSIDE the per-sequence loop, so after sequence 0 — the query,
    /// which by a3m construction has no insertions and therefore an empty slice — every later slice
    /// is empty too, and the deletion dict is never populated.
    ///
    /// This test is what makes emitting zeros a decision rather than an omission. If a future
    /// upstream fixes the bug, this fails and `MSAAlignment.insertionCounts` is already parsed and
    /// waiting to be wired in.
    func testDeletionFeaturesAreZeroDespiteTheAlignmentCarryingInsertions() throws {
        let alignment = try alignments(Self.monomer)["A"]!
        let rowsWithInsertions = alignment.insertionCounts.filter { $0.contains { $0 > 0 } }.count
        XCTAssertEqual(rowsWithInsertions, 217,
                       "fixture no longer exercises insertions; the zero assertions below prove nothing")

        let out = try featurize(Self.monomer)
        for name in ["deletion_value", "has_deletion", "deletion_mean"] {
            let sum = MLX.sum(out.features[name]!.asType(.float32)).item(Float.self)
            XCTAssertEqual(sum, 0, accuracy: 0, "\(name) must be identically zero")
        }
    }

    // MARK: Refusing a mismatched alignment

    /// Upstream discards a mismatched alignment and silently continues in single-sequence mode with
    /// only a printed warning — and this repository's own `examples/prot_custom_msa.yaml` trips it:
    /// its sequence is 117 residues while `seq2.a3m` is 136 columns, which is why the exported
    /// `prot_custom_msa` bundle has msa_depth 1. A design run that loses its target's alignment that
    /// way scores every candidate against a worse-folded target with nothing in the output to say
    /// so, so this throws instead.
    func testAMismatchedAlignmentIsRefusedRatherThanSilentlyDropped() throws {
        let structure = try CanonicalStructure.fromSequences([("A", "MVTPEG")])
        let tooShort = try MSAAlignment.a3m(">q\nMVT\n>h1\nMVA\n")
        XCTAssertThrowsError(try BoltzFeaturizer().featurize(structure, alignments: ["A": tooShort])) {
            guard case BoltzFeaturizerError.msaLengthMismatch(let chain, let expected, let found) = $0 else {
                return XCTFail("wrong error: \($0)")
            }
            XCTAssertEqual(chain, "A")
            XCTAssertEqual([expected, found], [6, 3])
        }
    }

    /// Right length, wrong residues — the alignment is for a different construct.
    func testAnAlignmentOfADifferentSequenceIsRefused() throws {
        let structure = try CanonicalStructure.fromSequences([("A", "MVTPEG")])
        let wrong = try MSAAlignment.a3m(">q\nMVTPEW\n>h1\nMVTPEA\n")
        XCTAssertThrowsError(try BoltzFeaturizer().featurize(structure, alignments: ["A": wrong])) {
            guard case BoltzFeaturizerError.msaQueryMismatch(let chain, let positions) = $0 else {
                return XCTFail("wrong error: \($0)")
            }
            XCTAssertEqual(chain, "A")
            XCTAssertEqual(positions, [5])
        }
    }

    /// Upstream tolerates exactly one class of disagreement: the alignment saying UNK where the
    /// sequence says MET (a common artefact of MSA construction). It overwrites the query row rather
    /// than discarding the alignment, and so does this.
    func testAnUnknownForMethionineIsReconciledRatherThanRefused() throws {
        let structure = try CanonicalStructure.fromSequences([("A", "MVTPEG")])
        let withUnknown = try MSAAlignment.a3m(">q\nXVTPEG\n>h1\nMVTPEA\n")
        let out = try BoltzFeaturizer().featurize(structure, alignments: ["A": withUnknown])
        let msa = out.features["msa"]!.reshaped([2, 6]).asType(.int32).asArray(Int32.self)
        XCTAssertEqual(msa[0], Int32(AAResidueTemplates.template(oneLetter: "M")!.restype),
                       "the query row must be reconciled to the structure's MET, not left as UNK")
    }

    // MARK: Without an alignment

    /// A chain with no alignment still gets the depth-1 synthetic row, so the no-MSA path is
    /// unchanged by all of this.
    func testAChainWithoutAnAlignmentKeepsTheSingleSyntheticRow() throws {
        let structure = try CanonicalStructure.fromSequences([("A", "MVTPEG")])
        let out = try BoltzFeaturizer().featurize(structure)
        XCTAssertEqual(out.features["msa"]!.shape, [1, 1, 6])
        XCTAssertEqual(out.layout.msaDepth, 1)
        let paired = MLX.sum(out.features["msa_paired"]!.asType(.float32)).item(Float.self)
        XCTAssertEqual(paired, 6, "the single query row is paired for every token")
    }
}
