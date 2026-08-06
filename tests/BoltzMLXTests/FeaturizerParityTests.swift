import XCTest
import Foundation
import MLX
@testable import BoltzMLX

/// PARITY AGAINST THE REFERENCE IMPLEMENTATION.
///
/// The Swift featurizer must reproduce, bit for bit, what upstream Boltz's Python featurizer emits
/// for the same input — with exactly one principled exception.
///
/// WHY ref_pos IS EXCLUDED FROM BITWISE PARITY. Upstream draws a random conformer per residue
/// instance and applies a random rigid transform, so its ref_pos differs on every run. Measured: two
/// invocations of the reference featurizer on byte-identical input agree on 77 of 78 tensors and
/// disagree on ref_pos by up to 10.7 A. There is no canonical realization to match, and the model is
/// trained to be invariant to it. ref_pos is therefore pinned STRUCTURALLY (occupancy, atom ordering,
/// per-residue internal geometry) — the property the model actually consumes — and the Swift side is
/// additionally held to a stricter standard than upstream: determinism.
///
/// The test INPUT is read from the same fixture YAML that generated the reference, so the two can
/// never drift apart through transcription. Generate references with:
///   boltz-mlx export-features tests/fixtures/<name>.yaml --output <dir>
/// then point BOLTZ_REF_DIR at the parent holding ref_allresidues/ and ref_twochain/.
/// (Bundles are tens of MB, so they are not committed.)
final class FeaturizerParityTests: XCTestCase {

    // MARK: Fixture plumbing

    private struct Fixture {
        let name: String
        let chains: [(chain: String, sequence: String)]
    }

    /// Minimal reader for the fixture subset we write: `id:` / `sequence:` pairs.
    private func fixture(_ name: String) throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "fixtures/\(name).yaml")
        let text = try String(contentsOf: url, encoding: .utf8)
        var chains: [(String, String)] = []
        var pendingID: String?
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("id:") {
                pendingID = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
            } else if line.hasPrefix("sequence:"), let id = pendingID {
                chains.append((id, String(line.dropFirst(9)).trimmingCharacters(in: .whitespaces)))
                pendingID = nil
            }
        }
        XCTAssertFalse(chains.isEmpty, "fixture \(name) parsed no chains")
        return Fixture(name: name, chains: chains)
    }

    private func reference(_ name: String) throws -> FeatureBundle {
        guard let root = ProcessInfo.processInfo.environment["BOLTZ_REF_DIR"] else {
            throw XCTSkip("set BOLTZ_REF_DIR to a directory holding ref_<fixture>/ bundles")
        }
        return try FeatureBundle.load(
            from: URL(fileURLWithPath: root).appending(path: "ref_\(name)"))
    }

    private func featurize(_ f: Fixture) throws -> BoltzFeaturizer.Output {
        try BoltzFeaturizer().featurize(try CanonicalStructure.fromSequences(f.chains))
    }

    /// Every tensor this engine reads, minus ref_pos (see the class comment).
    private static let bitwiseTensors = [
        "asym_id", "atom_backbone_feat", "atom_pad_mask", "atom_to_token",
        "contact_conditioning", "contact_threshold", "cyclic_period", "deletion_mean",
        "deletion_value", "entity_id", "has_deletion", "method_feature", "modified", "mol_type",
        "msa", "msa_mask", "msa_paired", "profile", "ref_atom_name_chars", "ref_charge",
        "ref_element", "ref_space_uid", "res_type", "residue_index", "sym_id",
        "template_ca", "template_cb", "template_frame_rot", "template_frame_t", "template_mask",
        "template_mask_cb", "template_mask_frame", "template_restype", "token_bonds",
        "token_index", "token_pad_mask", "type_bonds", "visibility_ids",
        // Confidence-head features. Not read by the structure path, so they were added when the
        // confidence module needed them — and they are pinned here rather than assumed, because the
        // representative atom is CB (CA only for glycine), which the obvious guess gets wrong.
        "token_to_rep_atom", "token_to_center_atom", "atom_resolved_mask",
    ]

    private func assertBitwiseParity(_ name: String) throws {
        let f = try fixture(name)
        let ref = try reference(name)
        let out = try featurize(f)

        XCTAssertEqual(out.layout.tokenCount, ref.metadata.tokenCount, "token count")
        XCTAssertEqual(out.layout.atomCount, ref.metadata.atomCount, "atom count")

        var mismatches: [String] = []
        for tensor in Self.bitwiseTensors {
            guard let expected = ref.arrays[tensor] else {
                mismatches.append("\(tensor): absent from reference"); continue
            }
            guard let actual = out.features[tensor] else {
                mismatches.append("\(tensor): not emitted"); continue
            }
            guard actual.shape == expected.shape else {
                mismatches.append("\(tensor): shape \(actual.shape) vs \(expected.shape)"); continue
            }
            let d = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32)))
                .item(Float.self)
            if d != 0 { mismatches.append("\(tensor): max|delta| = \(d)") }
        }
        XCTAssertTrue(mismatches.isEmpty,
                      "\(name): \(mismatches.count) of \(Self.bitwiseTensors.count) tensors differ:\n"
                      + mismatches.joined(separator: "\n"))
    }

    // MARK: The gate

    /// Exercises every canonical residue template exactly once — 20 tokens, 167 atoms.
    func testBitwiseParityOnEveryResidueType() throws {
        try assertBitwiseParity("allresidues")
    }

    /// Exercises chain identity (asym/entity/sym), per-chain residue_index reset, and the
    /// pad-to-multiple-of-32 atom axis at a realistic size.
    func testBitwiseParityOnTwoChainComplex() throws {
        try assertBitwiseParity("twochain")
    }

    /// Two IDENTICAL chains. This is the only case that exercises entity/sym sharing: the reference
    /// gives both chains entity_id 0 and sym_id 0/1, so a naive "one entity per chain" implementation
    /// passes both other fixtures and fails here.
    func testBitwiseParityOnHomodimer() throws {
        try assertBitwiseParity("homodimer")
    }

    func testChainTokenRangesMatchTheReferenceChainLayout() throws {
        let f = try fixture("twochain")
        let ref = try reference("twochain")
        let out = try featurize(f)
        let refAsym = ref.arrays["asym_id"]!.reshaped([ref.metadata.tokenCount]).asArray(Int32.self)
        for (i, entry) in out.layout.chainTokenRanges.enumerated() {
            for t in entry.range {
                XCTAssertEqual(Int(refAsym[t]), i,
                               "token \(t) assigned to chain \(entry.chain) but reference says asym \(refAsym[t])")
            }
        }
        XCTAssertEqual(out.layout.chainTokenRanges.map(\.chain), f.chains.map(\.chain))
    }

    // MARK: ref_pos — structural parity

    func testRefPosOccupancyMatchesTheReference() throws {
        let out = try featurize(try fixture("twochain"))
        let ref = try reference("twochain")
        let a = out.features["ref_pos"]!, e = ref.arrays["ref_pos"]!
        XCTAssertEqual(a.shape, e.shape)
        guard a.shape == e.shape else { return }
        let mine = MLX.sum(MLX.abs(a), axis: -1) .> 0
        let theirs = MLX.sum(MLX.abs(e), axis: -1) .> 0
        let disagree = MLX.sum(MLX.notEqual(mine, theirs).asType(.int32)).item(Int32.self)
        XCTAssertEqual(disagree, 0, "ref_pos occupancy differs in \(disagree) atom slots")
    }

    /// The property the model consumes: per-residue reference geometry, independent of the
    /// arbitrary rigid frame upstream applies.
    func testRefPosInternalGeometryIsPhysical() throws {
        let out = try featurize(try fixture("twochain"))
        let n = out.layout.paddedAtomCount
        let xyz = out.features["ref_pos"]!.reshaped([n, 3]).asArray(Float.self)
        let uid = out.features["ref_space_uid"]!.reshaped([n]).asArray(Int32.self)
        let mask = out.features["atom_pad_mask"]!.reshaped([n]).asArray(Float.self)

        var perResidue: [Int32: [SIMD3<Float>]] = [:]
        for i in 0 ..< n where mask[i] == 1 {
            perResidue[uid[i], default: []].append(SIMD3(xyz[i*3], xyz[i*3+1], xyz[i*3+2]))
        }
        XCTAssertEqual(perResidue.count, out.layout.tokenCount)
        func norm(_ v: SIMD3<Float>) -> Float { (v * v).sum().squareRoot() }
        for (res, atoms) in perResidue {
            guard atoms.count >= 3 else { XCTFail("residue \(res) has \(atoms.count) atoms"); continue }
            XCTAssertEqual(norm(atoms[1] - atoms[0]), 1.46, accuracy: 0.10, "N-CA in residue \(res)")
            XCTAssertEqual(norm(atoms[2] - atoms[1]), 1.52, accuracy: 0.10, "CA-C in residue \(res)")
        }
    }

    /// Augmentation must ACTUALLY be applied. Emitting every residue in the component's canonical
    /// frame passed the occupancy and bond-length tests above, yet folded to 14.8 A RMSD against a
    /// 4.4 A noise floor with a badly over-packed interface — so identical orientations are the
    /// specific defect these assertions exist to catch.
    func testEachResidueGetsAnIndependentOrientation() throws {
        let out = try featurize(try fixture("allresidues"))
        let n = out.layout.paddedAtomCount
        let xyz = out.features["ref_pos"]!.reshaped([n, 3]).asArray(Float.self)
        let uid = out.features["ref_space_uid"]!.reshaped([n]).asArray(Int32.self)
        let mask = out.features["atom_pad_mask"]!.reshaped([n]).asArray(Float.self)

        // N->CA direction per residue: if augmentation were skipped these would all coincide.
        var directions: [Int32: SIMD3<Float>] = [:]
        var firstTwo: [Int32: [SIMD3<Float>]] = [:]
        for i in 0 ..< n where mask[i] == 1 {
            let p = SIMD3(xyz[i*3], xyz[i*3+1], xyz[i*3+2])
            if firstTwo[uid[i], default: []].count < 2 { firstTwo[uid[i], default: []].append(p) }
        }
        for (res, pts) in firstTwo where pts.count == 2 {
            let d = pts[1] - pts[0]
            directions[res] = d / (d * d).sum().squareRoot()
        }
        XCTAssertGreaterThan(directions.count, 10)

        // Upstream leaves the LAST residue unaugmented, so exclude it from the spread check.
        let augmented = directions.filter { $0.key != Int32(out.layout.tokenCount - 1) }
        // Use the MEAN pairwise |cos|, not the max: for uniformly random directions cos is uniform
        // on [-1,1] so E|cos| = 0.5, whereas identical orientations give exactly 1.0. The max is an
        // extreme-value statistic — with 19 directions there are 171 pairs and P(|cos| > 0.999) is
        // about 1e-3 each, so a near-parallel pair arises by chance roughly one run in six and says
        // nothing about whether augmentation ran.
        let vals = Array(augmented.values)
        var total: Float = 0, pairs = 0
        for i in 0 ..< vals.count {
            for j in (i + 1) ..< vals.count {
                total += abs((vals[i] * vals[j]).sum()); pairs += 1
            }
        }
        let meanCos = total / Float(pairs)
        XCTAssertLessThan(meanCos, 0.7,
                          "orientations are not independent (mean pairwise |cos| = \(meanCos); "
                          + "0.5 is random, 1.0 means every residue shares one frame)")

        // Per-residue centroids should be spread by the N(0,1) translation, not stacked at the origin.
        var centroids: [SIMD3<Float>] = []
        var byRes: [Int32: [SIMD3<Float>]] = [:]
        for i in 0 ..< n where mask[i] == 1 {
            byRes[uid[i], default: []].append(SIMD3(xyz[i*3], xyz[i*3+1], xyz[i*3+2]))
        }
        for (res, pts) in byRes where res != Int32(out.layout.tokenCount - 1) {
            var c = SIMD3<Float>.zero
            for p in pts { c += p }
            centroids.append(c / Float(pts.count))
        }
        let spread = centroids.map { ($0 * $0).sum().squareRoot() }
        let mean = spread.reduce(0, +) / Float(spread.count)
        // E|N(0,1)^3| ~ 1.6 A; anything near zero means the translation was dropped.
        XCTAssertGreaterThan(mean, 0.8, "per-residue translations look absent (mean |centroid| = \(mean))")
        XCTAssertLessThan(mean, 3.0, "per-residue translations are far larger than s_trans = 1")
    }

    /// Determinism is a property we add over upstream.
    func testFeaturizationIsDeterministic() throws {
        let f = try fixture("twochain")
        let a = try featurize(f).features, b = try featurize(f).features
        for (name, x) in a {
            let d = MLX.max(MLX.abs(x.asType(.float32) - b[name]!.asType(.float32))).item(Float.self)
            XCTAssertEqual(d, 0, accuracy: 0, "\(name) is not deterministic")
        }
    }

    // MARK: Refusals — never silently approximate

    func testNonCanonicalResidueIsRefused() {
        XCTAssertThrowsError(try CanonicalStructure.fromSequences([("A", "ACDX")])) { err in
            guard case BoltzFeaturizerError.nonCanonicalResidue = err else {
                return XCTFail("expected nonCanonicalResidue, got \(err)")
            }
        }
    }

    func testResidueWithoutCAIsReportedAndRefused() throws {
        let s = CanonicalStructure.fromResidues([
            ("A", 1, nil, "ALA", [.init(name: "N", position: .zero), .init(name: "C", position: .zero)]),
        ])
        XCTAssertTrue(s.diagnostics.contains { $0.kind == .missingBackbone })
        XCTAssertThrowsError(try BoltzFeaturizer().featurize(s))
    }

    func testInsertionCodedResiduesStayDistinct() {
        let s = CanonicalStructure.fromResidues([
            ("H", 100, nil, "ALA", []), ("H", 100, "A", "GLY", []),
        ])
        XCTAssertEqual(s.residueCount, 2, "insertion-coded residues must not be merged")
        XCTAssertEqual(s.identities, ["H/100", "H/100A"])
        XCTAssertEqual(s.tokenIndex["H/100A"], 1)
    }
}
