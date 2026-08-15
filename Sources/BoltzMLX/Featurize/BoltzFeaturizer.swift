// BoltzFeaturizer.swift — build a Boltz-2 feature bundle in Swift, no Python.
//
// SCOPE. Protein-only, canonical-20, one or more polymer chains. A real per-chain MSA is supported —
// see `featurize(_:alignments:)` and `MSAAlignment` — and a chain with no alignment gets upstream's
// depth-1 dummy MSA, which is the designed-binder case (no homologs by construction). Mixed is the
// design case, not an edge case. Ligands, nucleic acids, modified residues and real templates are
// out of scope and are REFUSED rather than silently approximated.
//
// HOW THE ENCODINGS WERE ESTABLISHED. Every convention below was decoded from Python-generated
// reference bundles rather than inferred from upstream source, then cross-checked against the
// component pickles. The load-bearing findings:
//
//   * 39 of the 78 tensors in a reference bundle are actually read by this Swift engine; the rest are
//     training targets or ligand/chirality machinery. Only those 39 are emitted here.
//   * token count == residue count for protein (ligands would break this: one token per ATOM).
//   * residue_index is 0-based and RESETS per chain; token_index is 0-based and continuous.
//   * The atom axis is real atoms (GLY 4 ... TRP 14, OXT excluded) padded up to a multiple of 32,
//     and EVERY tensor is exactly zero in the padded region.
//   * token_bonds and type_bonds are all-zero for polymers — they encode ligand/covalent bonds only.
//   * The template block is all-zero EXCEPT template_restype, which is one-hot at index 0. That is
//     not a no-op: the template stack consumes it and contributes a constant nonzero bias, so it
//     must be emitted, not skipped.
//   * With msa: empty, the dummy MSA is just the query — msa == restype indices, profile == the
//     res_type one-hot as float, masks constant.
//
// ref_pos: SEEDED AUGMENTATION, NOT A FIXED CONFORMER.
//
// Upstream applies AF3 Algorithm 19 per residue — center on the residue centroid, apply a uniform
// random rotation, add a translation ~N(0, s_trans=1) — so its ref_pos changes on EVERY run: two
// invocations on byte-identical input agree on 77 of 78 tensors and differ on ref_pos by up to
// 10.7 A. There is therefore no canonical realization to match bitwise.
//
// AN EARLIER VERSION OF THIS FILE SKIPPED THE AUGMENTATION, arguing the model is invariant to
// ref_pos. THAT WAS WRONG, and it was measured wrong rather than argued wrong: emitting every
// residue in the component's own canonical frame folded to 14.8 / 13.0 A all-atom RMSD against a
// 4.4 A Python-vs-Python noise floor, with a systematically OVER-PACKED interface (radius of
// gyration 16.63 vs 18.65/18.32; chain separation 17.13 A vs 21.22/21.44; 821 inter-chain contacts
// under 5 A vs 650/507). Giving every residue an identical orientation is out of distribution, and
// an over-docked prediction biases an interface gate toward FALSE PASSES — the exact failure mode
// min_ipSAE exists to catch. So the augmentation is reproduced.
//
// It is driven by a deterministic PRNG seeded per residue index instead of a global RNG, which puts
// ref_pos in the same distribution upstream samples from while making it reproducible — a property
// upstream does not have. Parity for ref_pos is therefore structural (occupancy, atom ordering,
// per-residue internal geometry, which a rigid transform preserves exactly), not bitwise.
//
// The off-by-one is reproduced deliberately: upstream loops `for i in range(max(ref_space_uid))`,
// so the LAST residue is never augmented and stays in its canonical frame. Matching that keeps the
// distribution identical rather than merely similar.
import Foundation
import MLX

public enum BoltzFeaturizerError: Error, CustomStringConvertible {
    case nonCanonicalResidue(chain: String, resSeq: Int, code: String)
    case noChains
    case emptyChain(String)
    case blockingDiagnostics([CanonicalDiagnostic])
    case msaLengthMismatch(chain: String, expected: Int, found: Int)
    case msaQueryMismatch(chain: String, positions: [Int])

    public var description: String {
        switch self {
        case .nonCanonicalResidue(let c, let r, let code):
            return "residue \(code) at \(c)/\(r) is not one of the canonical 20; "
                 + "ligands, nucleic acids and modified residues are not supported"
        case .noChains: return "no chains supplied"
        case .emptyChain(let c): return "chain \(c) has no residues"
        case .blockingDiagnostics(let d):
            return "input cannot be featurized: " + d.map(\.description).joined(separator: "; ")
        case .msaLengthMismatch(let c, let expected, let found):
            return "the alignment for chain \(c) spans \(found) columns but the chain has "
                 + "\(expected) residues; the a3m must be the alignment OF this sequence"
        case .msaQueryMismatch(let c, let positions):
            return "the alignment for chain \(c) does not match its sequence at "
                 + "\(positions.count) position(s) (first: \(positions.prefix(5).map(String.init).joined(separator: ", ")))"
        }
    }
}

/// Deterministic PRNG (splitmix64) + the Gaussian and uniform-rotation draws Algorithm 19 needs.
/// Seeded per residue so ref_pos is reproducible without a global RNG, and so one residue's draw
/// cannot shift another's — which is what makes the output independent of iteration order.
struct AugmentRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed &* 0x9E37_79B9_7F4A_7C15 &+ 0x1234_5678_9ABC_DEF }

    private mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    private mutating func uniform() -> Float {
        Float(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal via Box-Muller.
    mutating func gaussian() -> Float {
        let u1 = Swift.max(uniform(), Float.leastNormalMagnitude)
        let u2 = uniform()
        return (-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * .pi * u2)
    }

    /// Uniform random rotation, as a 3x3 row-major matrix, from a normalised random quaternion.
    mutating func rotation() -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) {
        var q = SIMD4<Float>(gaussian(), gaussian(), gaussian(), gaussian())
        var n = (q * q).sum().squareRoot()
        if n < 1e-6 { q = SIMD4<Float>(1, 0, 0, 0); n = 1 }
        q /= n
        let w = q.x, x = q.y, y = q.z, z = q.w
        return (
            SIMD3(1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)),
            SIMD3(2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)),
            SIMD3(2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y))
        )
    }
}

public struct BoltzFeaturizer {

    /// Chain token ranges, carried out alongside the features so min_ipSAE can split the PAE matrix
    /// without re-deriving them (and therefore without any chance of disagreeing with the features).
    public struct Layout: Sendable {
        public let tokenCount: Int
        public let atomCount: Int
        public let paddedAtomCount: Int
        public let chainTokenRanges: [(chain: String, range: Range<Int>)]
        /// Token index -> the host's residue identity, so a caller can always map back.
        public let identities: [String]
        /// Rows in the paired MSA. 1 when no chain supplied an alignment.
        public let msaDepth: Int
    }

    /// `@unchecked Sendable` for the same reason `FeatureBundle` is: MLXArray is not Sendable, but
    /// this value is immutable after construction and is only ever handed across to the predictor
    /// actor, never mutated afterwards. Without it, `predictor.predict(featurized:)` — the whole
    /// point of the type — fails to compile under Swift 6 concurrency checking.
    public struct Output: @unchecked Sendable {
        public let features: [String: MLXArray]
        public let layout: Layout

        /// Metadata equivalent to what a Python-exported bundle carries, so in-memory features go
        /// through exactly the same MemoryPlanner validation as a loaded bundle. The MSA depth is
        /// the REAL one — reporting 1 unconditionally, as this did before alignments were supported,
        /// makes the planner's msa_depth check vacuous and lets a deep alignment OOM instead of
        /// being refused.
        public var metadata: FeatureMetadata {
            FeatureMetadata(
                schemaVersion: ArtifactManifest.supportedSchemaVersion,
                sampleID: "swift-featurized",
                tokenCount: layout.tokenCount,
                atomCount: layout.atomCount,
                msaDepth: layout.msaDepth
            )
        }
    }

    /// When true, ref_pos is emitted in each component's canonical frame with NO random
    /// roto-translation. This is NOT the production setting — it measurably over-packs the fold
    /// (see the file comment). It exists solely for matched-noise cross-backend comparison, where
    /// upstream's augmentation is patched to identity too so both trajectories are comparable.
    public let identityAugmentation: Bool

    public init(identityAugmentation: Bool = false) {
        self.identityAugmentation = identityAugmentation
    }

    /// Upstream requires an alignment's first row to BE the chain's sequence, and tolerates exactly
    /// two disagreements: the structure says MET where the alignment says UNK, or the structure says
    /// UNK anywhere. In both cases it overwrites the alignment's query row with the structure's
    /// residues. Anything else makes it discard the alignment and fall back to a dummy MSA with a
    /// printed warning.
    ///
    /// THIS THROWS WHERE UPSTREAM FALLS BACK, deliberately. Silently dropping the target's alignment
    /// is the worst outcome available here: the target folds worse, every min_ipSAE computed against
    /// it is measuring the wrong complex, and nothing in the output says so.
    private func reconcileQueryRow(
        _ alignment: MSAAlignment,
        with queryTokens: [Int32],
        chain: String
    ) throws -> MSAAlignment {
        let queryRow = alignment.rows[0]
        let mismatched = (0 ..< queryTokens.count).filter { queryRow[$0] != queryTokens[$0] }
        guard !mismatched.isEmpty else { return alignment }

        let unknown = MSAAlignment.unknownToken
        let methionine = Int32(AAResidueTemplates.template(threeLetter: "MET")!.restype)
        let structureAllUnknown = mismatched.allSatisfy { queryTokens[$0] == unknown }
        let methionineVersusUnknown = mismatched.allSatisfy {
            queryTokens[$0] == methionine && queryRow[$0] == unknown
        }
        guard structureAllUnknown || methionineVersusUnknown else {
            throw BoltzFeaturizerError.msaQueryMismatch(chain: chain, positions: mismatched)
        }
        var rows = alignment.rows
        rows[0] = queryTokens
        return MSAAlignment(rows: rows, taxonomies: alignment.taxonomies,
                            insertionCounts: alignment.insertionCounts)
    }

    /// - Parameter alignments: per-chain MSA, keyed by host chain name. A chain absent from this
    ///   dictionary gets upstream's `dummy_msa`: a single row that is its own query. That asymmetry
    ///   is the design case — the target carries an alignment because many targets do not fold
    ///   correctly without one, while a designed binder has no homologs to align.
    public func featurize(
        _ structure: CanonicalStructure,
        alignments: [String: MSAAlignment] = [:]
    ) throws -> Output {
        guard !structure.chains.isEmpty else { throw BoltzFeaturizerError.noChains }
        for c in structure.chains where c.residues.isEmpty {
            throw BoltzFeaturizerError.emptyChain(c.hostChain)
        }
        if structure.hasBlockingDiagnostics {
            throw BoltzFeaturizerError.blockingDiagnostics(
                structure.diagnostics.filter { $0.kind == .missingBackbone || $0.kind == .noTemplateAtoms })
        }

        let residues = structure.orderedResidues
        let nTok = residues.count
        let templates = try residues.map { r -> AAResidueTemplate in
            guard let t = AAResidueTemplates.template(threeLetter: r.threeLetter) else {
                throw BoltzFeaturizerError.nonCanonicalResidue(
                    chain: r.hostChain, resSeq: r.hostResSeq, code: r.threeLetter)
            }
            return t
        }
        let nAtom = templates.reduce(0) { $0 + $1.atoms.count }
        let pad = AAResidueTemplates.atomPadMultiple
        let nPad = (nAtom + pad - 1) / pad * pad
        let R = AAResidueTemplates.restypeCount

        // ---- chain-level identity ----
        // entity_id indexes UNIQUE sequences in order of first appearance; sym_id counts occurrences
        // within one entity (both chains distinct in the reference, so 0/1 and 0/0).
        var entityOfSequence: [String: Int] = [:]
        var occurrencesOfEntity: [Int: Int] = [:]
        var asymOf: [Int] = [], entityOf: [Int] = [], symOf: [Int] = [], resIdxOf: [Int] = []
        for (ci, chain) in structure.chains.enumerated() {
            let seq = chain.sequence
            let entity: Int
            if let e = entityOfSequence[seq] { entity = e } else {
                entity = entityOfSequence.count
                entityOfSequence[seq] = entity
            }
            let sym = occurrencesOfEntity[entity, default: 0]
            occurrencesOfEntity[entity] = sym + 1
            for i in 0 ..< chain.residues.count {
                asymOf.append(ci); entityOf.append(entity); symOf.append(sym); resIdxOf.append(i)
            }
        }

        // ---- token axis ----
        var restype = [Int32](repeating: 0, count: nTok)
        for (i, t) in templates.enumerated() { restype[i] = Int32(t.restype) }

        var resTypeOneHot = [Int32](repeating: 0, count: nTok * R)
        for i in 0 ..< nTok {
            resTypeOneHot[i * R + Int(restype[i])] = 1
        }

        var f: [String: MLXArray] = [:]
        func i64(_ v: [Int32], _ s: [Int]) -> MLXArray { MLXArray(v, s).asType(.int64) }
        func f32(_ v: [Float], _ s: [Int]) -> MLXArray { MLXArray(v, s) }

        f["asym_id"] = i64(asymOf.map(Int32.init), [1, nTok])
        f["entity_id"] = i64(entityOf.map(Int32.init), [1, nTok])
        f["sym_id"] = i64(symOf.map(Int32.init), [1, nTok])
        f["residue_index"] = i64(resIdxOf.map(Int32.init), [1, nTok])
        f["token_index"] = i64((0 ..< nTok).map(Int32.init), [1, nTok])
        f["mol_type"] = i64([Int32](repeating: 0, count: nTok), [1, nTok])
        f["modified"] = i64([Int32](repeating: 0, count: nTok), [1, nTok])
        f["method_feature"] = i64([Int32](repeating: 1, count: nTok), [1, nTok])
        f["token_pad_mask"] = f32([Float](repeating: 1, count: nTok), [1, nTok])
        f["cyclic_period"] = f32([Float](repeating: 0, count: nTok), [1, nTok])
        f["res_type"] = i64(resTypeOneHot, [1, nTok, R])

        // ---- MSA block ----
        // Per chain: the supplied alignment, or a synthetic depth-1 alignment of its own sequence.
        let chainAlignments: [MSAAlignment] = try structure.chains.enumerated().map { ci, chain in
            let queryTokens = (0 ..< nTok).filter { asymOf[$0] == ci }.map { restype[$0] }
            guard let supplied = alignments[chain.hostChain] else {
                return MSAAlignment.singleSequence(queryTokens)
            }
            guard supplied.queryLength == queryTokens.count else {
                throw BoltzFeaturizerError.msaLengthMismatch(
                    chain: chain.hostChain, expected: queryTokens.count, found: supplied.queryLength)
            }
            return try reconcileQueryRow(supplied, with: queryTokens, chain: chain.hostChain)
        }

        let msaRows = MSAPairing.rows(alignments: chainAlignments)
        let depth = msaRows.count

        // Expand the row plan into token space (`_prepare_msa_arrays_inner`). Gaps stay at the gap
        // token; `msa_paired` is written for EVERY token including gapped ones, which is why it is
        // filled from the row rather than from whether a sequence was found.
        var msa = [Int32](repeating: MSAAlignment.gapToken, count: depth * nTok)
        var paired = [Float](repeating: 0, count: depth * nTok)
        for (r, row) in msaRows.enumerated() {
            for t in 0 ..< nTok {
                let chain = asymOf[t]
                paired[r * nTok + t] = row.isPaired[chain] ? 1 : 0
                if let sequence = row.sequenceIndex[chain] {
                    msa[r * nTok + t] = chainAlignments[chain].rows[sequence][resIdxOf[t]]
                }
            }
        }

        // `profile` is the mean one-hot over the msa COLUMN, not the query one-hot. With depth 1 the
        // two coincide, which is why the single-sequence path could get away with the latter.
        //
        // COUNT FIRST, THEN DIVIDE ONCE. Upstream is `one_hot(msa).float().mean(dim=0)`: an exact
        // integer-valued float sum followed by a single division. Accumulating `1/depth` per row
        // instead drifts — measurably, 2.1e-06 at depth 249 — and breaks bitwise parity.
        var counts = [Int32](repeating: 0, count: nTok * R)
        for r in 0 ..< depth {
            for t in 0 ..< nTok {
                counts[t * R + Int(msa[r * nTok + t])] += 1
            }
        }
        let profile = counts.map { Float($0) / Float(depth) }

        f["msa"] = i64(msa, [1, depth, nTok])
        f["msa_mask"] = i64([Int32](repeating: 1, count: depth * nTok), [1, depth, nTok])
        f["msa_paired"] = f32(paired, [1, depth, nTok])
        f["profile"] = f32(profile, [1, nTok, R])

        // DELETION FEATURES ARE IDENTICALLY ZERO, MATCHING UPSTREAM — and upstream is wrong.
        // `construct_paired_msa` builds its deletion lookup with `chain_deletions` rebound to a
        // slice inside the per-sequence loop, so after sequence 0 (the query, which by a3m
        // construction has no insertions and so yields an empty slice) every subsequent slice is
        // empty and the lookup stays empty. Verified against a 249-row fixture in which 217 rows do
        // carry insertions and the reference still emits all zeros. Emitting the real counts would
        // feed the model a distribution it was never trained on, so the parsed
        // `MSAAlignment.insertionCounts` are deliberately not used here; see
        // MSAFeaturizerParityTests for the pinned assertion.
        f["deletion_mean"] = f32([Float](repeating: 0, count: nTok), [1, nTok])
        f["deletion_value"] = f32([Float](repeating: 0, count: depth * nTok), [1, depth, nTok])
        f["has_deletion"] = MLXArray([Bool](repeating: false, count: depth * nTok), [1, depth, nTok])

        // ---- pair axis: no covalent/ligand bonds in a pure polymer complex ----
        f["token_bonds"] = f32([Float](repeating: 0, count: nTok * nTok), [1, nTok, nTok, 1])
        f["type_bonds"] = i64([Int32](repeating: 0, count: nTok * nTok), [1, nTok, nTok])
        f["contact_threshold"] = f32([Float](repeating: 0, count: nTok * nTok), [1, nTok, nTok])
        var contact = [Int32](repeating: 0, count: nTok * nTok * 5)
        for p in 0 ..< (nTok * nTok) { contact[p * 5] = 1 }   // one-hot at bin 0 = unconstrained
        f["contact_conditioning"] = i64(contact, [1, nTok, nTok, 5])

        // ---- templates: all zero EXCEPT template_restype, one-hot at 0 (a real, constant bias) ----
        f["template_ca"] = f32([Float](repeating: 0, count: nTok * 3), [1, 1, nTok, 3])
        f["template_cb"] = f32([Float](repeating: 0, count: nTok * 3), [1, 1, nTok, 3])
        f["template_frame_t"] = f32([Float](repeating: 0, count: nTok * 3), [1, 1, nTok, 3])
        f["template_frame_rot"] = f32([Float](repeating: 0, count: nTok * 9), [1, 1, nTok, 3, 3])
        f["template_mask"] = f32([Float](repeating: 0, count: nTok), [1, 1, nTok])
        f["template_mask_cb"] = f32([Float](repeating: 0, count: nTok), [1, 1, nTok])
        f["template_mask_frame"] = f32([Float](repeating: 0, count: nTok), [1, 1, nTok])
        f["visibility_ids"] = f32([Float](repeating: 0, count: nTok), [1, 1, nTok])
        var tRest = [Int32](repeating: 0, count: nTok * R)
        for i in 0 ..< nTok { tRest[i * R] = 1 }
        f["template_restype"] = i64(tRest, [1, 1, nTok, R])

        // ---- atom axis (ragged real atoms, zero-padded to a multiple of 32) ----
        var padMask = [Float](repeating: 0, count: nPad)
        var spaceUID = [Int32](repeating: 0, count: nPad)
        var charge = [Float](repeating: 0, count: nPad)
        var refPos = [Float](repeating: 0, count: nPad * 3)
        var element = [Int32](repeating: 0, count: nPad * 128)
        var nameChars = [Int32](repeating: 0, count: nPad * 4 * 64)
        var backbone = [Int32](repeating: 0, count: nPad * 17)
        var atomToToken = [Int32](repeating: 0, count: nPad * nTok)
        // Confidence-head features. Not read by the structure path, which is why they were absent
        // until the confidence module needed them.
        var tokenToRepAtom = [Int32](repeating: 0, count: nTok * nPad)
        var tokenToCenterAtom = [Int32](repeating: 0, count: nTok * nPad)
        var atomResolved = [Bool](repeating: false, count: nPad)

        // Algorithm 19, per residue: centre on the residue centroid, uniform random rotation, then a
        // translation ~N(0, sTrans). Upstream's loop stops one short of the final residue, leaving it
        // in its canonical frame; reproduced so the distribution matches exactly.
        let sTrans: Float = 1.0
        let lastToken = templates.count - 1
        func augmented(_ atoms: [AAAtomTemplate], token: Int) -> [SIMD3<Float>] {
            let raw = atoms.map(\.refPos)
            var centroid = SIMD3<Float>.zero
            for p in raw { centroid += p }
            centroid /= Float(raw.count)
            // Identity augmentation still CENTRES: upstream's center_random_augmentation is called
            // with centering: true, so patching only the rotation/translation out leaves the
            // centred conformer. Returning the raw conformer here instead would differ from the
            // patched Python side by a per-residue translation.
            guard !identityAugmentation else { return raw.map { $0 - centroid } }
            guard token != lastToken else { return raw }
            var rng = AugmentRNG(seed: UInt64(token) &+ 1)
            let (r0, r1, r2) = rng.rotation()
            let shift = SIMD3<Float>(rng.gaussian(), rng.gaussian(), rng.gaussian()) * sTrans
            return raw.map { p in
                let c = p - centroid
                return SIMD3((r0 * c).sum(), (r1 * c).sum(), (r2 * c).sum()) + shift
            }
        }

        var a = 0
        for (tok, t) in templates.enumerated() {
            // One-hot over the PADDED atom axis, hence written with the running atom offset.
            tokenToRepAtom[tok * nPad + a + t.representativeAtomIndex] = 1
            tokenToCenterAtom[tok * nPad + a + t.centerAtomIndex] = 1
            let positions = augmented(t.atoms, token: tok)
            for (k, at) in t.atoms.enumerated() {
                padMask[a] = 1
                spaceUID[a] = Int32(tok)
                charge[a] = at.formalCharge
                refPos[a * 3 + 0] = positions[k].x
                refPos[a * 3 + 1] = positions[k].y
                refPos[a * 3 + 2] = positions[k].z
                element[a * 128 + at.atomicNumber] = 1
                backbone[a * 17 + at.backboneFeat] = 1
                atomToToken[a * nTok + tok] = 1
                atomResolved[a] = true
                // Atom names are right-padded to 4 characters and encoded as ord(c) - 32.
                let chars = Array(at.name.utf8)
                for c in 0 ..< 4 {
                    let code = c < chars.count ? Int(chars[c]) - 32 : 0
                    nameChars[(a * 4 + c) * 64 + code] = 1
                }
                a += 1
            }
        }

        f["atom_pad_mask"] = f32(padMask, [1, nPad])
        f["ref_space_uid"] = i64(spaceUID, [1, nPad])
        f["ref_charge"] = f32(charge, [1, nPad])
        f["ref_pos"] = f32(refPos, [1, nPad, 3])
        f["ref_element"] = i64(element, [1, nPad, 128])
        f["ref_atom_name_chars"] = i64(nameChars, [1, nPad, 4, 64])
        f["atom_backbone_feat"] = i64(backbone, [1, nPad, 17])
        f["atom_to_token"] = i64(atomToToken, [1, nPad, nTok])
        f["token_to_rep_atom"] = i64(tokenToRepAtom, [1, nTok, nPad])
        f["token_to_center_atom"] = i64(tokenToCenterAtom, [1, nTok, nPad])
        f["atom_resolved_mask"] = MLXArray(atomResolved, [1, nPad])

        return Output(features: f,
                      layout: .init(tokenCount: nTok, atomCount: nAtom, paddedAtomCount: nPad,
                                    chainTokenRanges: structure.chainTokenRanges,
                                    identities: structure.identities,
                                    msaDepth: depth))
    }
}
