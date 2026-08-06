// CanonicalStructure.swift — the parse/validate/canonicalize layer, deliberately MLX-free.
//
// WHY THIS IS SEPARATE FROM FEATURIZATION. Tensor emission differs irreconcilably between models
// (Boltz uses a ragged real-atom axis padded to a multiple of 32; RFdiffusion3 uses a dense 14-slot
// layout with virtual atoms), but the layer BENEATH it — turning a host selection into a validated,
// ordered set of residues with stable identity — is identical work, and it is where every seam bug
// in this pipeline has actually lived: silent non-canonical drops, insertion-code collapse, altloc
// ambiguity, and a host index space that diverges from the featurizer's after validation.
//
// Two invariants make that last class of bug unrepresentable rather than merely avoided:
//   1. Validation is EXPLICIT. Anything discarded is reported in `diagnostics`; nothing is dropped
//      silently. A caller can refuse to proceed.
//   2. Identity SURVIVES. Every canonical residue keeps the host's (chain, resSeq, insCode), so any
//      index computed downstream — hotspots, gate metrics, per-chain ranges — can be mapped back
//      exactly. Indices are never implied by position in a list the featurizer may have filtered.
//
// MLX-free on purpose: it needs no GPU, is unit-testable without weights or a metallib, and adds no
// node to the mlx-swift version lockstep shared by the engine packages.
import Foundation

/// One heavy atom as supplied by the host.
public struct CanonicalAtom: Sendable, Equatable {
    public let name: String
    public let position: SIMD3<Float>
    public init(name: String, position: SIMD3<Float>) {
        self.name = name
        self.position = position
    }
}

/// A residue that passed validation, with its host identity preserved.
public struct CanonicalResidue: Sendable, Equatable {
    public let threeLetter: String
    public let oneLetter: Character
    /// Host identity — the numbering the USER sees, never renumbered by this layer.
    public let hostChain: String
    public let hostResSeq: Int
    public let hostInsCode: Character?
    /// Resolved heavy atoms as supplied. May be empty when only a sequence was given.
    public let atoms: [CanonicalAtom]

    public init(threeLetter: String, oneLetter: Character, hostChain: String, hostResSeq: Int,
                hostInsCode: Character? = nil, atoms: [CanonicalAtom] = []) {
        self.threeLetter = threeLetter
        self.oneLetter = oneLetter
        self.hostChain = hostChain
        self.hostResSeq = hostResSeq
        self.hostInsCode = hostInsCode
        self.atoms = atoms
    }

    /// Stable key for mapping any downstream index back to the host's numbering.
    public var identity: String {
        "\(hostChain)/\(hostResSeq)\(hostInsCode.map { String($0) } ?? "")"
    }
}

/// One polymer chain.
public struct CanonicalChain: Sendable, Equatable {
    public let hostChain: String
    public let residues: [CanonicalResidue]
    public init(hostChain: String, residues: [CanonicalResidue]) {
        self.hostChain = hostChain
        self.residues = residues
    }
    public var sequence: String { String(residues.map(\.oneLetter)) }
}

/// Something was discarded or altered. Surfaced, never swallowed.
public struct CanonicalDiagnostic: Sendable, Equatable, CustomStringConvertible {
    public enum Kind: String, Sendable {
        case nonCanonicalResidue     // not one of the 20 standard amino acids
        case missingBackbone         // no CA — crashes downstream featurizers
        case noTemplateAtoms         // every atom fell outside the residue template
        case insertionCode           // insertion-coded residue kept as distinct (never merged)
        case duplicateAtom           // repeated atom name within one residue
        case unknownAtomName         // atom not in the residue template
        case emptyChain
    }
    public let kind: Kind
    public let identity: String
    public let detail: String
    public var description: String { "[\(kind.rawValue)] \(identity): \(detail)" }
}

/// A validated structure plus the full record of what validation did.
public struct CanonicalStructure: Sendable {
    public let chains: [CanonicalChain]
    public let diagnostics: [CanonicalDiagnostic]

    public init(chains: [CanonicalChain], diagnostics: [CanonicalDiagnostic] = []) {
        self.chains = chains
        self.diagnostics = diagnostics
    }

    public var residueCount: Int { chains.reduce(0) { $0 + $1.residues.count } }

    /// Residues in featurization order (chain order, then sequence order) — the ONLY ordering any
    /// downstream index should be derived from.
    public var orderedResidues: [CanonicalResidue] { chains.flatMap(\.residues) }

    /// Token index -> host identity. This is the map that makes index divergence impossible: a
    /// caller holding a token index can always recover which residue the user actually selected.
    public var identities: [String] { orderedResidues.map(\.identity) }

    /// Host identity -> token index, for translating a user selection into token space.
    public var tokenIndex: [String: Int] {
        var m = [String: Int]()
        for (i, r) in orderedResidues.enumerated() { m[r.identity] = i }
        return m
    }

    /// Half-open token range per chain, in featurization order. min_ipSAE needs exactly this to
    /// split the PAE matrix, and deriving it here means it can never disagree with the features.
    public var chainTokenRanges: [(chain: String, range: Range<Int>)] {
        var out: [(String, Range<Int>)] = []
        var start = 0
        for c in chains {
            out.append((c.hostChain, start ..< (start + c.residues.count)))
            start += c.residues.count
        }
        return out
    }

    public var hasBlockingDiagnostics: Bool {
        diagnostics.contains { $0.kind == .missingBackbone || $0.kind == .noTemplateAtoms }
    }

    // MARK: Construction from sequences (no coordinates — the binder case)

    /// Build from one-letter sequences, e.g. designed binder + target sequence.
    /// Throws on any residue outside the canonical 20 rather than silently dropping it.
    public static func fromSequences(_ chains: [(chain: String, sequence: String)]) throws -> CanonicalStructure {
        var built: [CanonicalChain] = []
        var diags: [CanonicalDiagnostic] = []
        for (chainID, seq) in chains {
            if seq.isEmpty {
                diags.append(.init(kind: .emptyChain, identity: chainID, detail: "no residues"))
                continue
            }
            var residues: [CanonicalResidue] = []
            for (i, ch) in seq.enumerated() {
                guard let t = AAResidueTemplates.template(oneLetter: ch) else {
                    throw BoltzFeaturizerError.nonCanonicalResidue(
                        chain: chainID, resSeq: i, code: String(ch))
                }
                let three = AAResidueTemplates.byThreeLetter.first { $0.value.oneLetter == ch }?.key ?? "UNK"
                residues.append(.init(threeLetter: three, oneLetter: ch,
                                      hostChain: chainID, hostResSeq: i))
            }
            built.append(.init(hostChain: chainID, residues: residues))
        }
        return .init(chains: built, diagnostics: diags)
    }

    // MARK: Construction from host-supplied residues (the selection case)

    /// Build from residues the host already has in memory, validating rather than dropping.
    /// Insertion codes are PRESERVED as distinct residues — never merged, which is the behaviour
    /// that silently corrupts antibody numbering elsewhere in this pipeline.
    public static func fromResidues(
        _ input: [(chain: String, resSeq: Int, insCode: Character?, threeLetter: String, atoms: [CanonicalAtom])]
    ) -> CanonicalStructure {
        var byChain: [String: [CanonicalResidue]] = [:]
        var order: [String] = []
        var diags: [CanonicalDiagnostic] = []

        for r in input {
            let ident = "\(r.chain)/\(r.resSeq)\(r.insCode.map { String($0) } ?? "")"
            guard let t = AAResidueTemplates.template(threeLetter: r.threeLetter) else {
                diags.append(.init(kind: .nonCanonicalResidue, identity: ident,
                                   detail: "\(r.threeLetter) is not one of the canonical 20; excluded"))
                continue
            }
            if r.insCode != nil {
                diags.append(.init(kind: .insertionCode, identity: ident,
                                   detail: "kept as a distinct residue"))
            }
            let allowed = Set(t.atoms.map(\.name))
            var seen = Set<String>()
            var kept: [CanonicalAtom] = []
            for a in r.atoms {
                if !allowed.contains(a.name) {
                    diags.append(.init(kind: .unknownAtomName, identity: ident,
                                       detail: "\(a.name) not in the \(r.threeLetter) template; excluded"))
                    continue
                }
                if !seen.insert(a.name).inserted {
                    diags.append(.init(kind: .duplicateAtom, identity: ident,
                                       detail: "\(a.name) repeated; first occurrence kept"))
                    continue
                }
                kept.append(a)
            }
            if !r.atoms.isEmpty && kept.isEmpty {
                diags.append(.init(kind: .noTemplateAtoms, identity: ident,
                                   detail: "no supplied atom matched the template"))
            }
            if !r.atoms.isEmpty && !seen.contains("CA") {
                diags.append(.init(kind: .missingBackbone, identity: ident, detail: "no CA atom"))
            }
            if byChain[r.chain] == nil { order.append(r.chain) }
            byChain[r.chain, default: []].append(
                .init(threeLetter: r.threeLetter.uppercased(), oneLetter: t.oneLetter,
                      hostChain: r.chain, hostResSeq: r.resSeq, hostInsCode: r.insCode, atoms: kept))
        }
        return .init(chains: order.map { .init(hostChain: $0, residues: byChain[$0] ?? []) },
                     diagnostics: diags)
    }
}
