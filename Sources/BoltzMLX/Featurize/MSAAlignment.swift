// MSAAlignment.swift — a3m parsing and cross-chain MSA pairing, with no MLX dependency.
//
// WHY THE FEATURIZER NEEDS THIS AT ALL. A designed binder needs no alignment, but the TARGET often
// does: without an MSA many targets do not fold to their known structure, and a binder scored
// against a mis-folded target is scored against the wrong thing. So the alignment is a property of
// the target chain, and the interesting work is what happens at the seam between a chain that has
// one and a chain that does not.
//
// Ported from:
//   src/boltz/data/parse/a3m.py::_parse_a3m               (parsing)
//   src/boltz/data/feature/featurizerv2.py::construct_paired_msa   (pairing)
//   src/boltz/data/feature/featurizerv2.py::_prepare_msa_arrays_inner (token expansion)
//
// THE WHOLE PATH IS DETERMINISTIC AT INFERENCE. `process()` takes `max_seqs_batch = max_seqs` and
// `msa_sampling = training and msa_sampling` when `training=False`, so neither the row subsample nor
// the depth is drawn from the RNG. That is why these features can be pinned BITWISE against a
// Python export, unlike `ref_pos` which is randomly augmented per instance.
//
// TWO UPSTREAM DEFECTS ARE REPRODUCED DELIBERATELY, both marked `UPSTREAM BUG` below. They are
// reproduced rather than fixed because the reference implementation defines the input distribution
// the checkpoint was trained and validated on; silently feeding the model better features than
// upstream does is still a parity break, and one that would be invisible in the output.
import Foundation

public enum MSAParseError: Error, Equatable {
    /// No sequence lines at all.
    case empty
    /// A row's aligned column count differs from the query's. Upstream does not check this and would
    /// read past the row into the next sequence's residues; throwing is a deliberate divergence.
    case rowLengthMismatch(row: Int, expected: Int, found: Int)
}

/// One chain's alignment: token ids per aligned column, plus the insertion counts a3m encodes as
/// lowercase runs.
public struct MSAAlignment: Sendable, Equatable {
    /// The gap token, `const.token_ids["-"]`. Also the fill value for a chain that has no sequence
    /// in a given paired row.
    public static let gapToken: Int32 = 1
    /// `const.token_ids["UNK"]` — where B/J/O/U/X/Z all land.
    public static let unknownToken: Int32 = 22

    /// `depth` rows of `queryLength` token ids. Row 0 is the query.
    public let rows: [[Int32]]
    /// Per-row taxonomy id, `-1` when unannotated. Only rows with a real taxon can ever be paired
    /// across chains, so an a3m parsed without a taxonomy database yields no pairing whatsoever.
    public let taxonomies: [Int32]
    /// `depth × queryLength`: how many inserted residues sit immediately BEFORE each column.
    ///
    /// PARSED BUT NOT FED TO THE MODEL, on purpose — see `BoltzFeaturizer`'s MSA block. Upstream
    /// drops these before they reach the features, so emitting them would be a parity break. Kept
    /// here because it costs nothing and is the seam to wire up if upstream ever fixes that.
    public let insertionCounts: [[Int32]]

    public var depth: Int { rows.count }
    public var queryLength: Int { rows.first?.count ?? 0 }

    public init(rows: [[Int32]], taxonomies: [Int32], insertionCounts: [[Int32]]) {
        self.rows = rows
        self.taxonomies = taxonomies
        self.insertionCounts = insertionCounts
    }

    /// `dummy_msa`: the depth-1 alignment a chain gets when it has none of its own — a designed
    /// binder, always. Unannotated, so it can never be paired.
    public static func singleSequence(_ queryTokens: [Int32]) -> MSAAlignment {
        MSAAlignment(rows: [queryTokens],
                     taxonomies: [-1],
                     insertionCounts: [[Int32](repeating: 0, count: queryTokens.count)])
    }

    /// Parse an a3m.
    ///
    /// - Parameters:
    ///   - taxonomy: accession -> taxon id. Upstream reads taxonomy ONLY from `>UniRef100_*` headers
    ///     and ONLY when this is supplied; the CLI supplies nothing, which is why locally generated
    ///     alignments never pair across chains.
    ///   - maximumSequences: stop after this many rows (the CLI default is 8192). The query is row 0
    ///     and therefore always kept.
    public static func a3m(
        _ text: String,
        maximumSequences: Int? = nil,
        taxonomy: [String: Int32]? = nil
    ) throws -> MSAAlignment {
        var rows: [[Int32]] = []
        var taxonomies: [Int32] = []
        var insertions: [[Int32]] = []
        var seen = Set<String>()
        // Carried across lines: the header applies to the sequence that follows it.
        var pendingTaxonomy: Int32 = -1

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }

            if line.hasPrefix(">") {
                pendingTaxonomy = -1
                if let taxonomy,
                   let header = line.split(separator: " ").first?.split(separator: "\t").first,
                   header.hasPrefix(">UniRef100") {
                    // `header.split("_")[1]` — the accession, which may itself contain underscores
                    // beyond the first, so only the first separator is significant.
                    let parts = header.dropFirst().split(separator: "_", maxSplits: 1,
                                                         omittingEmptySubsequences: false)
                    if parts.count == 2, let taxon = taxonomy[String(parts[1])] {
                        pendingTaxonomy = taxon
                    }
                }
                continue
            }

            // Dedup on the ungapped, uppercased sequence. Note this STRIPS gaps, so two rows with
            // identical residues in different columns collide, and it UPPERCASES, so insertion
            // residues take part in the key even though they are not columns.
            let key = line.replacingOccurrences(of: "-", with: "").uppercased()
            if seen.contains(key) { continue }
            seen.insert(key)

            var columns: [Int32] = []
            var rowInsertions: [Int32] = []
            var pending: Int32 = 0
            for character in line {
                if character != "-", character.isLowercase {
                    pending += 1          // an insertion: counted, not a column
                    continue
                }
                columns.append(token(for: character))
                rowInsertions.append(pending)
                pending = 0
            }
            // A trailing insertion run has no following column and is dropped — upstream flushes
            // `count` only when it reaches a column, so trailing lowercase never lands anywhere.

            if let expected = rows.first?.count, columns.count != expected {
                throw MSAParseError.rowLengthMismatch(row: rows.count, expected: expected,
                                                      found: columns.count)
            }
            rows.append(columns)
            insertions.append(rowInsertions)
            taxonomies.append(pendingTaxonomy)

            if let maximumSequences, rows.count >= maximumSequences { break }
        }

        guard !rows.isEmpty else { throw MSAParseError.empty }
        return MSAAlignment(rows: rows, taxonomies: taxonomies, insertionCounts: insertions)
    }

    /// One-letter code -> token id, via the same templates the res_type one-hot uses so the two can
    /// never disagree. `-` is the gap; anything unrecognised is UNK, matching upstream's
    /// `prot_letter_to_token` folding of B/J/O/U/X/Z.
    static func token(for character: Character) -> Int32 {
        if character == "-" { return gapToken }
        let upper = Character(character.uppercased())
        if let template = AAResidueTemplates.template(oneLetter: upper) {
            return Int32(template.restype)
        }
        return unknownToken
    }
}

/// Builds the paired row plan: for each row of the final MSA, which sequence each chain contributes
/// and whether that chain is genuinely paired in that row.
public enum MSAPairing {

    /// One row of the paired MSA.
    public struct Row: Sendable, Equatable {
        /// Per chain: the sequence index to take, or nil for a gap.
        public let sequenceIndex: [Int?]
        /// Per chain: whether this row is a genuine cross-chain pairing.
        ///
        /// THE DIRECTION MATTERS. For a target with homologs against a binder without, the homolog
        /// rows are unpaired. Marking them paired asserts co-evolution across the interface that
        /// min_ipSAE scores, which makes the gate read HIGH rather than failing loudly.
        public let isPaired: [Bool]
    }

    /// - Parameters:
    ///   - alignments: one per chain, in token order.
    ///   - maximumPairs: cap on taxonomy-paired rows (upstream 8192).
    ///   - maximumTotal: cap on rows before the final truncation (upstream 16384).
    ///   - maximumSequences: final row cap. Inference passes `const.max_msa_seqs` = 16384.
    public static func rows(
        alignments: [MSAAlignment],
        maximumPairs: Int = 8_192,
        maximumTotal: Int = 16_384,
        maximumSequences: Int = 16_384
    ) -> [Row] {
        let chainCount = alignments.count
        guard chainCount > 0 else { return [] }

        // ---- taxonomy -> [(chain, sequence)], in first-appearance order ----
        var taxonOrder: [Int32] = []
        var byTaxon: [Int32: [(chain: Int, sequence: Int)]] = [:]
        for (chain, alignment) in alignments.enumerated() {
            for (sequence, taxon) in alignment.taxonomies.enumerated() where taxon != -1 {
                if byTaxon[taxon] == nil { taxonOrder.append(taxon) }
                byTaxon[taxon, default: []].append((chain, sequence))
            }
        }
        // Drop taxa that cannot pair anything, then order by how many DISTINCT chains they span,
        // most first. Python's sort is stable over an insertion-ordered dict, so ties keep
        // first-appearance order — reproduced here by sorting on (−span, firstAppearance).
        let entries = taxonOrder.enumerated()
            .compactMap { index, taxon -> (span: Int, index: Int, pairs: [(chain: Int, sequence: Int)])? in
                guard let pairs = byTaxon[taxon], pairs.count > 1 else { return nil }
                return (Set(pairs.map(\.chain)).count, index, pairs)
            }
            .sorted { $0.span != $1.span ? $0.span > $1.span : $0.index < $1.index }

        // ---- sequences still available per chain ----
        // UPSTREAM BUG (reproduced): upstream intends to exclude sequences already used by taxonomy
        // pairing, but builds `visited` as {(taxon, (chain, seq))} while probing it with
        // `(chain, seq)` — so the membership test never hits and every non-query sequence stays
        // available. A chain's sequence can therefore appear in both a paired and an unpaired row.
        // Unreachable in practice for RayMol (no taxonomy database => no paired rows past 0), but
        // "fixing" it here would diverge from the reference on any annotated alignment.
        var nextAvailable = [Int](repeating: 1, count: chainCount)
        func take(_ chain: Int) -> Int? {
            guard nextAvailable[chain] < alignments[chain].depth else { return nil }
            defer { nextAvailable[chain] += 1 }
            return nextAvailable[chain]
        }
        func remaining(_ chain: Int) -> Int {
            max(0, alignments[chain].depth - nextAvailable[chain])
        }

        // ---- row 0: every chain's own query, the only row paired without taxonomy ----
        var rows = [Row(sequenceIndex: [Int?](repeating: 0, count: chainCount),
                        isPaired: [Bool](repeating: true, count: chainCount))]

        // ---- taxonomy-paired rows ----
        taxonomyRows: for entry in entries {
            // Group by chain, preserving order, so a chain with several sequences in one taxon
            // contributes a different one to each row.
            var occurrenceOrder: [Int] = []
            var occurrences: [Int: [Int]] = [:]
            for pair in entry.pairs {
                if occurrences[pair.chain] == nil { occurrenceOrder.append(pair.chain) }
                occurrences[pair.chain, default: []].append(pair.sequence)
            }
            let maximumOccurrences = occurrences.values.map(\.count).max() ?? 0

            for offset in 0 ..< maximumOccurrences {
                var sequenceIndex = [Int?](repeating: nil, count: chainCount)
                var isPaired = [Bool](repeating: false, count: chainCount)
                for chain in occurrenceOrder {
                    let sequences = occurrences[chain]!
                    sequenceIndex[chain] = sequences[offset % sequences.count]
                    isPaired[chain] = true
                }
                // Chains absent from this taxon are unpaired and consume their next spare sequence,
                // gapping when they have none left.
                for chain in 0 ..< chainCount where !isPaired[chain] {
                    sequenceIndex[chain] = take(chain)
                }
                rows.append(Row(sequenceIndex: sequenceIndex, isPaired: isPaired))
                if rows.count >= maximumPairs { break taxonomyRows }
            }
        }

        // ---- unpaired rows, until the deepest chain is exhausted ----
        let longestRemaining = (0 ..< chainCount).map(remaining).max() ?? 0
        let fillCount = max(0, min(maximumTotal - rows.count, longestRemaining))
        for _ in 0 ..< fillCount {
            var sequenceIndex = [Int?](repeating: nil, count: chainCount)
            for chain in 0 ..< chainCount { sequenceIndex[chain] = take(chain) }
            rows.append(Row(sequenceIndex: sequenceIndex,
                            isPaired: [Bool](repeating: false, count: chainCount)))
        }

        return Array(rows.prefix(maximumSequences))
    }
}
