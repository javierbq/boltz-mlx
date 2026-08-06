import XCTest
@testable import BoltzMLX

/// CROSS-CHAIN MSA PAIRING, pinned to `featurizerv2.py::construct_paired_msa`.
///
/// WHY THIS FILE EXISTS. `msa_paired` tells the model which rows are genuinely the same organism
/// across chains. For the design case — a target with a real alignment against a designed binder
/// with none — the correct answer is that the target's homolog rows are UNPAIRED. Marking them
/// paired would assert co-evolution across the very interface `min_ipSAE` scores, and the failure
/// mode is not a crash but a *better-looking* number. So the direction is asserted explicitly here
/// rather than left to the end-to-end parity test.
///
/// Row 0 is always every chain's own query, and is the only row upstream marks paired when no
/// taxonomy is available.
final class MSAPairingTests: XCTestCase {

    /// Alignment of `depth` rows over `length` columns; row contents are irrelevant to pairing, only
    /// the depth and the taxonomy annotations are.
    private func alignment(depth: Int, length: Int = 4, taxonomies: [Int32]? = nil) -> MSAAlignment {
        MSAAlignment(
            rows: (0 ..< depth).map { r in (0 ..< length).map { _ in Int32(2 + r % 20) } },
            taxonomies: taxonomies ?? [Int32](repeating: -1, count: depth),
            insertionCounts: (0 ..< depth).map { _ in [Int32](repeating: 0, count: length) })
    }

    // MARK: Monomer

    func testMonomerYieldsOneRowPerSequenceWithOnlyTheQueryPaired() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 5)])
        XCTAssertEqual(rows.count, 5)
        XCTAssertEqual(rows[0].sequenceIndex, [0])
        XCTAssertEqual(rows[0].isPaired, [true])
        XCTAssertEqual(rows.dropFirst().map(\.sequenceIndex), [[1], [2], [3], [4]])
        XCTAssertTrue(rows.dropFirst().allSatisfy { $0.isPaired == [false] })
    }

    // MARK: The design case — target with MSA, binder without

    /// THE CASE THAT MATTERS. Chain 0 is a designed binder (no alignment, depth 1); chain 1 is the
    /// target with 4 homologs. Every homolog row must be unpaired for BOTH chains, and the binder
    /// must be a GAP there rather than repeating its own sequence.
    func testTargetHomologRowsAreUnpairedAndTheBinderIsAGap() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 1), alignment(depth: 5)])
        XCTAssertEqual(rows.count, 5, "1 query row + 4 target homolog rows")

        XCTAssertEqual(rows[0].sequenceIndex, [0, 0])
        XCTAssertEqual(rows[0].isPaired, [true, true])

        for (i, row) in rows.enumerated().dropFirst() {
            XCTAssertEqual(row.sequenceIndex, [nil, i],
                           "row \(i): binder must be a gap, target must advance")
            XCTAssertEqual(row.isPaired, [false, false],
                           "row \(i): an unannotated homolog is NOT paired across the interface")
        }
    }

    /// Repeating the binder's own sequence down the homolog rows instead of gapping it would tell the
    /// model the binder is perfectly conserved across every one of the target's homologs. Asserted
    /// separately because it is the specific wrong implementation that looks reasonable.
    func testBinderIsNeverRepeatedDownTheAlignment() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 1), alignment(depth: 20)])
        XCTAssertEqual(rows.dropFirst().filter { $0.sequenceIndex[0] != nil }.count, 0,
                       "the binder appears in exactly one row: its own query")
    }

    // MARK: Unequal depths

    func testTheShallowerChainGapsOnceItRunsOut() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 3), alignment(depth: 5)])
        XCTAssertEqual(rows.count, 5, "row count follows the DEEPEST chain")
        XCTAssertEqual(rows.map(\.sequenceIndex), [[0, 0], [1, 1], [2, 2], [nil, 3], [nil, 4]])
        XCTAssertTrue(rows.dropFirst().allSatisfy { $0.isPaired == [false, false] })
    }

    // MARK: Taxonomy pairing

    /// When taxonomy IS available and two chains share a taxon, that row is paired for both. This
    /// path is unreachable from a local a3m (no taxonomy database, so every row is -1), but it is the
    /// only thing that ever sets `is_paired` on a row past 0, so it is pinned.
    func testASharedTaxonAppearsAsAPairedRow() {
        let a = alignment(depth: 3, taxonomies: [-1, 42, -1])
        let b = alignment(depth: 3, taxonomies: [-1, -1, 42])
        let rows = MSAPairing.rows(alignments: [a, b])

        guard let paired = rows.dropFirst().first(where: { $0.isPaired == [true, true] }) else {
            return XCTFail("expected a paired row for the shared taxon 42; got \(rows)")
        }
        XCTAssertEqual(paired.sequenceIndex, [1, 2], "the row must take each chain's taxon-42 sequence")
    }

    /// A taxon present in only ONE chain cannot pair anything; upstream drops such taxa before
    /// building rows.
    func testATaxonInOnlyOneChainDoesNotCreateAPairedRow() {
        let a = alignment(depth: 3, taxonomies: [-1, 42, -1])
        let b = alignment(depth: 3, taxonomies: [-1, -1, -1])
        let rows = MSAPairing.rows(alignments: [a, b])
        XCTAssertTrue(rows.dropFirst().allSatisfy { $0.isPaired == [false, false] },
                      "a single-chain taxon must not produce a paired row")
    }

    // MARK: Caps

    func testTotalRowsAreCappedAtMaximumSequences() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 50)], maximumSequences: 8)
        XCTAssertEqual(rows.count, 8)
        XCTAssertEqual(rows[0].sequenceIndex, [0], "truncation keeps the query")
    }

    func testASingleSequenceChainAloneYieldsExactlyOneRow() {
        let rows = MSAPairing.rows(alignments: [alignment(depth: 1)])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].isPaired, [true])
    }
}
