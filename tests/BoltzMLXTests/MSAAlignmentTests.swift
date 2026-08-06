import XCTest
@testable import BoltzMLX

/// A3M PARSING, pinned to `src/boltz/data/parse/a3m.py::_parse_a3m`.
///
/// These are pure Swift over a string — no MLX, no weights, no fixtures — so they run in
/// milliseconds and catch the conventions that are easy to get plausibly wrong:
///
///   * lowercase columns are INSERTIONS and are removed from the alignment, not aligned columns;
///   * `-` is an aligned column carrying the gap token (id 1), not a skip;
///   * de-duplication is on the ungapped UPPERCASED sequence, so the insertion residues take part
///     in the dedup key even though they are not columns;
///   * taxonomy is read ONLY from `>UniRef100_*` headers AND only when a taxonomy database is
///     supplied, which is why a locally-generated a3m produces no paired rows at all.
///
/// Every expected token id below is `AAResidueTemplates.template(oneLetter:)!.restype`, i.e. the
/// same vocabulary the res_type one-hot uses: MET 14, VAL 21, THR 18, ALA 2, GLY 9, UNK 22, gap 1.
final class MSAAlignmentTests: XCTestCase {

    private func tok(_ s: String) -> [Int32] {
        s.map { $0 == "-" ? MSAAlignment.gapToken
                          : Int32(AAResidueTemplates.template(oneLetter: $0)!.restype) }
    }

    // MARK: Columns, gaps, insertions

    func testParsesQueryAndGapColumns() throws {
        let a = try MSAAlignment.a3m(">query\nMVT\n>h1\nM-T\n")
        XCTAssertEqual(a.depth, 2)
        XCTAssertEqual(a.queryLength, 3)
        XCTAssertEqual(a.rows[0], tok("MVT"))
        XCTAssertEqual(a.rows[1], [14, MSAAlignment.gapToken, 18])
    }

    /// The load-bearing a3m rule. A lowercase residue is an insertion relative to the query: it is
    /// NOT a column. Treating it as a column would make the row longer than the query and silently
    /// shift every subsequent residue.
    func testLowercaseColumnsAreInsertionsAndDoNotBecomeColumns() throws {
        let a = try MSAAlignment.a3m(">query\nMVT\n>h1\nMaaVT\n")
        XCTAssertEqual(a.queryLength, 3, "insertions must not widen the alignment")
        XCTAssertEqual(a.rows[1], tok("MVT"))
        // Two insertions sit before column 1.
        XCTAssertEqual(a.insertionCounts[1], [0, 2, 0])
        XCTAssertEqual(a.insertionCounts[0], [0, 0, 0], "the query has no insertions by construction")
    }

    /// Dedup key is `line.replace("-", "").upper()`. An exact repeat of the query is dropped.
    func testDeduplicatesAgainstTheQuery() throws {
        let a = try MSAAlignment.a3m(">q\nMVT\n>h1\nMVT\n>h2\nMVA\n")
        XCTAssertEqual(a.depth, 2, "an exact duplicate of the query must be dropped")
        XCTAssertEqual(a.rows[1], tok("MVA"))
    }

    /// The key STRIPS gaps, so two rows with the same residues in different columns collide even
    /// though they are different alignments. Comparing the aligned columns instead would keep both.
    func testDedupKeyIgnoresGapPlacement() throws {
        let a = try MSAAlignment.a3m(">q\nMVT\n>h1\nM-T\n>h2\n-MT\n")
        XCTAssertEqual(a.depth, 2, "\"M-T\" and \"-MT\" both strip to \"MT\" and must collide")
        XCTAssertEqual(a.rows[1], [14, MSAAlignment.gapToken, 18], "the FIRST spelling is the one kept")
    }

    func testInsertionResiduesParticipateInTheDedupKey() throws {
        // "MaVT" strips to "MAVT" which is NOT "MVT", so it survives dedup even though its
        // alignment columns are identical to the query's.
        let a = try MSAAlignment.a3m(">q\nMVT\n>h1\nMaVT\n")
        XCTAssertEqual(a.depth, 2)
        XCTAssertEqual(a.rows[1], tok("MVT"), "columns match the query")
        XCTAssertEqual(a.insertionCounts[1], [0, 1, 0])
    }

    // MARK: Validation

    func testRejectsARowWhoseColumnCountDiffersFromTheQuery() {
        XCTAssertThrowsError(try MSAAlignment.a3m(">q\nMVT\n>h1\nMV\n")) { error in
            guard case MSAParseError.rowLengthMismatch(let row, let expected, let found) = error else {
                return XCTFail("wrong error: \(error)")
            }
            XCTAssertEqual([row, expected, found], [1, 3, 2])
        }
    }

    func testRejectsAnEmptyAlignment() {
        XCTAssertThrowsError(try MSAAlignment.a3m("# just a comment\n")) { error in
            XCTAssertEqual(error as? MSAParseError, .empty)
        }
    }

    /// Unknown one-letter codes map to UNK (22) rather than throwing — upstream's
    /// `prot_letter_to_token` folds B/J/O/U/X/Z to UNK, and an MSA row is not worth failing a
    /// prediction over.
    func testUnknownLettersBecomeUNK() throws {
        let a = try MSAAlignment.a3m(">q\nMVT\n>h1\nMXT\n")
        XCTAssertEqual(a.rows[1], [14, 22, 18])
    }

    // MARK: Truncation and taxonomy

    func testMaximumSequencesTruncatesKeepingTheQueryFirst() throws {
        let text = ">q\nMVT\n>h1\nMVA\n>h2\nMVG\n>h3\nMVC\n"
        let a = try MSAAlignment.a3m(text, maximumSequences: 2)
        XCTAssertEqual(a.depth, 2)
        XCTAssertEqual(a.rows[0], tok("MVT"), "the query must always be row 0")
        XCTAssertEqual(a.rows[1], tok("MVA"))
    }

    /// Without a taxonomy database every row is unannotated, which is the RayMol case: a
    /// locally-generated or user-supplied a3m yields NO cross-chain pairing.
    func testTaxonomyIsAbsentWhenNoDatabaseIsSupplied() throws {
        let a = try MSAAlignment.a3m(">UniRef100_A0A0D4WTP2\tstuff\nMVT\n>UniRef100_XYZ\nMVA\n")
        XCTAssertEqual(a.taxonomies, [-1, -1])
    }

    func testTaxonomyIsReadOnlyFromUniRef100HeadersWhenADatabaseIsSupplied() throws {
        let a = try MSAAlignment.a3m(
            ">UniRef100_A0A0D4WTP2\nMVT\n>UniRef100_MISSING\nMVA\n>somethingelse\nMVG\n",
            taxonomy: ["A0A0D4WTP2": 9606])
        XCTAssertEqual(a.taxonomies, [9606, -1, -1],
                       "unknown accession and non-UniRef100 headers must stay unannotated")
    }

    // MARK: The no-MSA chain

    /// `dummy_msa`: a chain with no alignment gets a depth-1 MSA that is exactly the query, with no
    /// taxonomy so it can never be paired. This is what a designed binder chain always gets.
    func testSingleSequenceAlignmentIsTheQueryAndIsUnannotated() {
        let a = MSAAlignment.singleSequence([14, 21, 18])
        XCTAssertEqual(a.depth, 1)
        XCTAssertEqual(a.rows, [[14, 21, 18]])
        XCTAssertEqual(a.taxonomies, [-1])
        XCTAssertEqual(a.insertionCounts, [[0, 0, 0]])
    }
}
