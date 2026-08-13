import XCTest
@testable import BoltzMLX

/// `BoltzStructure` carries only coordinates, but identity is strictly recoverable:
/// the atom axis is the concatenation, over `canonical.orderedResidues`, of
/// `AAResidueTemplates.template(threeLetter:)!.atoms` in template order. These tests
/// pin that contract and the column layout, because a writer that silently
/// mis-numbers or mis-orders atoms produces a file that looks fine and compares wrong.
final class StructureWriterTests: XCTestCase {

  private func atomCount(_ canonical: CanonicalStructure) -> Int {
    canonical.orderedResidues.reduce(0) { total, residue in
      total + (AAResidueTemplates.template(threeLetter: residue.threeLetter)?.atoms.count ?? 0)
    }
  }

  private func structure(for canonical: CanonicalStructure) -> BoltzStructure {
    let count = atomCount(canonical)
    return BoltzStructure(
      coordinates: (0..<count).map { SIMD3<Float>(Float($0), Float($0) * 2, Float($0) * 3) },
      atomMask: Array(repeating: true, count: count))
  }

  /// Two residues, one chain: ALA then GLY.
  private func fixture() throws -> (BoltzStructure, CanonicalStructure) {
    let canonical = try CanonicalStructure.fromSequences([("A", "AG")])
    return (structure(for: canonical), canonical)
  }

  private func atomLines(_ text: String) -> [String] {
    text.split(separator: "\n").map(String.init).filter { $0.hasPrefix("ATOM  ") }
  }

  func testEmitsOneAtomRecordPerTemplateAtom() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    XCTAssertEqual(atomLines(text).count, structure.coordinates.count)
  }

  /// hostResSeq is 0-based for sequence-derived structures. Emitting it verbatim
  /// offsets every model by one residue against a crystal, which silently defeats
  /// residue-wise comparison in a viewer.
  func testResidueNumbersAreOneBasedPerChain() throws {
    let canonical = try CanonicalStructure.fromSequences([("A", "AG"), ("B", "G")])
    let text = try StructureWriter.pdb(structure: structure(for: canonical),
                                      canonical: canonical)
    func resSeq(_ line: String) -> String {
      let s = line.index(line.startIndex, offsetBy: 22)
      let e = line.index(line.startIndex, offsetBy: 26)
      return String(line[s..<e]).trimmingCharacters(in: .whitespaces)
    }
    let lines = atomLines(text)
    XCTAssertEqual(resSeq(lines[0]), "1", "first residue of chain A must be 1")
    // Chain B's first atom must restart at 1, not continue from chain A.
    let chainB = lines.filter { line -> Bool in
      let i = line.index(line.startIndex, offsetBy: 21)
      return line[i] == "B"
    }
    XCTAssertFalse(chainB.isEmpty)
    XCTAssertEqual(resSeq(chainB[0]), "1", "numbering must restart per chain")
  }

  func testAtomAndResidueNamesComeFromTheTemplate() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    let first = atomLines(text)[0]
    XCTAssertTrue(first.contains(" ALA "), "first residue is ALA: \(first)")
    let nameStart = first.index(first.startIndex, offsetBy: 12)
    let nameEnd = first.index(first.startIndex, offsetBy: 16)
    XCTAssertEqual(String(first[nameStart..<nameEnd]).trimmingCharacters(in: .whitespaces),
                   "N", "template order starts at N")
  }

  func testCoordinatesAreWrittenInTemplateOrder() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    func xColumn(_ line: String) -> String {
      let s = line.index(line.startIndex, offsetBy: 30)
      let e = line.index(line.startIndex, offsetBy: 38)
      return String(line[s..<e]).trimmingCharacters(in: .whitespaces)
    }
    let lines = atomLines(text)
    XCTAssertEqual(xColumn(lines[0]), "0.000")
    XCTAssertEqual(xColumn(lines[1]), "1.000")
    XCTAssertEqual(xColumn(lines[2]), "2.000")
  }

  func testColumnsAreFixedWidth() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    for line in atomLines(text) {
      XCTAssertGreaterThanOrEqual(line.count, 78, "short ATOM record: \(line)")
    }
  }

  func testEmitsTERAndEND() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    XCTAssertTrue(text.contains("\nTER"), "TER missing; multi-chain reads as fused")
    XCTAssertTrue(text.hasSuffix("END\n"))
  }

  func testOneTERPerChain() throws {
    let canonical = try CanonicalStructure.fromSequences([("A", "AG"), ("B", "G")])
    let text = try StructureWriter.pdb(structure: structure(for: canonical),
                                      canonical: canonical)
    XCTAssertEqual(text.components(separatedBy: "\nTER").count - 1, 2)
  }

  func testElementColumnIsPopulated() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    for line in atomLines(text) {
      let s = line.index(line.startIndex, offsetBy: 76)
      let element = String(line[s...]).trimmingCharacters(in: .whitespaces)
      XCTAssertTrue(["C", "N", "O", "S"].contains(element),
                    "unexpected element \(element) in: \(line)")
    }
  }

  private func bFactor(_ line: String) -> String {
    let s = line.index(line.startIndex, offsetBy: 60)
    let e = line.index(line.startIndex, offsetBy: 66)
    return String(line[s..<e]).trimmingCharacters(in: .whitespaces)
  }

  /// pLDDT is per TOKEN, so every atom of a residue carries that residue's value.
  func testPLDDTIsWrittenIntoTheBFactorPerResidue() throws {
    let (structure, canonical) = try fixture()          // ALA (5 atoms) then GLY (4)
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical,
                                      plddt: [88.5, 42.25])
    let lines = atomLines(text)
    XCTAssertEqual(lines.count, 9)
    for line in lines.prefix(5) { XCTAssertEqual(bFactor(line), "88.50") }
    for line in lines.suffix(4) { XCTAssertEqual(bFactor(line), "42.25") }
  }

  /// A count mismatch means the scores and the structure are not the same prediction.
  /// Writing anyway would attach one residue's confidence to another.
  func testWrongPLDDTCountThrowsRatherThanMisattributing() throws {
    let (structure, canonical) = try fixture()
    XCTAssertThrowsError(try StructureWriter.pdb(structure: structure, canonical: canonical,
                                                plddt: [50.0]))
  }

  func testPLDDTDoesNotDisturbTheColumnLayout() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical,
                                      plddt: [100.0, 0.0])
    for line in atomLines(text) { XCTAssertGreaterThanOrEqual(line.count, 78) }
  }

  /// Omitting it must stay a documented constant, NOT a fabricated confidence -- a caller
  /// on the plain `predict` path has no pLDDT, and inventing one would be worse than zero.
  func testBFactorIsZero() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    let line = atomLines(text)[0]
    let s = line.index(line.startIndex, offsetBy: 60)
    let e = line.index(line.startIndex, offsetBy: 66)
    XCTAssertEqual(String(line[s..<e]).trimmingCharacters(in: .whitespaces), "0.00")
  }

  func testSerialNumbersAreSequentialFromOne() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    for (index, line) in atomLines(text).enumerated() {
      let s = line.index(line.startIndex, offsetBy: 6)
      let e = line.index(line.startIndex, offsetBy: 11)
      XCTAssertEqual(String(line[s..<e]).trimmingCharacters(in: .whitespaces),
                     String(index + 1))
    }
  }

  func testCoordinateCountMismatchThrows() throws {
    let (_, canonical) = try fixture()
    let short = BoltzStructure(coordinates: [SIMD3<Float>(0, 0, 0)], atomMask: [true])
    XCTAssertThrowsError(try StructureWriter.pdb(structure: short, canonical: canonical))
  }

  /// No OXT, ever -- the featurizer drops the trailing OXT on every residue including
  /// chain termini, and that is what the checkpoint was trained on.
  func testNoOXTIsEmitted() throws {
    let (structure, canonical) = try fixture()
    let text = try StructureWriter.pdb(structure: structure, canonical: canonical)
    XCTAssertFalse(text.contains(" OXT"))
  }
}
