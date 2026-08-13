import Foundation

/// Serializes a predicted structure to PDB.
///
/// Promoted from `MSAEndToEndTests.writePDB`, which already implemented the correct
/// identity walk; this is that walk made public, with the gaps a viewer-grade writer
/// has to close (TER records, insertion codes, 1-based renumbering, a populated
/// element column).
///
/// `BoltzStructure` carries no atom identity, but identity is strictly *recoverable*:
///
///     atom axis = concat, over canonical.orderedResidues,
///                 of AAResidueTemplates.template(threeLetter:)!.atoms in template order
///
/// and `orderedResidues` is documented as the ONLY ordering any downstream index
/// should derive from. Coordinates come back in exactly that order, already unpadded.
///
/// Deliberate limitations, all inherited from what the checkpoint was trained on:
/// heavy atoms only (no hydrogens); no `OXT`, ever, including at chain termini; and a
/// single model, because `diffusion_samples` is not plumbed and only diffusion sample 0
/// escapes `BoltzPredictor`.
///
/// The B-factor column carries **pLDDT (0...100) when it is supplied**, which is what a
/// viewer colours by, and `0.00` when it is not. Pass the real thing or nothing: a
/// per-token PAE row-reduction is a DIFFERENT quantity and must never be written here as
/// though it were pLDDT. `ConfidenceModule` computes pLDDT only on the scored path, so a
/// caller using plain `predict` has none and correctly gets zeros.
///
/// pLDDT is per TOKEN, i.e. per residue for a protein chain, so every atom of a residue
/// gets that residue's value — the same convention AlphaFold and Boltz write.
public enum StructureWriter {

  public enum WriteError: Error, LocalizedError, Equatable {
    case atomCountMismatch(expected: Int, found: Int)
    case missingTemplate(String)
    case plddtCountMismatch(expected: Int, found: Int)

    public var errorDescription: String? {
      switch self {
      case let .atomCountMismatch(expected, found):
        return "structure has \(found) coordinates, expected \(expected)"
      case let .missingTemplate(code):
        return "no residue template for \(code)"
      case let .plddtCountMismatch(expected, found):
        return "got \(found) pLDDT values, expected \(expected) (one per residue)"
      }
    }
  }

  /// Render `structure` as a PDB file, taking identity from `canonical`.
  ///
  /// - Throws: ``WriteError/atomCountMismatch`` when the coordinate count does not
  ///   match the template walk — which means the two arguments do not describe the
  ///   same prediction, and is always a caller bug rather than something to paper over.
  public static func pdb(structure: BoltzStructure,
                        canonical: CanonicalStructure,
                        plddt: [Double]? = nil) throws -> String {
    // One value per residue, or none. A mismatch means the scores and the structure do
    // not describe the same prediction, which is a caller bug: writing them anyway would
    // silently attach one residue's confidence to another.
    if let plddt, plddt.count != canonical.orderedResidues.count {
      throw WriteError.plddtCountMismatch(expected: canonical.orderedResidues.count,
                                          found: plddt.count)
    }

    var expected = 0
    for residue in canonical.orderedResidues {
      guard let template = AAResidueTemplates.template(threeLetter: residue.threeLetter)
      else { throw WriteError.missingTemplate(residue.threeLetter) }
      expected += template.atoms.count
    }
    guard structure.coordinates.count == expected else {
      throw WriteError.atomCountMismatch(expected: expected,
                                         found: structure.coordinates.count)
    }

    var text = ""
    var serial = 1
    var atom = 0
    var previousChain: String?
    // Renumber 1-based per chain: hostResSeq is 0-based for sequence-derived
    // structures, and emitting it verbatim offsets every model by one residue
    // against a crystal, which silently defeats residue-wise comparison.
    var resSeq = 0

    for (residueIndex, residue) in canonical.orderedResidues.enumerated() {
      let template = AAResidueTemplates.template(threeLetter: residue.threeLetter)!
      let confidence = plddt?[residueIndex] ?? 0.0
      if let previous = previousChain, previous != residue.hostChain {
        text += "TER\n"
        resSeq = 0
      }
      previousChain = residue.hostChain
      resSeq += 1

      for spec in template.atoms {
        let position = structure.coordinates[atom]
        atom += 1
        text += record(serial: serial,
                       name: spec.name,
                       element: element(for: spec.atomicNumber),
                       residue: residue.threeLetter,
                       chain: chainCharacter(residue.hostChain),
                       resSeq: resSeq,
                       insCode: residue.hostInsCode.map(String.init) ?? " ",
                       x: position.x, y: position.y, z: position.z,
                       bFactor: confidence)
        serial += 1
      }
    }
    if previousChain != nil { text += "TER\n" }
    return text + "END\n"
  }

  /// One fixed-column ATOM record, columns 1-78.
  ///
  /// Layout, 1-indexed: 1-6 `ATOM  `, 7-11 serial, 13-16 atom name, 17 altLoc,
  /// 18-20 resName, 22 chain, 23-26 resSeq, 27 insCode, 31-38/39-46/47-54 x/y/z,
  /// 55-60 occupancy, 61-66 B-factor, 77-78 element. The atom name is padded by one
  /// leading space, which is the convention for single-character elements — the only
  /// kind that occurs across the canonical-20 heavy-atom templates.
  private static func record(serial: Int, name: String, element: String,
                             residue: String, chain: Character, resSeq: Int,
                             insCode: String, x: Float, y: Float, z: Float,
                             bFactor: Double) -> String {
    let field = String((" " + name + "   ").prefix(4))
    // Right-justify the element by hand. `%2@` does NOT pad: Foundation's
    // String(format:) honours a width for %s but not for %@, so `%2@` with "N"
    // emits "N" and the record comes out 77 columns instead of 78, putting the
    // element in column 77 where a strict reader expects 78.
    let elementField = String(repeating: " ", count: max(0, 2 - element.count)) + element
    return String(
      format: "ATOM  %5d %@ %@ %@%4d%@   %8.3f%8.3f%8.3f%6.2f%6.2f          %@\n",
      serial, field, residue, String(chain), resSeq, insCode,
      x, y, z, 1.00, bFactor, elementField)
  }

  /// Only C, N, O and S occur across the canonical-20 heavy-atom templates. Anything
  /// else would be a template change, and `X` makes that visible rather than guessing.
  private static func element(for atomicNumber: Int) -> String {
    switch atomicNumber {
    case 6: return "C"
    case 7: return "N"
    case 8: return "O"
    case 16: return "S"
    default: return "X"
    }
  }

  /// PDB chain is a single column. Callers constrain chain ids to one uppercase
  /// character on the way in; anything longer would shift every later column.
  private static func chainCharacter(_ chain: String) -> Character {
    chain.first ?? "A"
  }
}
