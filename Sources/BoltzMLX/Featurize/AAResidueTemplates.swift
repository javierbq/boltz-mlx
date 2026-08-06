// AAResidueTemplates.swift — GENERATED. Do not edit by hand.
//
// Canonical-20 amino-acid templates for Boltz-2 featurization: ordered heavy atoms with atomic
// number, formal charge, backbone-feature index, and one reference conformer.
//
// PROVENANCE. Atom order, elements and formal charges were cross-validated two independent ways and
// agree exactly: (a) decoded out of a Python-generated reference feature bundle
// (ref_atom_name_chars / ref_element / ref_charge / atom_backbone_feat / ref_space_uid), and
// (b) read from the 20 canonical component pickles under ~/.boltz/mols via RDKit. Conformer
// coordinates are conformer 0 from those pickles.
//
// OXT IS EXCLUDED. The pickles carry a trailing OXT on every residue; the reference featurizer drops
// it, including at chain termini (verified: every instance of every residue type in a 228-token
// two-chain reference has the counts below). 187 pickle atoms - 20 OXT = 167, matching the bundle.
//
// This file is deliberately MLX-free so the canonicalization layer can be shared and unit-tested
// without a GPU. Regenerate with scripts/gen_aa_templates.py.
import Foundation

/// One heavy atom of a canonical residue template.
public struct AAAtomTemplate: Sendable {
    public let name: String
    public let atomicNumber: Int
    public let formalCharge: Float
    /// Index into the 17-wide atom_backbone_feat one-hot (0=N, 1=CA, 2=C, 3=O, 4=side chain).
    public let backboneFeat: Int
    /// Reference conformer position. See BoltzFeaturizer for why an arbitrary but FIXED conformer is
    /// the correct choice: upstream re-randomizes ref_pos on every run, so no realization is canonical.
    public let refPos: SIMD3<Float>

    public init(_ name: String, _ atomicNumber: Int, _ formalCharge: Float, _ backboneFeat: Int,
                _ x: Float, _ y: Float, _ z: Float) {
        self.name = name
        self.atomicNumber = atomicNumber
        self.formalCharge = formalCharge
        self.backboneFeat = backboneFeat
        self.refPos = SIMD3<Float>(x, y, z)
    }
}

/// One canonical amino acid: its Boltz restype index and ordered heavy atoms.
public struct AAResidueTemplate: Sendable {
    public let oneLetter: Character
    /// Boltz token vocabulary index. Derived from the reference one-hot: ALA=2 ... VAL=21,
    /// i.e. alphabetical by three-letter code offset by 2.
    public let restype: Int
    public let atoms: [AAAtomTemplate]

    public init(oneLetter: Character, restype: Int, atoms: [AAAtomTemplate]) {
        self.oneLetter = oneLetter
        self.restype = restype
        self.atoms = atoms
    }

    /// Index within this residue of the atom the confidence head treats as REPRESENTATIVE of the
    /// token. Decoded from a reference bundle: CB for every canonical residue except glycine, which
    /// has no CB and uses CA. Assuming CA throughout — the obvious guess — would shift 19 of 20
    /// residue types by one atom and produce a plausible but wrong PAE.
    public var representativeAtomIndex: Int {
        atoms.firstIndex { $0.name == "CB" } ?? atoms.firstIndex { $0.name == "CA" } ?? 0
    }

    /// Index of the atom used as the token CENTRE — always CA, unlike the representative atom.
    public var centerAtomIndex: Int {
        atoms.firstIndex { $0.name == "CA" } ?? 0
    }
}

public enum AAResidueTemplates {
    /// Width of the res_type / profile / template_restype one-hot.
    public static let restypeCount = 33
    /// Atom axis pads up to a multiple of this.
    public static let atomPadMultiple = 32

    public static let byThreeLetter: [String: AAResidueTemplate] = [
        "ALA": .init(oneLetter: "A", restype: 2, atoms: [
            .init("N", 7, 0.0, 1, -0.9242, 1.1821, 0.7127), .init("CA", 6, 0.0, 2, -0.2664, -0.0883, 0.4009), .init("C", 6, 0.0, 3, 1.1189, 0.1387, -0.1437), .init("O", 8, 0.0, 4, 1.2882, 0.8058, -1.2001), .init("CB", 6, 0.0, 0, -1.1134, -0.8915, -0.5877)
        ]),
        "ARG": .init(oneLetter: "R", restype: 3, atoms: [
            .init("N", 7, 0.0, 1, 3.2533, -1.6996, -0.9853), .init("CA", 6, 0.0, 2, 2.1915, -0.692, -0.9506), .init("C", 6, 0.0, 3, 2.8002, 0.6723, -1.116), .init("O", 8, 0.0, 4, 2.4654, 1.3988, -2.0899), .init("CB", 6, 0.0, 0, 1.3811, -0.795, 0.3632), .init("CG", 6, 0.0, 0, 0.2034, 0.1929, 0.4573), .init("CD", 6, 0.0, 0, -0.9024, -0.0995, -0.5672), .init("NE", 7, 0.0, 0, -2.0868, 0.7147, -0.2912), .init("CZ", 6, 0.0, 0, -3.0182, 0.4804, 0.7831), .init("NH1", 7, 0.0, 0, -4.0801, 1.4155, 0.9939), .init("NH2", 7, 1.0, 0, -2.9226, -0.5637, 1.5534)
        ]),
        "ASN": .init(oneLetter: "N", restype: 4, atoms: [
            .init("N", 7, 0.0, 1, -1.5768, -1.7228, 0.2427), .init("CA", 6, 0.0, 2, -0.7042, -0.5622, 0.4277), .init("C", 6, 0.0, 3, -1.2467, 0.6073, -0.3484), .init("O", 8, 0.0, 4, -1.5549, 0.4764, -1.5639), .init("CB", 6, 0.0, 0, 0.7264, -0.9033, -0.0286), .init("CG", 6, 0.0, 0, 1.6872, 0.2081, 0.2841), .init("OD1", 8, 0.0, 0, 2.1969, 0.2892, 1.4335), .init("ND2", 7, 0.0, 0, 1.9967, 1.1967, -0.6982)
        ]),
        "ASP": .init(oneLetter: "D", restype: 5, atoms: [
            .init("N", 7, 0.0, 1, -0.1177, -1.631, 0.3744), .init("CA", 6, 0.0, 2, -0.3847, -0.1929, 0.2812), .init("C", 6, 0.0, 3, -1.8063, 0.0594, -0.1472), .init("O", 8, 0.0, 4, -2.2238, -0.3682, -1.2575), .init("CB", 6, 0.0, 0, 0.5982, 0.4785, -0.692), .init("CG", 6, 0.0, 0, 2.0088, 0.3602, -0.203), .init("OD1", 8, 0.0, 0, 2.7473, -0.5712, -0.6217), .init("OD2", 8, 0.0, 0, 2.4909, 1.2578, 0.7444)
        ]),
        "CYS": .init(oneLetter: "C", restype: 6, atoms: [
            .init("N", 7, 0.0, 1, -0.0587, 1.771, 0.2434), .init("CA", 6, 0.0, 2, -0.0655, 0.4766, -0.4457), .init("C", 6, 0.0, 3, -1.2739, -0.3504, -0.0836), .init("O", 8, 0.0, 4, -1.7325, -0.3431, 1.0909), .init("CB", 6, 0.0, 0, 1.2432, -0.2892, -0.1925), .init("SG", 16, 0.0, 0, 1.4842, -0.7191, 1.565)
        ]),
        "GLN": .init(oneLetter: "Q", restype: 7, atoms: [
            .init("N", 7, 0.0, 1, -1.8543, -1.0025, -1.6279), .init("CA", 6, 0.0, 2, -1.2922, -0.6787, -0.3154), .init("C", 6, 0.0, 3, -2.2264, 0.2517, 0.4068), .init("O", 8, 0.0, 4, -2.7255, -0.0883, 1.5129), .init("CB", 6, 0.0, 0, 0.1166, -0.0621, -0.4566), .init("CG", 6, 0.0, 0, 0.8141, 0.1046, 0.9002), .init("CD", 6, 0.0, 0, 2.1951, 0.6569, 0.7155), .init("OE1", 8, 0.0, 0, 2.387, 1.9004, 0.7825), .init("NE2", 7, 0.0, 0, 3.2874, -0.2131, 0.4196)
        ]),
        "GLU": .init(oneLetter: "E", restype: 8, atoms: [
            .init("N", 7, 0.0, 1, -1.3493, -1.1143, -1.3739), .init("CA", 6, 0.0, 2, -1.2766, -0.5063, -0.0429), .init("C", 6, 0.0, 3, -1.9196, 0.8547, -0.0421), .init("O", 8, 0.0, 4, -1.9091, 1.5702, -1.0805), .init("CB", 6, 0.0, 0, 0.1827, -0.4321, 0.4521), .init("CG", 6, 0.0, 0, 1.1058, 0.3659, -0.4818), .init("CD", 6, 0.0, 0, 2.509, 0.3514, 0.0365), .init("OE1", 8, 0.0, 0, 2.8964, 1.2446, 0.8365), .init("OE2", 8, 0.0, 0, 3.3728, -0.6754, -0.3314)
        ]),
        "GLY": .init(oneLetter: "G", restype: 9, atoms: [
            .init("N", 7, 0.0, 1, -1.2915, 0.6081, -0.4229), .init("CA", 6, 0.0, 2, -0.4896, -0.2882, 0.4019), .init("C", 6, 0.0, 3, 0.935, -0.2544, -0.0479), .init("O", 8, 0.0, 4, 1.3474, -1.0837, -0.9022)
        ]),
        "HIS": .init(oneLetter: "H", restype: 10, atoms: [
            .init("N", 7, 0.0, 1, 1.0371, -1.5621, 0.4179), .init("CA", 6, 0.0, 2, 1.198, -0.4024, -0.4631), .init("C", 6, 0.0, 3, 2.6561, -0.0712, -0.6408), .init("O", 8, 0.0, 4, 3.1796, -0.1237, -1.786), .init("CB", 6, 0.0, 0, 0.4405, 0.8117, 0.0985), .init("CG", 6, 0.0, 0, -1.0357, 0.552, 0.1525), .init("ND1", 7, 1.0, 0, -1.9089, 0.6491, -0.9682), .init("CD2", 6, 0.0, 0, -1.7096, 0.1138, 1.2062), .init("CE1", 6, 0.0, 0, -3.0752, 0.2813, -0.5512), .init("NE2", 7, 0.0, 0, -3.065, -0.0785, 0.8257)
        ]),
        "ILE": .init(oneLetter: "I", restype: 11, atoms: [
            .init("N", 7, 0.0, 1, -1.2379, -1.8147, -0.1644), .init("CA", 6, 0.0, 2, -1.278, -0.3994, 0.2244), .init("C", 6, 0.0, 3, -1.988, 0.4498, -0.8037), .init("O", 8, 0.0, 4, -2.0292, 0.1029, -2.0154), .init("CB", 6, 0.0, 0, 0.1423, 0.1467, 0.5415), .init("CG1", 6, 0.0, 0, 1.1062, 0.0452, -0.6694), .init("CG2", 6, 0.0, 0, 0.7215, -0.5686, 1.7751), .init("CD1", 6, 0.0, 0, 2.3718, 0.8853, -0.4905)
        ]),
        "LEU": .init(oneLetter: "L", restype: 12, atoms: [
            .init("N", 7, 0.0, 1, 1.6452, -1.0176, -1.0626), .init("CA", 6, 0.0, 2, 1.3129, 0.0807, -0.1505), .init("C", 6, 0.0, 3, 2.4312, 0.2831, 0.8357), .init("O", 8, 0.0, 4, 2.9231, -0.7013, 1.4512), .init("CB", 6, 0.0, 0, 0.0061, -0.2141, 0.6198), .init("CG", 6, 0.0, 0, -1.2534, -0.3773, -0.2664), .init("CD1", 6, 0.0, 0, -2.4557, -0.7604, 0.6071), .init("CD2", 6, 0.0, 0, -1.5742, 0.8986, -1.0599)
        ]),
        "LYS": .init(oneLetter: "K", restype: 13, atoms: [
            .init("N", 7, 0.0, 1, -2.3939, -1.4751, -0.9336), .init("CA", 6, 0.0, 2, -2.2095, -0.6204, 0.2421), .init("C", 6, 0.0, 3, -3.4513, 0.1961, 0.471), .init("O", 8, 0.0, 4, -3.9688, 0.2503, 1.6188), .init("CB", 6, 0.0, 0, -1.0087, 0.3269, 0.048), .init("CG", 6, 0.0, 0, 0.3365, -0.4156, 0.047), .init("CD", 6, 0.0, 0, 1.5089, 0.5716, -0.0063), .init("CE", 6, 0.0, 0, 2.8534, -0.1679, -0.0209), .init("NZ", 7, 1.0, 0, 3.969, 0.7766, -0.0701)
        ]),
        "MET": .init(oneLetter: "M", restype: 14, atoms: [
            .init("N", 7, 0.0, 1, -1.7435, -0.4404, 1.8518), .init("CA", 6, 0.0, 2, -1.2169, 0.4042, 0.778), .init("C", 6, 0.0, 3, -2.3118, 0.6834, -0.2135), .init("O", 8, 0.0, 4, -2.666, 1.8706, -0.4448), .init("CB", 6, 0.0, 0, -0.0005, -0.2626, 0.0993), .init("CG", 6, 0.0, 0, 0.6952, 0.6768, -0.893), .init("SD", 16, 0.0, 0, 2.1368, -0.1491, -1.6563), .init("CE", 6, 0.0, 0, 3.3404, 0.3481, -0.381)
        ]),
        "PHE": .init(oneLetter: "F", restype: 15, atoms: [
            .init("N", 7, 0.0, 1, 2.7976, -1.621, -0.4779), .init("CA", 6, 0.0, 2, 1.58, -0.829, -0.6608), .init("C", 6, 0.0, 3, 1.9536, 0.5758, -1.0446), .init("O", 8, 0.0, 4, 1.5211, 1.0726, -2.119), .init("CB", 6, 0.0, 0, 0.7137, -0.8474, 0.6163), .init("CG", 6, 0.0, 0, -0.6229, -0.1867, 0.3913), .init("CD1", 6, 0.0, 0, -0.8259, 1.0978, 0.7382), .init("CD2", 6, 0.0, 0, -1.716, -0.9426, -0.2701), .init("CE1", 6, 0.0, 0, -2.1313, 1.7412, 0.48), .init("CE2", 6, 0.0, 0, -2.896, -0.3552, -0.5053), .init("CZ", 6, 0.0, 0, -3.1146, 1.0512, -0.1101)
        ]),
        "PRO": .init(oneLetter: "P", restype: 16, atoms: [
            .init("N", 7, 0.0, 1, -0.5773, -0.5124, -1.1504), .init("CA", 6, 0.0, 2, 0.512, 0.2357, -0.5195), .init("C", 6, 0.0, 3, 1.8393, -0.4671, -0.655), .init("O", 8, 0.0, 4, 2.0837, -1.1663, -1.6751), .init("CB", 6, 0.0, 0, 0.1029, 0.4535, 0.9297), .init("CG", 6, 0.0, 0, -1.4098, 0.5215, 0.8787), .init("CD", 6, 0.0, 0, -1.7971, -0.0604, -0.4788)
        ]),
        "SER": .init(oneLetter: "S", restype: 17, atoms: [
            .init("N", 7, 0.0, 1, 0.896, -1.4233, -0.2693), .init("CA", 6, 0.0, 2, -0.0188, -0.2875, -0.3864), .init("C", 6, 0.0, 3, 0.7676, 0.993, -0.4215), .init("O", 8, 0.0, 4, 0.5327, 1.8535, -1.3116), .init("CB", 6, 0.0, 0, -0.9974, -0.277, 0.7988), .init("OG", 8, 0.0, 0, -1.9, 0.7916, 0.6915)
        ]),
        "THR": .init(oneLetter: "T", restype: 18, atoms: [
            .init("N", 7, 0.0, 1, 0.005, 1.7656, 0.0639), .init("CA", 6, 0.0, 2, 0.4232, 0.3755, 0.2735), .init("C", 6, 0.0, 3, 1.7185, 0.1101, -0.4465), .init("O", 8, 0.0, 4, 1.9253, 0.5967, -1.5913), .init("CB", 6, 0.0, 0, -0.6634, -0.6156, -0.2117), .init("OG1", 8, 0.0, 0, -1.0763, -0.298, -1.5172), .init("CG2", 6, 0.0, 0, -1.8869, -0.6176, 0.7084)
        ]),
        "TRP": .init(oneLetter: "W", restype: 19, atoms: [
            .init("N", 7, 0.0, 1, -3.044, 1.0628, 0.0157), .init("CA", 6, 0.0, 2, -2.4637, 0.0247, -0.8414), .init("C", 6, 0.0, 3, -2.1867, -1.2229, -0.0469), .init("O", 8, 0.0, 4, -2.3365, -2.3535, -0.5831), .init("CB", 6, 0.0, 0, -1.2063, 0.5196, -1.5924), .init("CG", 6, 0.0, 0, -0.0645, 0.8841, -0.6835), .init("CD1", 6, 0.0, 0, 0.1611, 2.0852, -0.148), .init("CD2", 6, 0.0, 0, 0.9782, -0.0055, -0.1773), .init("NE1", 7, 0.0, 0, 1.3107, 2.0507, 0.6981), .init("CE2", 6, 0.0, 0, 1.7529, 0.7044, 0.6196), .init("CE3", 6, 0.0, 0, 1.2361, -1.4356, -0.4236), .init("CZ2", 6, 0.0, 0, 2.907, 0.102, 1.307), .init("CZ3", 6, 0.0, 0, 2.2852, -2.0072, 0.195), .init("CH2", 6, 0.0, 0, 3.1586, -1.2035, 1.1003)
        ]),
        "TYR": .init(oneLetter: "Y", restype: 20, atoms: [
            .init("N", 7, 0.0, 1, -1.795, 0.4912, -1.3951), .init("CA", 6, 0.0, 2, -1.8437, -0.2695, -0.1434), .init("C", 6, 0.0, 3, -3.2403, -0.2822, 0.4197), .init("O", 8, 0.0, 4, -3.815, 0.7977, 0.7253), .init("CB", 6, 0.0, 0, -0.8541, 0.3099, 0.883), .init("CG", 6, 0.0, 0, 0.5673, 0.2126, 0.3916), .init("CD1", 6, 0.0, 0, 1.2694, -0.9248, 0.5432), .init("CD2", 6, 0.0, 0, 1.1873, 1.3606, -0.3155), .init("CE1", 6, 0.0, 0, 2.652, -1.0213, 0.0299), .init("CE2", 6, 0.0, 0, 2.4384, 1.2694, -0.7832), .init("CZ", 6, 0.0, 0, 3.2113, 0.0226, -0.6021), .init("OH", 8, 0.0, 0, 4.5135, -0.0526, -1.093)
        ]),
        "VAL": .init(oneLetter: "V", restype: 21, atoms: [
            .init("N", 7, 0.0, 1, 0.9408, -1.2609, 0.6524), .init("CA", 6, 0.0, 2, 0.7288, -0.3938, -0.5121), .init("C", 6, 0.0, 3, 1.767, 0.6979, -0.5508), .init("O", 8, 0.0, 4, 2.245, 1.1684, 0.5171), .init("CB", 6, 0.0, 0, -0.7016, 0.2148, -0.5226), .init("CG1", 6, 0.0, 0, -1.7663, -0.8455, -0.8412), .init("CG2", 6, 0.0, 0, -1.05, 0.9375, 0.7892)
        ]),
    ]

    public static let byOneLetter: [Character: AAResidueTemplate] = {
        var m = [Character: AAResidueTemplate]()
        for (_, t) in byThreeLetter { m[t.oneLetter] = t }
        return m
    }()

    public static func template(threeLetter: String) -> AAResidueTemplate? {
        byThreeLetter[threeLetter.uppercased()]
    }

    public static func template(oneLetter: Character) -> AAResidueTemplate? {
        byOneLetter[oneLetter]
    }
}
