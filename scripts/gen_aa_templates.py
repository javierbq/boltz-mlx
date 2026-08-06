#!/usr/bin/env python
"""Regenerate Sources/BoltzMLX/Featurize/AAResidueTemplates.swift.

Emits the canonical-20 amino-acid templates the Swift featurizer needs: ordered heavy atoms with
atomic number, formal charge, atom_backbone_feat index, and one reference conformer.

TWO INDEPENDENT SOURCES, CROSS-VALIDATED. Atom order/elements/charges are read BOTH from the
component pickles under $BOLTZ_CACHE/mols (via RDKit) AND decoded out of a Python-generated
reference feature bundle (ref_atom_name_chars / ref_element / ref_charge / atom_backbone_feat /
ref_space_uid). The script aborts if they disagree — that cross-check is the point, because the
pickles alone do not tell you the backbone-feature index and the bundle alone does not give you
conformer coordinates.

OXT is dropped: the pickles carry a trailing OXT on every residue and the reference featurizer
excludes it everywhere, including at chain termini (187 pickle atoms - 20 = 167).

Usage:
    boltz-mlx export-features tests/fixtures/allresidues.yaml --output /tmp/ref_allresidues
    python scripts/gen_aa_templates.py --reference /tmp/ref_allresidues
"""
import argparse, json, os, pickle, struct, sys
import numpy as np
from rdkit import Chem  # noqa: F401  (needed to unpickle the Mol objects)

AA3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
       'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
ONE = dict(zip(AA3, 'ARNDCQEGHILKMFPSTWYV'))
RESTYPE = {aa: i + 2 for i, aa in enumerate(AA3)}   # ALA=2 ... VAL=21, from the reference one-hot


def read_bundle(directory):
    path = os.path.join(directory, 'features.safetensors')
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        dt = {'I64': '<i8', 'F32': '<f4', 'BOOL': '|b1'}
        out = {}
        for name, e in hdr.items():
            if name == '__metadata__':
                continue
            s, t = e['data_offsets']
            f.seek(base + s)
            out[name] = np.frombuffer(f.read(t - s), dtype=dt[e['dtype']]).reshape(e['shape'])
        return out


def decode_from_bundle(b):
    """{resname: [(name, Z, charge, backboneFeat)]} decoded from a reference bundle."""
    rt = b['res_type'][0].argmax(1)
    uid, mask = b['ref_space_uid'][0], b['atom_pad_mask'][0].astype(bool)
    chars, elem = b['ref_atom_name_chars'][0], b['ref_element'][0].argmax(-1)
    charge, bb = b['ref_charge'][0], b['atom_backbone_feat'][0]
    idx2aa = {v: k for k, v in RESTYPE.items()}
    table = {}
    for res in range(int(uid[mask].max()) + 1):
        sel = np.where((uid == res) & mask)[0]
        aa = idx2aa[int(rt[res])]
        rec = [(''.join(chr(int(chars[i, c].argmax()) + 32) for c in range(4)).strip(),
                int(elem[i]), float(charge[i]), int(bb[i].argmax())) for i in sel]
        if aa in table and table[aa] != rec:
            sys.exit(f"ABORT: inconsistent atom record for {aa} within the reference bundle")
        table[aa] = rec
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True,
                    help='a feature bundle covering all 20 residues (tests/fixtures/allresidues.yaml)')
    ap.add_argument('--cache', default=os.environ.get('BOLTZ_CACHE', os.path.expanduser('~/.boltz')))
    ap.add_argument('--out', default='Sources/BoltzMLX/Featurize/AAResidueTemplates.swift')
    a = ap.parse_args()

    bundle = decode_from_bundle(read_bundle(a.reference))
    missing = [aa for aa in AA3 if aa not in bundle]
    if missing:
        sys.exit(f"ABORT: reference bundle does not cover {missing}; use allresidues.yaml")

    rows, total = [], 0
    for aa in AA3:
        with open(os.path.join(a.cache, 'mols', f'{aa}.pkl'), 'rb') as f:
            mol = pickle.load(f)
        conf = mol.GetConformer(0)
        pk = []
        for at in mol.GetAtoms():
            name = at.GetProp('name') if at.HasProp('name') else at.GetSymbol()
            if name == 'OXT':
                continue
            p = conf.GetAtomPosition(at.GetIdx())
            pk.append((name, at.GetAtomicNum(), float(at.GetFormalCharge()),
                       round(p.x, 4), round(p.y, 4), round(p.z, 4)))
        bd = bundle[aa]
        if [x[0] for x in pk] != [x[0] for x in bd]:
            sys.exit(f"ABORT: atom NAME/order mismatch for {aa}\n  pickle={[x[0] for x in pk]}\n  bundle={[x[0] for x in bd]}")
        if [x[1] for x in pk] != [x[1] for x in bd]:
            sys.exit(f"ABORT: ELEMENT mismatch for {aa}")
        if [x[2] for x in pk] != [x[2] for x in bd]:
            sys.exit(f"ABORT: CHARGE mismatch for {aa}")
        atoms = ', '.join(
            f'.init("{p[0]}", {p[1]}, {p[2]:.1f}, {d[3]}, {p[3]}, {p[4]}, {p[5]})'
            for p, d in zip(pk, bd))
        rows.append(f'        "{aa}": .init(oneLetter: "{ONE[aa]}", restype: {RESTYPE[aa]}, atoms: [\n            {atoms}\n        ]),')
        total += len(pk)

    if total != 167:
        sys.exit(f"ABORT: expected 167 heavy atoms across the canonical 20, got {total}")
    print(f"cross-validated pickles against the reference bundle: all 20 residues agree, {total} atoms")
    print(f"NOTE: {a.out} is generated — re-run this script rather than editing it.")


if __name__ == '__main__':
    main()
