"""Validate the Swift MSA featurizer on a real binder against a crystal structure.

Compares CA coordinates dumped by MSAEndToEndTests against PDB 1BRS (barnase + barstar, a natural
femtomolar complex) and reports four RMSDs per variant. Run the Swift test first:

    BOLTZ_CONF_MODEL=.artifacts/boltz2-mlx-conf BOLTZ_CA_OUT=/tmp/ca_out \
        swift test --filter testARealBinderAgainstADeeplyAlignedTargetScoresAsABinder
    .venv/bin/python scripts/validate_msa_real_binder.py --ca-dir /tmp/ca_out

WHY FOUR NUMBERS AND NOT ONE. A whole-complex superposition hides a mis-docked binder: fit the two
chains together and the error is shared out between them, so a binder placed on the wrong face can
still report a middling complex RMSD. The number that answers "is the binder where it belongs" is
target-anchored -- superpose on the TARGET only, then measure the binder without refitting it. The
binder-alone fit is reported separately so a wrong FOLD can be told apart from a wrong POSE.

Pairs by residue number rather than array position, so unresolved crystal residues (1BRS chain A
starts at 3) cannot silently shift the comparison by two residues.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gemmi
import numpy as np

# 1BRS: barnase is chains A/B/C, barstar D/E/F. The A-D pair is one complex in the asymmetric unit.
# Maps the crystal's chains onto the fixture's chain ids.
CHAIN_MAP = {"A": "A", "B": "D"}


def crystal_alpha_carbons(model: gemmi.Model, chain: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for residue in model[chain]:
        atom = residue.find_atom("CA", "*")
        if atom is not None:
            out[residue.seqid.num] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    return out


def predicted_alpha_carbons(path: Path) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["chain"], {})[int(row["residue"])] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])]
            )
    return out


def superposition(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rotation+translation taking `mobile` onto `target`, reflection-corrected."""
    mobile_centre, target_centre = mobile.mean(0), target.mean(0)
    correlation = (mobile - mobile_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(correlation)
    reflection = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
    return rotation, target_centre - rotation @ mobile_centre


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def apply(rotation: np.ndarray, translation: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (rotation @ points.T).T + translation


def write_reference_pair(structure: gemmi.Structure, path: Path) -> None:
    """Write the crystal's one biological pair as a PDB, chain A = target, chain B = binder.

    WHY NOT LET THE VIEWER READ THE mmCIF. PyMOL's chain selection on this entry is ambiguous — for
    1BRS `count_atoms("chain A and name CA")` reports 110 while `iterate_state` over the same
    selection yields 108 and finds nothing below residue 3 (which is what the file actually contains,
    barnase residues 1-2 being unresolved in chain A but present in chain B). Extracting the pair here
    means the session and the RMSD table are built from one unambiguous view of the crystal instead of
    two that might disagree about two residues.
    """
    out = gemmi.Structure()
    out.spacegroup_hm = "P 1"
    model = gemmi.Model("1")
    for fixture_chain, crystal_chain in CHAIN_MAP.items():
        source = structure[0][crystal_chain]
        chain = gemmi.Chain(fixture_chain)
        for residue in source:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is not None and info.is_amino_acid():
                chain.add_residue(residue)
        model.add_chain(chain)
    out.add_model(model)
    out.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_pdb(str(path))
    counts = {c.name: sum(1 for r in c if r.find_atom("CA", "*")) for c in out[0]}
    print(f"wrote {path}: " + ", ".join(f"chain {k} {v} CA" for k, v in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-dir", type=Path, required=True,
                        help="directory of <variant>.csv files written by the Swift test")
    parser.add_argument("--structure", type=Path, default=Path(".artifacts/1brs.cif"))
    parser.add_argument("--variants", nargs="*",
                        default=["target_only", "both", "no_MSA"])
    parser.add_argument("--write-reference", type=Path, default=None,
                        help="also write the crystal pair as a PDB with chains A (target) and "
                             "B (binder), for the session script to load")
    args = parser.parse_args()

    structure = gemmi.read_structure(str(args.structure))
    structure.setup_entities()
    model = structure[0]
    reference = {
        fixture: crystal_alpha_carbons(model, crystal)
        for fixture, crystal in CHAIN_MAP.items()
    }
    for fixture, crystal in CHAIN_MAP.items():
        resolved = reference[fixture]
        print(f"crystal chain {crystal} -> fixture chain {fixture}: "
              f"{len(resolved)} CA, residues {min(resolved)}-{max(resolved)}")

    if args.write_reference is not None:
        write_reference_pair(structure, args.write_reference)

    header = f"\n{'variant':<14}{'complex':>9}{'target':>9}{'binder fold':>13}{'binder docked':>15}"
    print(header)
    print("-" * len(header.strip()))
    for variant in args.variants:
        path = args.ca_dir / f"{variant}.csv"
        if not path.exists():
            print(f"{variant:<14}  (missing {path})")
            continue
        predicted = predicted_alpha_carbons(path)

        target_keys = sorted(set(predicted["A"]) & set(reference["A"]))
        binder_keys = sorted(set(predicted["B"]) & set(reference["B"]))
        target_mobile = np.array([predicted["A"][k] for k in target_keys])
        target_fixed = np.array([reference["A"][k] for k in target_keys])
        binder_mobile = np.array([predicted["B"][k] for k in binder_keys])
        binder_fixed = np.array([reference["B"][k] for k in binder_keys])

        rotation, translation = superposition(
            np.vstack([target_mobile, binder_mobile]), np.vstack([target_fixed, binder_fixed]))
        complex_rmsd = rmsd(apply(rotation, translation, np.vstack([target_mobile, binder_mobile])),
                            np.vstack([target_fixed, binder_fixed]))

        # Anchor on the target, then measure the binder WITHOUT refitting it.
        rotation, translation = superposition(target_mobile, target_fixed)
        target_rmsd = rmsd(apply(rotation, translation, target_mobile), target_fixed)
        docked_rmsd = rmsd(apply(rotation, translation, binder_mobile), binder_fixed)

        rotation, translation = superposition(binder_mobile, binder_fixed)
        fold_rmsd = rmsd(apply(rotation, translation, binder_mobile), binder_fixed)

        print(f"{variant:<14}{complex_rmsd:8.2f}A{target_rmsd:8.2f}A"
              f"{fold_rmsd:12.2f}A{docked_rmsd:14.2f}A")


if __name__ == "__main__":
    main()
