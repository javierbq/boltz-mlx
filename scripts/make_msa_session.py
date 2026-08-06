"""Build a PyMOL session comparing the three MSA variants against the 1BRS crystal.

    $HOME/repos/RayMol/.venv/bin/pymol -cq scripts/make_msa_session.py -- \
        --ca-dir /tmp/ca_out --structure .artifacts/1brs.cif --output /tmp/msa_barnase_barstar.pse

Every model is superposed on the CRYSTAL TARGET ONLY (chain A CA, cycles=0 so no outlier rejection),
which is the frame the binder-docking metric is defined in: fitting the whole complex would spread the
error across both chains and make a mis-docked binder look closer than it is.

Also re-derives target and binder RMSD using PyMOL's own align/rms_cur as an independent check on
scripts/validate_msa_real_binder.py -- two implementations agreeing is worth more than one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pymol import cmd

# label -> (file stem, colour, min_ipSAE measured by MSAEndToEndTests)
VARIANTS = [
    ("both_aligned",     "both",        "green",  0.9413),
    ("target_msa_only",  "target_only", "orange", 0.7000),
    ("no_msa",           "no_MSA",      "red",    0.0000),
]


def common_ca(object_a: str, object_b: str, chain: str) -> tuple[str, str]:
    """Selections of CA atoms present in BOTH objects for `chain`, ordered by residue.

    The crystal does not resolve every residue (1BRS chain A starts at 3, and barstar is missing two),
    so an unfiltered rms_cur would either fail on the atom count or, worse, pair the wrong residues.
    """
    def residues(obj: str) -> set[int]:
        found: set[int] = set()
        cmd.iterate(f"{obj} and chain {chain} and name CA and polymer",
                    "found.add(int(resi))", space={"found": found})
        return found

    shared = sorted(residues(object_a) & residues(object_b))
    if not shared:
        raise SystemExit(f"no shared chain {chain} residues between {object_a} and {object_b}")
    spec = "+".join(str(r) for r in shared)
    return (f"{object_a} and chain {chain} and name CA and resi {spec}",
            f"{object_b} and chain {chain} and name CA and resi {spec}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="crystal pair PDB from "
                             "`validate_msa_real_binder.py --write-reference` (chain A target, "
                             "chain B binder). NOT the raw mmCIF -- see that script for why.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cmd.set("retain_order", 0)
    cmd.set("cartoon_transparency", 0)

    # ---- crystal reference: already reduced to one pair with chains A/B ----
    cmd.load(str(args.reference), "crystal_1BRS")
    print(f"crystal_1BRS: {cmd.count_atoms('crystal_1BRS and name CA')} CA "
          f"(target {cmd.count_atoms('crystal_1BRS and chain A and name CA')}, "
          f"binder {cmd.count_atoms('crystal_1BRS and chain B and name CA')})")

    print(f"\n{'variant':<18}{'min_ipSAE':>10}{'target fit':>12}{'binder docked':>15}")
    print("-" * 55)
    rows = []
    for label, stem, colour, gate in VARIANTS:
        path = args.ca_dir / f"{stem}.pdb"
        if not path.exists():
            print(f"{label:<18}  (missing {path}) -- run the Swift test with BOLTZ_CA_OUT")
            continue
        # Object names carry the gate value so the session is self-describing.
        obj = f"{label}_ipSAE{gate:.2f}"
        cmd.load(str(path), obj)

        mobile, fixed = common_ca(obj, "crystal_1BRS", "A")
        target_rmsd = cmd.align(mobile, fixed, cycles=0)[0]
        # rms_cur does NOT refit -- it measures where the binder landed once the target was anchored.
        binder_mobile, binder_fixed = common_ca(obj, "crystal_1BRS", "B")
        binder_rmsd = cmd.rms_cur(binder_mobile, binder_fixed, matchmaker=-1)

        print(f"{label:<18}{gate:>10.4f}{target_rmsd:>11.2f}A{binder_rmsd:>14.2f}A")
        rows.append((obj, colour))

        cmd.color("grey70", f"{obj} and chain A")
        cmd.color(colour, f"{obj} and chain B")

    # ---- presentation ----
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("cartoon_fancy_helices", 1)

    cmd.hide("everything")
    # Crystal target as a translucent surface (the receptor), crystal binder as the ANSWER in blue.
    cmd.show("surface", "crystal_1BRS and chain A")
    cmd.color("grey80", "crystal_1BRS and chain A")
    cmd.set("transparency", 0.55, "crystal_1BRS")
    cmd.show("cartoon", "crystal_1BRS and chain B")
    cmd.color("marine", "crystal_1BRS and chain B")

    # The predicted TARGETS all superpose onto the crystal target by construction; leaving them on
    # just thickens the same tube. The binder poses are the comparison.
    for obj, _ in rows:
        cmd.show("cartoon", f"{obj} and chain B")

    cmd.group("predictions", " ".join(obj for obj, _ in rows))
    cmd.orient("crystal_1BRS")

    # One scene per variant: three overlapping semi-transparent binders in one view is unreadable,
    # and the question being asked is per-variant anyway ("did THIS one dock where barstar sits").
    def only(visible: str | None, name: str, message: str) -> None:
        for obj, _ in rows:
            if visible is not None and obj == visible:
                cmd.enable(obj)
            else:
                cmd.disable(obj)
        cmd.scene(name, "store", message=message)

    only(None, "0_crystal_reference",
         "Crystal 1BRS: barnase surface (grey), barstar in blue = where a real binder sits")
    for (obj, _), (label, _, _, gate) in zip(rows, VARIANTS):
        only(obj, f"{len(cmd.get_scene_list())}_{label}",
             f"{label}: min_ipSAE {gate:.4f}. Blue = crystal barstar.")
    for obj, _ in rows:
        cmd.enable(obj)
    cmd.zoom("crystal_1BRS or predictions", buffer=3)
    cmd.scene("4_all_variants", "store",
              message="All three: green=both aligned, orange=target MSA only, red=no MSA")
    cmd.scene("0_crystal_reference", "recall")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd.save(str(args.output))
    print(f"\nwrote {args.output}")


# PyMOL execs a `-cq script.py` with __name__ == "pymol", NOT "__main__". A plain
# `if __name__ == "__main__"` guard therefore does nothing at all, and the run looks like a success
# that produced no session.
if __name__ in {"__main__", "pymol"}:
    main(sys.argv[1:])
