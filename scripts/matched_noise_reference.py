#!/usr/bin/env python
"""Run upstream Boltz and RECORD its reverse-diffusion noise, for matched-noise comparison.

WHY. A bare cross-backend RMSD on a no-MSA target is uninterpretable: two upstream runs differing
only in seed disagree by 9-20 A, because the prediction is not converged. Matching the injected
noise removes trajectory stochasticity so the remaining difference is attributable to the
implementations (features + network) rather than to the sampler.

WHAT IS MATCHED. Three sources of randomness in the reverse diffusion, all patched or recorded:
  1. ref_pos roto-translation at featurization  -> forced to IDENTITY here, and the Swift side is
     run with BoltzFeaturizer(identityAugmentation: true) to match.
  2. per-step centre_random_augmentation of the running coordinates -> forced to IDENTITY here; the
     Swift sampler defaults to identity when rotations/translations are not supplied.
  3. the initial noise and the per-step noise -> RECORDED here and injected into Swift via
     BoltzPredictor.MatchedNoise.

Outputs <out>/noise.safetensors (initial + step_000..) and <out>/reference.pdb.

Usage:
  python scripts/matched_noise_reference.py tests/fixtures/twochain.yaml /tmp/matched \
      --recycling 3 --steps 200
"""
import argparse
import pathlib
import sys

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--accelerator", default="gpu", help="gpu resolves to MPS on Apple silicon")
    ap.add_argument("--export-features", action="store_true",
                    help="also export the identity-augmented feature bundle, for isolating the "
                         "network from the featurizer")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import boltz.model.modules.diffusionv2 as dv2
    import boltz.data.feature.featurizerv2 as fv2
    from safetensors.numpy import save_file

    # ---- (1) identity ref_pos augmentation -------------------------------------------------
    # featurizerv2 calls center_random_augmentation per residue to roto-translate the reference
    # conformer. Returning the centred coordinates unchanged makes it deterministic and matches
    # BoltzFeaturizer(identityAugmentation: true).
    def identity_center(coords, mask, **kw):
        mean = (coords * mask[:, :, None]).sum(1, keepdim=True) / mask[:, :, None].sum(1, keepdim=True)
        return coords - mean

    fv2.center_random_augmentation = identity_center

    # ---- (2) identity per-step augmentation ------------------------------------------------
    def identity_augmentation(batch_size, device=None, **kw):
        R = torch.eye(3, device=device).expand(batch_size, 3, 3).clone()
        tr = torch.zeros(batch_size, 1, 3, device=device)
        return R, tr

    dv2.compute_random_augmentation = identity_augmentation

    # ---- (3) record the sampler's noise ----------------------------------------------------
    # Only draws whose shape matches the atom-coordinate tensor [B, A, 3] belong to the reverse
    # diffusion; anything else (e.g. training-time draws) is passed through untouched.
    recorded: list[np.ndarray] = []
    real_randn = torch.randn

    def recording_randn(*shape, **kw):
        out = real_randn(*shape, **kw)
        dims = out.shape
        if len(dims) == 3 and dims[-1] == 3:
            recorded.append(out.detach().to("cpu", torch.float32).numpy().copy())
        return out

    torch.randn = recording_randn

    # ---- run ------------------------------------------------------------------------------
    from boltz.main import cli

    argv = [
        "predict", str(a.input),
        "--out_dir", str(a.out / "prediction"),
        "--accelerator", a.accelerator,
        "--devices", "1",
        "--num_workers", "0",
        "--recycling_steps", str(a.recycling),
        "--sampling_steps", str(a.steps),
        "--diffusion_samples", "1",
        "--seed", "0",
        "--output_format", "pdb",
        "--no_kernels",
        "--override",
    ]
    try:
        cli(argv, standalone_mode=False)
    finally:
        torch.randn = real_randn

    if not recorded:
        sys.exit("ABORT: recorded no [B, A, 3] noise draws — the hook did not fire")

    # The first matching draw is the initial noise; the rest are the per-step draws.
    initial, steps = recorded[0], recorded[1:]
    print(f"recorded initial noise {initial.shape} + {len(steps)} step draws")
    if len(steps) != a.steps:
        print(f"NOTE: {len(steps)} step draws for {a.steps} requested steps — "
              f"the Swift side must consume exactly this many")

    tensors = {"initial": initial}
    for i, s in enumerate(steps):
        tensors[f"step_{i:04d}"] = s
    save_file(tensors, str(a.out / "noise.safetensors"))

    pdbs = list((a.out / "prediction").rglob("*_model_0.pdb"))
    if not pdbs:
        sys.exit("ABORT: no prediction PDB produced")
    (a.out / "reference.pdb").write_bytes(pdbs[0].read_bytes())
    # Record WHICH fixture produced this noise, so a consumer cannot pair it with a different one
    # (the atom axis would mismatch and the comparison would be meaningless).
    (a.out / "fixture.txt").write_text(a.input.stem + "\n")
    print(f"wrote {a.out}/noise.safetensors, reference.pdb and fixture.txt ({a.input.stem})")

    # ---- also export the FEATURES, through the same identity-patched featurizer ---------------
    # This is what isolates featurizer from network: feeding these to MLX puts Python features on
    # both sides, so any remaining difference is the quantised network alone.
    if a.export_features:
        from boltz_mlx_export.feature_export import export_input_features

        bundle = a.out / "features"
        export_input_features(input_path=a.input, output=bundle)
        print(f"wrote {bundle} (identity-augmented features for network isolation)")


if __name__ == "__main__":
    main()
