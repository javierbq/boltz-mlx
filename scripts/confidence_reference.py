#!/usr/bin/env python
"""Record a PyTorch confidence-head reference: its inputs AND its PAE output.

The Swift ConfidenceModule must reproduce PAE from (s_inputs, s, z, x_pred, feats). Capturing all
of those alongside the resulting pae_logits makes the Swift side directly comparable without having
to reproduce the trunk bit-exactly first — which is the only way to attribute a mismatch to the
confidence head rather than to everything upstream of it.

Usage: python scripts/confidence_reference.py <features-dir> <out-dir>
"""
import argparse, pathlib, sys
import numpy as np, torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--checkpoint", type=pathlib.Path,
                    default=pathlib.Path(".artifacts/boltz2_conf.ckpt"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import load_file, save_file
    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2
    from boltz.model.models.boltz2 import Boltz2
    from dataclasses import asdict

    model = Boltz2.load_from_checkpoint(
        a.checkpoint, strict=True, map_location="cpu",
        predict_args={"recycling_steps": 0, "sampling_steps": 20,
                      "diffusion_samples": 1, "max_parallel_samples": 1},
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams()),
    ).eval()

    feats = {k: torch.from_numpy(v.copy())
             for k, v in load_file(str(a.features / "features.safetensors")).items()}
    n = int(feats["token_index"].shape[1])
    aPad = int(feats["atom_pad_mask"].shape[1])
    tokenS, tokenZ = model.confidence_module.s_norm.normalized_shape[0], \
                     model.confidence_module.z_norm.normalized_shape[0]
    print(f"tokens {n}  padded atoms {aPad}  token_s {tokenS}  token_z {tokenZ}")

    # Synthetic but DETERMINISTIC trunk outputs. The point is to test the confidence head in
    # isolation, so its inputs must be reproducible on both sides rather than inherited from a
    # trunk run: any trunk difference would otherwise be misread as a confidence bug.
    g = torch.Generator().manual_seed(7)
    def rnd(*shape):
        return (torch.rand(*shape, generator=g, dtype=torch.float32) - 0.5) * 2.0
    s_inputs = rnd(1, n, tokenS)
    s = rnd(1, n, tokenS)
    z = rnd(1, n, n, tokenZ) * 0.1
    z = z + z.transpose(1, 2)                      # keep it symmetric, like a real pair repr
    x_pred = rnd(1, aPad, 3) * 20.0

    with torch.no_grad():
        out = model.confidence_module(
            s_inputs=s_inputs, s=s, z=z, x_pred=x_pred, feats=feats,
            pred_distogram_logits=torch.zeros(1, n, n, 64), multiplicity=1,
        )
    if "pae_logits" in out:
        pae_logits = out["pae_logits"]
    else:
        print("keys:", sorted(out)); sys.exit("ABORT: no pae_logits in the output")

    tensors = {
        "s_inputs": s_inputs.numpy(), "s": s.numpy(), "z": z.numpy(), "x_pred": x_pred.numpy(),
        "pae_logits": pae_logits.detach().numpy().astype(np.float32),
    }
    for k in ("pae", "pae_bins"):
        if k in out:
            tensors[k] = out[k].detach().numpy().astype(np.float32)
    save_file(tensors, str(a.out / "confidence_reference.safetensors"))
    print(f"wrote {a.out}/confidence_reference.safetensors  pae_logits {tuple(pae_logits.shape)}")
    print("output keys:", sorted(out))


if __name__ == "__main__":
    main()
