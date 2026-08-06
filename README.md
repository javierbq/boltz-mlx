# boltz-mlx

A native **[MLX](https://github.com/ml-explore/mlx) Swift** port of the
[Boltz-2](https://github.com/jwohlwend/boltz) biomolecular structure-prediction network,
quantized to **int8** and runnable on **iOS / iPadOS / macOS** — including on-device on an
iPhone 15 Pro.

> Structure-prediction path only (trunk + diffusion). Confidence, affinity, potentials, and
> training are out of scope. Built on and requires the upstream Boltz package for feature
> preprocessing and weight export.

## What's here

- **`Sources/BoltzMLX/`** — the Swift/MLX inference library: int8 affine-quantized layers,
  the Boltz-2 trunk (input embedding, template, MSA, 64-block Pairformer) and the diffusion
  score model + sampler.
- **`Sources/BoltzMLXCLI/`** — a macOS command-line runner.
- **`src/boltz_mlx_export/`** — the Python exporter: converts an upstream Boltz checkpoint to
  an int8 MLX artifact (`export-model`), precomputes feature bundles (`export-features`), and
  records PyTorch module-boundary fixtures for parity testing (`make-fixtures`).
- **`examples/BoltzMLXDemo/`** — a SwiftUI iOS demo app + on-device benchmark harness.
- **`validation/`** — int8-vs-fp32 numerical-parity and structural-quality reports, plus the
  iPhone 15 Pro speed/memory benchmarks.
- **`src/boltz/`** — vendored upstream Boltz (MIT), used by the exporter.

## Highlights

- **Numerically faithful to fp32:** every module boundary matches upstream PyTorch at
  Pearson r ≥ 0.9997 (the int8 group-64 quantization floor).
- **Structurally validated:** int8-vs-fp32 matched-noise superposition is sub-Ångström for
  well-determined targets; passes a 2.0 Å / 0.9 lDDT gate at adequate diffusion-step counts.
- **Small:** the int8 model is **529 MB** (~3.75× smaller than the fp32 network).
- **Runs on device:** ~3–13 s per small-protein prediction on an iPhone 15 Pro at ~0.6–1.4 GB
  peak — well within the 8 GB budget.

## Quick start (macOS)

```bash
# 1. Export an int8 artifact from an upstream Boltz checkpoint
pip install -e '.[mlx-export]'
boltz-mlx export-model --checkpoint boltz2_conf.ckpt --output artifacts/boltz2-mlx-int8
boltz-mlx export-features examples/prot_no_msa.yaml --output artifacts/features/prot_no_msa

# 2. Predict with the native MLX CLI
swift run BoltzMLXCLI predict \
  --model artifacts/boltz2-mlx-int8 \
  --features artifacts/features/prot_no_msa \
  --output prediction.safetensors \
  --recycling-steps 0 --diffusion-steps 200 --seed 0
```

See `examples/BoltzMLXDemo/DEVICE_BENCHMARK.md` for the on-device (iPhone) flow.

## Credits & license

Derived from **Boltz** (Wohlwend, Corso, Passaro, et al.) — https://github.com/jwohlwend/boltz.
Both this port and upstream Boltz are released under the **MIT License** (see `LICENSE`).
Uses [mlx-swift](https://github.com/ml-explore/mlx-swift).
