# Entry 1 — Detailed Log

## Context

Benchmarking the int8 (affine, group-size 64) MLX Swift port of the Boltz-2
structure-prediction network for **speed and peak memory across protein sizes**, on
macOS (Apple Silicon) and on a physical iPhone 15 Pro. Also root-caused and fixed a
large-protein OOM discovered during the sweep.

## Inputs

Six no-MSA proteins — truncations of the `prot_no_msa` test sequence (`QLEDSEV…`):
`prot040` (40 aa), `prot080` (80), `prot115` (115), `prot200` (200), `prot300` (300),
`prot384` (384). Feature bundles built with the upstream Boltz preprocessor:

```bash
~/.venvs/boltz-mlx/bin/boltz-mlx export-features .artifacts/protNNN.yaml \
  --output .artifacts/features/protNNN
```

int8 model artifact: `.artifacts/boltz2-mlx-int8/model.safetensors` (504 MB), exported
from `boltz2_conf.ckpt` (upstream v2.2.1, commit `cb04aecc`).

## Mac size sweep

Driven by `BoundaryDumpTests.testBenchLocal` (raised the provisional `MemoryPlanner` cap
to allow >256 tokens; peak reset per protein via `GPU.resetPeakMemory()`; model loaded
once, recycling 0, 20 diffusion steps):

```bash
xcodebuild test -scheme BoltzMLX-Package -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation \
  -only-testing:BoltzMLXTests/BoundaryDumpTests/testBenchLocal
```

Result: `.artifacts/boundary/local_scaling.json` → `mac_scaling.json`.

## iPhone 15 Pro run

App `examples/BoltzMLXDemo` (bundle `io.github.javierbq.boltzmlx`), signed with the
personal team (YOUR_TEAM_ID), installed over cable; artifacts pushed with `devicectl` into
`Documents/<name>` subdirs; `BenchmarkRunner.runAll()` launched via a `BENCH_ALL` env var
and `results_all.json` read back. Clean 3-point run (40/80/115) → `iphone_scaling.json`.
On-device 200/300/384 did not complete — the app suspends during the longer runs when it
is not kept in the foreground.

## Large-protein OOM fix (commit `1f17753`)

The 384-token sweep initially OOM'd. Root cause (exact byte match + isolation in
`TriangleMemoryTests`): the AtomEncoder `z_to_p_trans` 3-way einsum
`bijd,bwki,bwlj->bwkld` — MLX's contraction-path optimizer materialized an
`atoms × tokens² × atom_z` intermediate (28 GB at 384 tok / 2976 atoms; 432 GB attempted
at 192 tok in isolation; fine ≤96 tokens — the path is size-dependent). Fixed by
contracting `i` then `j` in two einsums (numerically identical; boundary parity unchanged
at r=0.99987). Full end-to-end at 384 tok now runs on the Mac (~28 s).

## Environment

- Python venv: `~/.venvs/boltz-mlx` (torch 2.13, boltz v2.2.1 + mlx-export).
- MLX Swift pinned 0.31.6; iOS deployment target 17.0; device iOS 26.5.2.
- Mac MLX cache 256 MB; iPhone MLX cache 64 MB (default) — affects peak readings.
