# Swift featurizer + MLX int8 vs upstream PyTorch fp32 (MPS)

Same machine (M3 Pro, 36 GiB), same settings on both sides: **recycling 3, 50 diffusion steps,
seed 0, single-sequence (`msa: empty`)**. Upstream is Boltz v2.2.1 — the version the MLX port was
made from — run with `--accelerator gpu`, which Lightning resolves to MPS
(`GPU available: True (mps), used: True`). The MPS fork at v2.1.1 was deliberately NOT used: it is
a different model version and would confound the comparison.

## Featurization

The new capability, and the largest ratio in this report.

| fixture | tokens | atoms | Swift | Python in-process (warm) | Python as a process |
| --- | ---: | ---: | ---: | ---: | ---: |
| allresidues | 20 | 167 | **0.001 s** | — | — |
| protnomsa | 117 | 899 | **0.004 s** | 0.490 s | 3.65 s |
| twochain | 225 | 1779 | **0.011 s** | 0.948 s | 4.12 s |

~90x faster in-process, ~370x against the realistic per-call cost of a Python process. But the
speed was never the argument: Swift featurization removes torch (~501 MB), rdkit (~102 MB across 62
`.so` files), numba's LLVM JIT and the 2.1 GB `~/.boltz` component cache from the deployment, and
replaces them with a 167-atom compiled table. Verified: no python/torch/rdkit/numpy linkage in the
binary, no file I/O in the featurize path, and the suite passes with `BOLTZ_CACHE` pointed at a
nonexistent directory.

## Prediction time

| fixture | tokens | MLX int8 predict | MPS fp32 predict | MPS total wall |
| --- | ---: | ---: | ---: | ---: |
| allresidues | 20 | 3.90 s | ~2 s | 16.9 s |
| protnomsa | 117 | 8.90 s | ~7 s | 20.9 s |
| twochain | 225 | 32.16 s | ~34 s | 48.2 s |

MLX is NOT uniformly faster here, and it is worth being precise about why this differs from the
earlier notes-entry table (which reported 1.2-3.2x in MLX's favour): that was measured at recycling
0 / 20 steps and without a confidence head. At recycling 3 the trunk runs four times, and both sides
now also run the 8-block confidence pairformer. MPS wins at small sizes where MLX's fixed per-call
overhead dominates; they converge by ~225 tokens.

MPS predict times are from the progress bar and have ~1 s resolution. "Total wall" includes ~14 s of
interpreter start and checkpoint load, which the MLX figures exclude — MLX is measured after a
warm-up, matching the Python side being measured after its own load.

## Memory

| fixture | tokens | MLX peak (unified) | MPS peak RSS |
| --- | ---: | ---: | ---: |
| allresidues | 20 | 0.61 GB | 5.05 GB |
| protnomsa | 117 | 2.24 GB | 5.05 GB |
| twochain | 225 | 3.47 GB | 5.05 GB |

**These two columns measure different things and must not be subtracted.** The MLX figure is MLX's
own unified-memory high-water mark, excluding the host process. The MPS figure is whole-process peak
RSS, including the interpreter, torch and the 2.3 GB checkpoint. That is why MPS RSS is flat at
5.05 GB across a 10x range of problem size: at these sizes the fixed cost dwarfs the activations,
so it says almost nothing about the workload. The MLX column does scale (~N^2, as expected from the
pair tensor and pairformer).

The honest comparison is deployment footprint rather than peak: MLX needs a 530 MB pack and no
interpreter; the Python path needs ~5 GB resident plus a 2.1 GB component cache on disk.

## RMSD between solutions

All-atom, paired by (residue ordinal, atom name) rather than by position — position-based pairing
would silently assume both sides order side chains identically. Zero unmatched atoms in every case.

| fixture | atoms | MLX vs MPS (all-atom) | MLX vs MPS (CA) | **upstream floor** (all-atom) | floor (CA) |
| --- | ---: | ---: | ---: | ---: | ---: |
| allresidues | 167 | 2.04 A | 1.31 A | 1.41 A | 0.78 A |
| protnomsa | 899 | **6.29 A** | 5.72 A | **6.50 A** | 5.95 A |
| twochain | 1779 | 19.19 A | 19.26 A | 15.96 A | 15.27 A |

**FLOOR = upstream seed 0 vs upstream seed 1** — the same code, same settings, disagreeing with
itself because these are unconverged single-sequence targets. An MLX-vs-MPS number is only
meaningful relative to it, and on `protnomsa` — the target the repo's own validation uses — MLX vs
MPS (6.29 A) is INSIDE upstream's own run-to-run spread (6.50 A).

`twochain` is above its floor, but both are in a regime where the prediction does not exist in any
useful sense: two upstream seeds differ by 16 A on it, and the chains are separately measured 28.7 A
apart with 28 CA contacts under 8 A. `allresidues` is 20 residues of one of each type — a
degenerate "protein", included to exercise every template rather than to be folded.

### The interpretable number

Because the floors above are large, the attributable difference must be measured with the noise
matched (identical injected initial and per-step noise, identity augmentation on both sides). Under
that protocol, on `protnomsa`, all-atom over 899 atoms:

| comparison | all-atom RMSD |
| --- | ---: |
| Python features + MLX int8 vs MPS fp32 (**network alone**) | **1.69 A** |
| Swift features + MLX int8 vs MPS fp32 | 2.62 A |
| attributable to the featurizer | 0.93 A |

1.69 A is inside the repo's established 2.0 A all-atom gate, measured the same way that gate was
defined. The 0.93 A featurizer term is upstream drawing a random side-chain CONFORMER per residue
instance, which a fixed conformer cannot reproduce — evidenced by the internal-geometry spread
between two instances of the same residue type in upstream's own identity-augmented `ref_pos`:
GLY (no side chain) 0.055 A, ALA 0.63 A, GLU 2.56 A, TRP 2.55 A. It scales with side-chain
flexibility and vanishes where there is no side chain.

## Reproducing

```bash
# MLX side
BOLTZ_CONF_MODEL=<pack> BOLTZ_BENCH_OUT=/tmp/bench \
  BOLTZ_BENCH_RECYCLING=3 BOLTZ_BENCH_STEPS=50 \
  swift test --filter testBenchmarkAgainstUpstream

# MPS side, both seeds for the floor
/usr/bin/time -l boltz predict tests/fixtures/<f>.yaml --out_dir <d> \
  --accelerator gpu --devices 1 --num_workers 0 \
  --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 \
  --seed <0|1> --output_format pdb --no_kernels --override

# matched-noise variant
python scripts/matched_noise_reference.py tests/fixtures/protnomsa.yaml /tmp/mn50 \
  --recycling 3 --steps 50 --export-features
```
