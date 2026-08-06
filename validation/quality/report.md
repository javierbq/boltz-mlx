# MLX int8 structural quality report

Status: **int8 validated end-to-end at adequate diffusion-step counts.**

## Method

int8-MLX vs fp32-PyTorch, two complementary comparisons:

1. **Module-boundary parity.** Every trunk / diffusion-conditioning / score-model
   boundary is compared (Pearson r) against `make-fixtures` outputs from the fp32
   checkpoint on the same feature bundle.
2. **Matched-noise end-to-end.** Reverse diffusion is driven with *identical* injected
   initial/step noise and identity augmentation on both backends, so the trajectories are
   directly comparable. All-atom RMSD and lDDT are computed over resolved atoms after
   Kabsch alignment.

## Boundary parity (`prot_no_msa`)

All module boundaries match fp32 at **Pearson r >= 0.9997** — the network port is faithful
to the int8 group-64 quantization floor (~1-2 % per tensor).

## Matched-noise end-to-end (`prot_no_msa`, 899 resolved atoms)

| Diffusion steps | Aligned all-atom RMSD | lDDT | Gate (<=2.0 A, >=0.90) |
| --- | --- | --- | --- |
| 20 | 3.19 A | 0.685 | fail |
| **50** | **0.64 A** | **0.977** | **pass** |

## Conclusion

- **The network port is faithful** (boundaries r >= 0.9997).
- **int8-vs-fp32 end-to-end agreement is step-count dependent.** With few steps the Euler
  corrections are large, so the ~1-2 % per-step int8 error has high leverage and the
  trajectories diverge (20 steps -> 3.19 A). Finer integration removes it (50 steps ->
  0.64 A / lDDT 0.977, comfortably inside the gate) — on the *hardest* target (no MSA,
  under-determined structure).
- **int8 (group-64, 8-bit) is viable end-to-end at production step counts; fp16 is not
  required for structural fidelity.**

## Not yet run

- `prot_msa_monomer` (384 tokens, MSA depth 249): **now runs end-to-end on the Mac (~28 s)**
  after fixing the AtomEncoder `z_to_p_trans` einsum OOM (commit `1f17753`) — the earlier
  ~28 GB single-buffer allocation is gone. A matched-noise quality comparison for it is not
  yet computed.
- `examples/ligand.yaml` (846 tokens): larger still; not yet re-tested since the fix (may
  hit other size limits).
- Physical iPhone 15 Pro benchmark: done for small proteins (see `validation/device/`).

The release gate is an aligned all-atom RMSD <= 2.0 A and lDDT >= 0.90.
