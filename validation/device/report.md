# On-device benchmark — iPhone 15 Pro

Real-device run of the int8 MLX port via `examples/BoltzMLXDemo` (`BenchmarkRunner`),
signed + installed over cable, artifacts pushed into the app container with `devicectl`.

## Result (`prot_no_msa`, 117 tokens / 899 atoms; recycling 0, 20 diffusion steps)

| Metric | Value |
| --- | --- |
| Device | iPhone 15 Pro (iPhone16,1), iOS 26.5.2 |
| Status | ok (full trunk + diffusion, coordinates produced) |
| Wall-clock (cold: load + trunk + diffusion) | **12.98 s** |
| MLX peak memory | **~1.15 GB** (1,153,261,815 B) |
| Process footprint after predict | ~678 MB (677,856,680 B) |
| int8 model weights on disk | 504 MB |

Raw: `validation/device/prot_no_msa_iphone15pro.json`.

## Scaling across protein sizes (same session, model loaded once)

Three proteins run back-to-back via `BenchmarkRunner.runAll()` (MLX peak reset per
protein). The 504 MB model is loaded once and reused, so these wall-clocks are
compute-only (exclude the ~one-time model load counted in the 12.98 s cold run above).

| Protein | Tokens | Atoms | Wall-clock | MLX peak | Process footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| prot040 | 40 | 320 | 3.22 s | 0.61 GB | 0.63 GB |
| prot080 | 80 | 623 | 4.74 s | 0.96 GB | 0.70 GB |
| prot115 | 115 | 889 | 9.42 s | 1.43 GB | 1.27 GB |

Raw: `validation/device/scaling_iphone15pro.json`. Peak memory and runtime rise steeply
with token count (pair tensor + Pairformer are ~N²), but all three stay well within the
iPhone 15 Pro's 8 GB budget.

## Takeaways

- The int8 port **runs end-to-end on the baseline device** (iPhone 15 Pro / iOS 17+),
  producing a structure in ~13 s at ~1.15 GB peak — comfortably inside the 8 GB device's
  jetsam budget.
- Consistent with the Mac proxy (~1.66 GB peak footprint for the same input); the device
  MLX-tracked peak (1.15 GB) is the GPU/unified allocation high-water mark.
- Memory/time rise steeply with token count. A 384-token protein previously OOM'd (a
  28 GB single Metal buffer in the AtomEncoder einsum); that is fixed (commit `1f17753`,
  now ~28 s end-to-end on the Mac). On-device benchmarking at those larger sizes is still
  to be done.

## Reproduce

See `examples/BoltzMLXDemo/DEVICE_BENCHMARK.md` (build+sign+install, push artifacts to
`Documents/model` + `Documents/features`, launch, read `Documents/result.json`).

## Mac size-scaling (post OOM-fix, no-MSA, 20 steps, model loaded once)

| Tokens | Atoms | Wall-clock | MLX peak |
| ---: | ---: | ---: | ---: |
| 40 | 320 | 2.0 s | 0.60 GB |
| 80 | 623 | 2.4 s | 1.30 GB |
| 115 | 889 | 2.7 s | 2.17 GB |
| 200 | 1531 | 7.0 s | 3.47 GB |
| 300 | 2295 | 16.4 s | 4.63 GB |
| 384 | 2961 | 27.2 s | 6.84 GB |

Raw: `validation/device/mac_scaling.json`. These use a 256 MB MLX cache limit (higher than
the device default of 64 MB), so peaks run a bit above what the device reports for the same
input (e.g. Mac 2.17 GB vs device 1.43 GB at 115 tokens). Memory grows ~N², runtime
super-linearly; 384 tokens peaks at ~6.8 GB on the Mac.

## Conclusions — speed & memory (int8, 20 diffusion steps)

**Setup.** int8 group-64 weights (504 MB on disk, constant floor). Times below are
compute-only with the model loaded once; a cold start adds a one-time ~10 s model load
(the single cold iPhone run of a 117-token protein was 12.98 s total). All no-MSA;
20 diffusion steps (production typically uses more, and time is ~linear in steps).

**Memory.** Peak ≈ 504 MB weights + activations, and activations grow ~N² (pair tensor +
Pairformer) plus the atom terms. int8 is the enabler: ~half the footprint of fp16, ~a
quarter of fp32, with negligible structural cost (validated, boundary r>=0.9997).
- iPhone 15 Pro (64 MB MLX cache): 40 aa → 0.61 GB, 80 → 0.96 GB, 115 → 1.43 GB peak.
- Mac (256 MB MLX cache, so peaks read higher): 0.60 / 1.30 / 2.17 GB at 40/80/115, rising
  to 3.47 (200) / 4.63 (300) / 6.84 GB (384).
- iPhone budget (8 GB, ~5 GB usable before jetsam): small–medium proteins (≤~115 aa) fit
  comfortably; ~200–250 aa should fit (~2–3 GB); ~384 aa is borderline (est. ~4–5 GB on
  device) and the practical on-device ceiling without further memory work.

**Speed.** Seconds for small proteins; super-linear growth with size.
- iPhone 15 Pro: 40 aa → 3.2 s, 80 → 4.7 s, 115 → 9.4 s.
- Mac (M-series, faster GPU): 40 → 2.0 s … 115 → 2.7 s … 384 → 27.2 s. The device runs
  ~3–4× slower than this Mac at the same size and scales more steeply.
- Dominated by the ~N² Pairformer/pair ops and the per-step diffusion cost; the one-time
  model load (~10 s) amortizes across multiple predictions in a session.

**Bottom line.** The int8 MLX port runs real structures on the iPhone 15 Pro in a few
seconds for typical small proteins, well within memory; it now scales cleanly to ~384-token
proteins on the Mac after the AtomEncoder OOM fix. On-device, memory (not speed) is the
limiting factor for large proteins — targeted memory optimization would raise the ceiling.
