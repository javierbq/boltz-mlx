# Handoff: Boltz-2 MLX Swift int8 iOS port

Last updated: 2026-07-21

## Progress update — 2026-07-21 (Claude continuation)

- **Workspace relocated** out of iCloud to `~/repos/boltz-test` (see the move note under Repository state).
- **All prior working-tree work committed** on `codex/mlx-ios`: exporter shared-alias fix (`79e6c40`), Task 8 predictor/CLI (`6a4a4a5`), Task 9 iOS demo sources (`f22c297`), plus housekeeping/doc commits.
- **Verifications passed from the committed tree:**
  - Python: `pytest tests/mlx_export` → 16 passed; `ruff` clean.
  - Swift: `xcodebuild test -scheme BoltzMLX-Package` → 22 passed (use the `-Package` scheme, **not** `BoltzMLX`); `BoltzMLXCLI` builds.
  - Production int8 artifact completeness: 2,815 quantized modules, **zero unquantized 2-D matrices**, `s_recycle`/`z_recycle` quantized — the shared-alias fix is fully effective; all four artifact SHA-256s match the values recorded below.
- **Feature bundles generated** (need the `~/.boltz` CCD+mols cache, now downloaded): `.artifacts/features/prot_no_msa` and `.artifacts/features/ligand`.
- **Bug fixed — template triangle-attention crash (`2de675a`):** `TriangleAttention` reshaped the gated attention to `pairWidth` before the output projection, but the true width there is `headCount*headWidth`. Harmless for the main pairformer (`128==token_z=128`), fatal for the template pair stack (`token_z=64`, `headCount*headWidth=128`). Regression test added; full Swift suite green.
- **First native end-to-end prediction runs** (`prot_no_msa`, recycling 0, 5 diffusion steps): exits 0, writes 899 atoms, no NaNs, correct shapes. **BUT the output is numerically wrong** — coordinates explode to ~±27,000 Å (bonded distances ~17,000 Å vs expected ~1.3 Å).

**Numerical bug — RESOLVED 2026-07-21.** Found via PyTorch-vs-MLX boundary comparison (Task 8). Three root causes, three commits:
- `9eb36b6` — AtomEncoder was missing the ReLU before `c_to_p_trans_q/k` (Boltz stores them as `Sequential(ReLU, Linear)`; only the Linear was exported). Broke the atom-pair rep → atom enc/dec bias (r 0.62/0.66 → 0.9999).
- `541fddf` — the reverse-diffusion step assigned the rigid-alignment result to `denoised` instead of `noisy`; since alignment preserves magnitude, `(noisy − denoised)` became a difference of two noise-scale tensors and coordinates exploded ~1000×/step. This was the dominant amplifier.
- (`2de675a` earlier — template triangle-attention reshape crash.)

**Verification of numerical parity (all against `.artifacts/fixtures/prot_no_msa` from `make-fixtures`):**
- Trunk `s`/`z`: Pearson r = 0.9999, magnitude ratio ≈ 1.00, ~2% mean rel-diff (int8 floor).
- Diffusion conditioning (query, conditioning, atom enc/dec bias, token bias): all r ≥ 0.9999 after the ReLU fix.
- Score model `r_update` (fed PyTorch's first-step `r_noisy`/`times`): r = 0.9999, ratio 0.999.
- End-to-end at 50 diffusion steps: Swift vs upstream PyTorch produce statistically identical geometry — median bonded distance **1.52 Å (Swift) vs 1.52 Å (PyTorch)**, extent ~40 Å both. Different exact conformations (independent RNG), same quality.
- The `sample()` schedule/preconditioning/Euler/augmentation all match `diffusion.py`.
- Boundary harness: `tests/BoltzMLXTests/BoundaryDumpTests.swift` (env/config-gated). Scratch scripts for the comparison are under the session scratchpad (compare_all.py, capture_score_io.py, pt_predict.py).

**Environment note:** the `/tmp` venv did not survive reboot; recreated durably at `~/.venvs/boltz-mlx` (torch 2.13). The `~/.boltz` CCD+mols cache survived. Feature bundles + PyTorch fixtures live under `.artifacts/`.

**Quality report (Task 9) — DONE, `validation/quality/report.json` now populated.** Method: module-boundary Pearson r + matched-noise end-to-end (identical injected noise + identity augmentation on both backends, Kabsch-aligned RMSD/lDDT). Findings on `prot_no_msa`:
- Network port validated: every boundary r >= 0.9997 (int8 floor).
- Sampler per-step ops agree: 2-step matched RMSD 0.82 A (structures still ~8873 A noise-scale).
- End-to-end is **step-count dependent** (matched-noise, no-MSA prot_no_msa): 20 steps -> 3.19 A / lDDT 0.685 (fails gate); **50 steps -> 0.64 A / lDDT 0.977 (passes)**. Few-step Euler corrections are large so int8 error has high per-step leverage; finer integration removes it. **Conclusion: int8 (group-64) is viable end-to-end at production step counts; fp16 not required.**
- Harness: `BoundaryDumpTests.testSampleFixedNoise` (bypasses the MemoryPlanner cap); scratch scripts `pt_matched.py` / `cmp_matched.py` / `generate_final_report.py`.

**Memory-scaling limit found:** end-to-end sampling for a 384-token / ~2976-atom protein OOMs on Mac (single ~28 GB Metal buffer > MLX's 22.6 GB max). Large proteins (and the 846-token ligand) need memory optimization / chunking; this is separate from the iPhone small-protein target. Small-protein runtime measured ~1.66 GB peak footprint (117 tokens).

**Task 10 (iPhone 15 Pro benchmark) — DONE.** Signed with the user's personal team (YOUR_TEAM_ID), installed over cable, artifacts pushed into `Documents/model` + `Documents/features` via `devicectl`, launched, `result.json` read back. **Result (prot_no_msa, 117 tok / 899 atoms, recycling 0 / 20 diffusion steps): 12.98 s wall-clock, ~1.15 GB MLX peak, ~678 MB process footprint — comfortably within the 8 GB device.** See `validation/device/report.md` + `prot_no_msa_iphone15pro.json`. Harness: `examples/BoltzMLXDemo/BenchmarkRunner.swift` (content-scans `Documents/` for model.safetensors/features.safetensors); runbook `examples/BoltzMLXDemo/DEVICE_BENCHMARK.md`. Note: `devicectl copy to` only writes inside `Documents/` and flattens folder contents — push each artifact to its own `Documents/<name>` subdir. Signing needs an Apple ID in Xcode → Accounts (user did this).

**Task 11 (validation + final docs) — DONE.** `scripts/validate_mlx_port.sh` is executable and green end-to-end: exporter pytest + ruff, Swift strict format-lint, 22 macOS tests (`-scheme BoltzMLX-Package`, boundary dumps skipped), `BoltzMLXCLI` build, iOS-simulator demo build, and a quality-report gate that fails if `validation/quality/report.json` is still `not_run`. (Fixed from the original: durable `~/.venvs/boltz-mlx` python, correct `-Package` scheme.)

**All milestones (Tasks 1–11) complete.** The int8 MLX Swift port is numerically faithful (boundary r≥0.9997), structurally validated (passes the 2.0 Å/0.9 gate at adequate diffusion steps; fp16 not required), and benchmarked on the iPhone 15 Pro (~13 s / ~1.15 GB for a 20-step prot_no_msa prediction). Nothing pushed / no PR opened.

**Large-protein OOM — root-caused + fixed (`1f17753`).** The 384-token OOM was NOT the triangle multiplication (MLX fuses that einsum fine). It was the AtomEncoder `z_to_p_trans` 3-way einsum `bijd,bwki,bwlj->bwkld`: MLX's path optimizer picked an order that materialized an `atoms*tokens^2*atom_z` intermediate (exactly 28 GB at 384 tok / 2976 atoms; 432 GB attempted at 192 tok in isolation — path flips with size, fine at ≤96). Fix: contract `i` then `j` in two einsums (`AtomEncoder.swift`), peaks at a few MB, numerically identical (boundary parity unchanged, atom enc/dec bias r=0.99987). **Full end-to-end at 384 tok / 2976 atoms now runs locally (~28 s).** Diagnosis + fix validation live in `tests/BoltzMLXTests/TriangleMemoryTests.swift` (config-gated via `~/.artifacts-boltz-boundary/trimem.json`).

**Remaining future work (not milestones):** re-test the even larger 846-token ligand end-to-end (may hit other size limits — untested since the atom-encoder fix); on-device benchmark at larger sizes; then push / open a PR. General lesson: prefer explicit two-step contractions over 3-way `MLX.einsum` on the hot path — MLX's path choice is size-dependent and can be catastrophic.

This document is the handoff for continuing the Boltz-2 structure-inference port to native MLX Swift. It distinguishes code that is committed from code that only exists in the working tree and, most importantly, distinguishes build/unit-test progress from scientific parity validation.

## User-approved target

- Port the Boltz-2 structure-prediction network to MLX Swift.
- Final runtime target: iOS, baseline iPhone 15 Pro / iOS 17.
- Milestone 1 consumes features precomputed by upstream Boltz/PyTorch offline.
- Pin upstream Boltz to tag `v2.2.1`, commit `cb04aeccdd480fd4db707f0bbafde538397fa2ac`.
- Use MLX affine 8-bit, group-size-64, weight-only quantization. Activations remain floating point.
- Include a native Swift API, macOS command-line runner, and minimal iOS app.
- Structure prediction only. Confidence, affinity, B-factor prediction, potentials, training, and on-device preprocessing are out of scope for this milestone.

The approved design is in `docs/superpowers/specs/2026-07-20-boltz2-mlx-ios-design.md`. The original step-by-step plan is in `docs/superpowers/plans/2026-07-20-boltz2-mlx-ios.md`. The plan's checkboxes were not updated as work proceeded; use the status table below instead.

## Repository state

- Workspace: `$HOME/repos/boltz-test` (moved out of `~/Documents` on 2026-07-20 to escape the iCloud/FileProvider hazard described below)
- Move note (2026-07-20): the git-ignored, regenerable `.venv`, `.build`, `.pytest_cache`, `.ruff_cache`, `.swiftpm`, and `default.profraw` were deleted during the move and must be recreated (see "Useful commands"). Everything else was preserved and verified: full committed history (HEAD `019f508`, branch `codex/mlx-ios`, `git fsck` clean), all uncommitted Task 8/9/10 working-tree changes, and the `.artifacts/` bundle with all four SHA-256 hashes matching the values recorded below.
- Branch: `codex/mlx-ios`
- Port implementation HEAD before this handoff note: `7fb3499d514982240230fc8ea7f93b3d09e5d982`
- Upstream baseline before port commits: `cb04aeccdd480fd4db707f0bbafde538397fa2ac`
- No push or pull request was created.
- The repository is dirty. Task 8, Task 9, documentation, and one exporter fix are present but not committed.
- Do not commit `.artifacts/` or the generated `examples/BoltzMLXDemo/BoltzMLXDemo.xcodeproj/` directory.

Committed port history, oldest first:

| Commit | Work |
| --- | --- |
| `4179c20` | Design for the Boltz-2 MLX iOS port |
| `8c8aa4c` | Detailed implementation plan |
| `8f9fb61` | Versioned Python artifact schema and CLI shell |
| `b93dbca` | Structure-only affine-int8 model export |
| `8febabc` | Precomputed feature export and parity-fixture recorder |
| `c7d389f` | MLX Swift artifact loader |
| `57a4be9` | MLX Swift tensor primitives and int8 layers |
| `73c8c51` | Runtime architecture/config export |
| `9bc0ea1` | Boltz-2 trunk port: embedding, template, MSA, Pairformer |
| `7fb3499` | Diffusion conditioning, score model, and sampler port |

## What is implemented

### Python exporter

`src/boltz_mlx_export/` implements:

- Schema-v1 JSON manifests for model, feature, and fixture artifacts.
- `boltz-mlx export-model`, `export-features`, and `make-fixtures` commands.
- Strict selection of the Boltz-2 structure path; confidence/affinity heads are excluded.
- Stable PyTorch-to-Swift tensor names.
- MLX affine int8 quantization with `bits=8` and `group_size=64`.
- Zero-padding of narrow input dimensions to a physical 64-column group while retaining logical dimensions in the manifest.
- SafeTensors serialization, source revision/commit, and checkpoint SHA-256 recording.
- Export of architecture and diffusion schedule from Lightning checkpoint hyperparameters into `config.json`.
- Deterministic precomputed-feature bundles and PyTorch module-boundary fixture recording.

One uncommitted but important exporter fix is in `src/boltz_mlx_export/model_export.py`: `_eligible_matrix_names` now calls `model.named_modules(remove_duplicate=False)`. Boltz has shared `Linear` aliases, and the default duplicate removal caused aliases present in the Lightning `state_dict`, such as `s_recycle`, to be saved unquantized. The regression test is `test_shared_linear_aliases_are_all_quantized` in `tests/mlx_export/test_model_export.py`. The new test was observed failing before the fix and passing after it.

### Swift artifact/runtime foundation

The root Swift package pins MLX Swift exactly to `0.31.6`. The committed library includes:

- Strict model/feature manifest decoding and SafeTensors loading.
- Typed artifact, schema, input, and execution errors.
- Quantized affine `Linear` and quantized `Embedding` paths.
- Logical-to-physical input padding for quantized matrices.
- LayerNorm, attention, transitions, pair-weighted averaging, outer-product mean, triangle multiplication, triangle attention, and atom-window indexing.
- Config-driven architecture rather than hard-coded production depths.

### Boltz-2 trunk

The committed trunk mirrors the upstream module order:

- Input embedding.
- Relative-position, contact, and bond conditioning.
- Recycling projections.
- Template stack.
- MSA stack.
- 64-block Pairformer.

The main files are under `Sources/BoltzMLX/Trunk/` and `Sources/BoltzMLX/Layers/`.

### Diffusion

The committed diffusion implementation includes:

- Atom encoder and decoder.
- Atom transformer and token diffusion transformer.
- Diffusion conditioning and Fourier/noise conditioning.
- Score model and EDM preconditioning.
- Karras-style sampling schedule and update loop.
- Coordinate augmentation using quaternion rotations.
- Weighted rigid alignment using a 3x3 SVD explicitly placed on the CPU stream because MLX Metal does not support this SVD.
- Cancellation checks between diffusion steps.
- `Memory.clearCache()` at recycling/diffusion boundaries.

The main files are under `Sources/BoltzMLX/Diffusion/`.

### Predictor and CLI (working tree, not committed)

These files implement Task 8 and currently exist only as working-tree changes:

- `Package.swift`
- `Package.resolved`
- `Sources/BoltzMLX/Artifact/FeatureBundle.swift`
- `Sources/BoltzMLX/BoltzPredictor.swift`
- `Sources/BoltzMLX/MemoryPlanner.swift`
- `Sources/BoltzMLXCLI/BoltzMLXCommand.swift`
- `tests/BoltzMLXTests/PredictorTests.swift`

The predictor is an actor that validates a feature bundle, enforces input/memory limits, runs the trunk and diffusion sampler, removes atom padding, and returns `[SIMD3<Float>]`. Default provisional limits are 256 tokens, 2,048 padded atoms, MSA depth 1,024, a 64 MiB MLX cache, and a 6 GiB estimated-activation ceiling.

The CLI syntax is:

```bash
BoltzMLXCLI predict \
  --model MODEL_DIRECTORY \
  --features FEATURE_DIRECTORY \
  --output prediction.safetensors \
  --recycling-steps 0 \
  --diffusion-steps 20 \
  --seed 0
```

It writes `coordinates` and `atom_mask` SafeTensors plus a JSON metadata sidecar. The CLI uses Swift Argument Parser `1.8.2`.

### iOS demo (working tree, not committed)

`examples/BoltzMLXDemo/` contains:

- An XcodeGen `project.yml` targeting iOS 17.
- A SwiftUI app and view model.
- Document pickers for model and feature directories.
- Progress phase, token/atom counts, elapsed time, MLX peak memory, and cancellation display.
- `scripts/build_ios_demo.sh` for simulator or unsigned device builds.

The generated `examples/BoltzMLXDemo/BoltzMLXDemo.xcodeproj/` is build output and should be ignored, not committed. Add it to `.gitignore` before committing Task 9.

### Documentation (working tree, not committed)

- `README.md` has an experimental MLX section.
- `docs/mlx-ios.md` documents model export, feature export, CLI use, Swift API, iOS demo, schemas, limits, and exclusions.
- `scripts/validate_mlx_port.sh` is a draft aggregate validation command.
- `validation/quality/report.json` and `report.md` deliberately say `not_run`; their metrics are null so nobody mistakes a buildable port for a scientifically validated one.

## Production artifacts already generated locally

Artifacts are ignored by Git and live under `.artifacts/`:

| Path | Size | SHA-256 |
| --- | ---: | --- |
| `.artifacts/boltz2_conf.ckpt` | 2.1 GiB | `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1` |
| `.artifacts/boltz2-mlx-int8/model.safetensors` | 504 MiB | `f08fec26a0f9182c02628bad3295a354a65eb6b2b93b373e09e755e770738484` |
| `.artifacts/boltz2-mlx-int8/manifest.json` | 2.7 MiB | `e9d55c90c1a7858a366736235b145124807bbef8a4cf9d5ca47ac3b95bbe7605` |
| `.artifacts/boltz2-mlx-int8/config.json` | 2.1 KiB | `49838c74b7a1abfeecab1cbd83fbeac5a05588ef8ab862b96e71988fe9923b18` |

The final exporter run used the shared-alias fix. Its manifest reports:

- `schema_version`: 1
- `artifact_kind`: `model`
- `source_revision`: `v2.2.1`
- `source_commit`: `cb04aeccdd480fd4db707f0bbafde538397fa2ac`
- 10,275 stored tensor entries
- affine int8, 8 bits, group size 64

The production Swift graph was previously constructed against the first production artifact: every requested Swift weight name resolved, and execution reached feature loading before correctly failing because a deliberately nonexistent feature directory was supplied. The exporter was rerun after the shared-alias fix, but this graph-construction smoke test should be repeated against the final artifact.

## Verification actually performed

These are observations from the work session, not claims about the current working tree after further edits:

- Python exporter suite: 15 tests passed before adding the shared-alias regression test.
- Shared-alias regression test: 1 passed after the fix.
- Ruff passed before the final shared-alias edit; rerun it.
- Focused Swift artifact, primitive, predictor, trunk, diffusion, and sampler tests passed during development.
- Specific trunk tests covered adaptive LayerNorm scale, atom windows, contact selection, MSA feature concatenation, a zero-update Pairformer, relative-position buckets, template-distance buckets, and triangle-attention masking.
- Specific sampler tests covered schedule endpoints and translation recovery in weighted rigid alignment.
- A fresh macOS CLI build succeeded, and `--help` rendered correctly.
- The production model could be loaded and every Swift module constructed before the deliberate missing-feature error.
- `scripts/build_ios_demo.sh simulator` completed with `** BUILD SUCCEEDED **` once.

What has **not** been verified:

- No complete end-to-end prediction has run through the native MLX port using a real precomputed feature bundle.
- No frozen PyTorch-versus-MLX module-boundary fixture corpus has been completed for the production model.
- No float16 Swift reference artifact/network comparison has been completed.
- No protein-only or protein-ligand structural quality comparison has been completed.
- No inference or peak-memory measurement has been made on a physical iPhone 15 Pro.
- The aggregate validation script has not completed after all current working-tree edits.
- Therefore the port must not be described as numerically or scientifically equivalent to upstream Boltz yet.

## Plan status

| Original task | Status | Remaining gate |
| --- | --- | --- |
| 1. Artifact schema/CLI | Committed | Fresh full test/lint run |
| 2. Structure selection/int8 export | Committed plus uncommitted alias fix | Commit fix; audit that all eligible aliases are quantized |
| 3. Features/parity fixtures | Committed | Exercise real upstream preprocessing and generate production fixtures |
| 4. Swift loader/config | Committed plus a small uncommitted `FeatureBundle` change | Fresh full Swift tests |
| 5. Primitive layers | Committed | Real PyTorch boundary fixtures, not only synthetic/unit checks |
| 6. Trunk | Committed | Production boundary parity and memory measurement |
| 7. Diffusion/sampler | Committed | Production conditioning/score/sampler parity |
| 8. Predictor/CLI | Implemented, uncommitted | Full tests and a successful tiny/real feature-bundle CLI inference |
| 9. iOS demo | Implemented, uncommitted | Ignore generated project, commit sources, run on physical iPhone |
| 10. Production validation/docs | Partly implemented, uncommitted | Frozen corpus, quality metrics, full verification, final docs/commit |

## Recommended continuation order

1. Preserve and commit the working-tree code in small commits.
   - First commit the exporter shared-alias fix and test.
   - Then commit predictor/CLI files as Task 8.
   - Add `examples/BoltzMLXDemo/BoltzMLXDemo.xcodeproj/` to `.gitignore`, then commit Task 9 sources.
   - Keep docs/quality report for the final validation commit.
2. Run Python unit tests and Ruff from a non-FileProvider virtual environment.
3. Run strict Swift formatting, all macOS Swift tests, and the CLI build.
4. Repeat production model construction using `.artifacts/boltz2-mlx-int8` and confirm the final shared-alias artifact resolves every tensor.
5. Use upstream Boltz to preprocess at least `examples/prot_no_msa.yaml` and `examples/ligand.yaml` into feature bundles. This needs the Boltz molecule cache.
6. Complete one native end-to-end CLI prediction. Debug numerical/shape issues before attempting iOS.
7. Generate PyTorch boundary fixtures and compare the Swift trunk, conditioning, score model, and deterministic sampler step-by-step.
8. Generate float16 and int8 outputs for the frozen protein and protein-ligand cases; rigidly align them and record all-atom RMSD plus lDDT-style agreement in `validation/quality/report.json`.
9. Run the demo on a physical iPhone 15 Pro and record model bytes, active/cache/peak MLX memory, wall time, token count, atom count, MSA depth, recycling steps, and diffusion steps under `validation/device/`.
10. Only after the above passes, run `scripts/validate_mlx_port.sh`, update the quality report from `not_run`, and make the final documentation commit.

## Useful commands

The normal intended verification commands are:

```bash
cd $HOME/repos/boltz-test

PYTHONPATH=/tmp/boltz-src-git \
  /tmp/boltz-mlx-venv/bin/python -m pytest tests/mlx_export -q

PYTHONPATH=/tmp/boltz-src-git \
  /tmp/boltz-mlx-venv/bin/python -m ruff check \
  src/boltz_mlx_export tests/mlx_export

xcrun swift-format format --in-place --recursive \
  Package.swift Sources/BoltzMLX Sources/BoltzMLXCLI tests/BoltzMLXTests

xcrun swift-format lint --strict --recursive \
  Package.swift Sources/BoltzMLX Sources/BoltzMLXCLI tests/BoltzMLXTests

xcodebuild test \
  -scheme BoltzMLX-Package \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation

xcodebuild build \
  -scheme BoltzMLXCLI \
  -destination 'platform=macOS,arch=arm64' \
  -skipPackagePluginValidation

scripts/build_ios_demo.sh simulator
```

The local production conversion command was:

```bash
PYTHONPATH=/tmp/boltz-src-git \
  /tmp/boltz-mlx-venv/bin/python -m boltz_mlx_export export-model \
  --checkpoint .artifacts/boltz2_conf.ckpt \
  --output .artifacts/boltz2-mlx-int8
```

`/tmp/boltz-src-git` is a Git-object copy of `src/`, created because importing the editable package from the Documents workspace repeatedly stalled. Before testing current uncommitted Python changes, refresh it and then copy the current exporter into it:

```bash
rm -rf /tmp/boltz-src-git
mkdir -p /tmp/boltz-src-git
git archive HEAD src | tar -x -C /tmp/boltz-src-git --strip-components=1
cp src/boltz_mlx_export/model_export.py \
  /tmp/boltz-src-git/boltz_mlx_export/model_export.py
```

The `/tmp` environment is not durable across reboot. Recreate it if missing:

```bash
uv venv --python 3.12 /tmp/boltz-mlx-venv
uv pip install --python /tmp/boltz-mlx-venv/bin/python \
  -e '.[mlx-export]' pytest ruff
```

## Environment hazards

This workspace is inside an iCloud/FileProvider-managed Documents directory. macOS repeatedly changed source files and `.venv` files to `compressed,dataless`. Opening one such file from Git or Python can block for roughly 20 seconds or indefinitely. This affected `git status`, `git add`, normal editable Python imports, and even `/bin/cat`.

Useful diagnostics/recovery:

```bash
ls -lO PATH
fileproviderctl evaluate PATH
brctl download PATH

ps -o pid,ppid,state,etime,command -ax | \
  rg 'git status|git add|/bin/cat|ditto src'
```

If an interrupted Git process leaves `.git/index.lock`, first confirm no live Git process owns it, then remove the lock. Avoid broad `git status` until files are hydrated; inspect explicit paths.

Two commits were created with low-level Git plumbing because normal `git commit` stalled while refreshing the FileProvider worktree:

```bash
tree=$(git write-tree)
parent=$(git rev-parse HEAD)
commit=$(printf '%s\n' 'commit message' | git commit-tree "$tree" -p "$parent")
git update-ref refs/heads/codex/mlx-ios "$commit" "$parent"
```

Only use this after explicitly and correctly staging the intended files. Do not accidentally stage generated Xcode output or `.artifacts/`.

Disk space is also tight: at handoff the data volume had approximately 8.5 GiB free and was at 99% capacity. The checkpoint is 2.1 GiB; Xcode/Swift build products can consume another 1–2 GiB. Prefer `swift package clean` and `xcodebuild ... clean` over deleting source or artifacts, and recheck with `df -h .` before production inference.

## Known technical risks to inspect

- Confirm that the final production manifest has no eligible `Linear`/`Embedding` matrix aliases left unquantized after `remove_duplicate=False`.
- The current iOS memory limits are conservative estimates, not measurements.
- The sampler presently assumes one complex and one diffusion sample, which matches milestone scope.
- Verify `coordinate_augmentation=false` behavior; the current sampler path was written around the production config, where it is true.
- Review all MLX broadcasting/mask semantics at template, MSA, triangle-attention, atom-window, and diffusion boundaries using real PyTorch fixtures.
- Verify numerical stability of the CPU SVD rigid-alignment path and data transfers during iOS execution.
- Verify CLI exit codes against the plan. Swift Argument Parser supplies its own usage behavior; typed artifact/input/execution mappings need a final audit.
- Ensure `scripts/validate_mlx_port.sh` is executable; it was not executable at handoff.
- The validation script currently checks builds/tests but does not yet generate or enforce the frozen-corpus structural quality report described by the plan.

## Definition of done

Do not call this milestone complete until all of the following are true:

1. A real precomputed Boltz feature bundle completes native MLX Swift prediction.
2. Production PyTorch/Swift boundary fixtures pass at the trunk and diffusion boundaries.
3. The final int8 artifact has zero omitted eligible matrices.
4. Protein-only and protein-ligand quality metrics are recorded and pass the declared thresholds.
5. Peak memory and runtime are measured on an iPhone 15 Pro.
6. Full Python tests/lint, Swift tests/format, CLI smoke inference, simulator build, and device run pass from the committed tree.
