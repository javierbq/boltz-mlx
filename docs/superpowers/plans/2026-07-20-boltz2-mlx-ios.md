# Boltz-2 MLX iOS Structure Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native MLX Swift, int8 weight-only Boltz-2 structure predictor for iPhone 15 Pro/iOS 17 that consumes offline-precomputed feature bundles.

**Architecture:** A Python exporter pinned to Boltz `v2.2.1` produces versioned SafeTensors model, feature, and parity artifacts. A Swift package mirrors the upstream trunk and diffusion module boundaries, loads either float16 or affine-int8 weights, and exposes an actor-based predictor used by a macOS CLI and an iOS demo.

**Tech Stack:** Python 3.12, PyTorch/Boltz 2.2.1, MLX Python 0.31.2 or newer for offline affine quantization, SafeTensors, Swift 6, MLX Swift 0.31.6, XCTest, Xcode 26.

## Global Constraints

- Freeze upstream architecture at tag `v2.2.1`, commit `cb04aeccdd480fd4db707f0bbafde538397fa2ac`.
- Pin MLX Swift exactly to `0.31.6`.
- Deploy to iOS 17 or newer; validate on iPhone 15 Pro/A17 Pro.
- Run learned inference only in MLX Swift; Python/PyTorch are offline tools.
- Quantize all eligible Linear and Embedding matrices using affine `bits=8`, `group_size=64`; zero-pad incompatible input widths and record logical/physical dimensions rather than silently skipping them.
- Keep confidence, affinity, B-factor, potentials, training, and on-device preprocessing out of scope.
- Process one complex and one diffusion sample at a time on iOS.

---

### Task 1: Versioned artifact schema and Python CLI shell

**Files:**
- Create: `src/boltz_mlx_export/__init__.py`
- Create: `src/boltz_mlx_export/__main__.py`
- Create: `src/boltz_mlx_export/cli.py`
- Create: `src/boltz_mlx_export/schema.py`
- Modify: `pyproject.toml`
- Test: `tests/mlx_export/test_schema.py`
- Test: `tests/mlx_export/test_cli.py`

**Interfaces:**
- Produces: `ArtifactManifest`, `TensorSpec`, `FeatureMetadata`, `BoltzMLXExportError`, and the `boltz-mlx` console script.
- Manifest JSON uses sorted keys and schema version `1`.

- [ ] **Step 1: Write failing schema round-trip and CLI help tests**

```python
def test_manifest_round_trip(tmp_path):
    manifest = ArtifactManifest.model_v1(source_checkpoint_sha256="a" * 64)
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert ArtifactManifest.read(path) == manifest

def test_cli_lists_export_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "export-model" in result.output
    assert "export-features" in result.output
    assert "make-fixtures" in result.output
```

- [ ] **Step 2: Run tests and verify imports fail**

Run: `python3.12 -m pytest tests/mlx_export/test_schema.py tests/mlx_export/test_cli.py -q`

Expected: collection fails because `boltz_mlx_export` does not exist.

- [ ] **Step 3: Implement immutable dataclasses and Click command group**

```python
@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    artifact_kind: Literal["model", "features", "fixture"]
    source_revision: str
    source_checkpoint_sha256: str | None
    tensors: tuple[TensorSpec, ...]

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
```

Register `boltz-mlx = "boltz_mlx_export.cli:cli"` in `[project.scripts]` and include `mlx>=0.31.2`, `safetensors>=0.5.3`, and `platformdirs>=4.3.8` in a new `mlx-export` optional dependency.

- [ ] **Step 4: Run tests and lint**

Run: `python3.12 -m pytest tests/mlx_export/test_schema.py tests/mlx_export/test_cli.py -q`

Expected: all tests pass.

Run: `python3.12 -m ruff check src/boltz_mlx_export tests/mlx_export`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/boltz_mlx_export tests/mlx_export
git commit -m "feat: add Boltz MLX artifact schema"
```

### Task 2: Structure-only checkpoint selection and affine-int8 export

**Files:**
- Create: `src/boltz_mlx_export/model_export.py`
- Create: `src/boltz_mlx_export/names.py`
- Create: `src/boltz_mlx_export/quantization.py`
- Test: `tests/mlx_export/test_model_export.py`
- Test: `tests/mlx_export/test_quantization.py`

**Interfaces:**
- Consumes: Boltz `Boltz2.state_dict()` and Task 1 schema.
- Produces: `select_structure_state_dict(state)`, `swift_tensor_name(torch_name)`, `quantize_affine_int8(weight, group_size=64)`, and sharded `model.safetensors` files.
- Structure prefixes are exactly `input_embedder`, `s_init`, `z_init_1`, `z_init_2`, `rel_pos`, `token_bonds`, `token_bonds_type`, `contact_conditioning`, `s_norm`, `z_norm`, `s_recycle`, `z_recycle`, `template_module`, `msa_module`, `pairformer_module`, `diffusion_conditioning`, and `structure_module`.

- [ ] **Step 1: Write failing selection and quantization tests**

```python
def test_structure_selection_excludes_confidence():
    state = {
        "s_init.weight": torch.ones(4, 4),
        "structure_module.score_model.out.weight": torch.ones(4, 4),
        "confidence_module.head.weight": torch.ones(4, 4),
    }
    selected = select_structure_state_dict(state)
    assert set(selected) == {"s_init.weight", "structure_module.score_model.out.weight"}

def test_short_linear_is_zero_padded():
    quantized = quantize_affine_int8(np.ones((8, 1), dtype=np.float16))
    assert quantized.logical_input_width == 1
    assert quantized.physical_input_width == 64
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python3.12 -m pytest tests/mlx_export/test_model_export.py tests/mlx_export/test_quantization.py -q`

Expected: imports fail for missing exporter functions.

- [ ] **Step 3: Implement strict selection, naming, sharding, hashing, and quantization**

Pad eligible matrices with zero columns to a multiple of 64, then use `mlx.core.quantize(mx.array(weight), group_size=64, bits=8, mode="affine")`. For a stable name such as `trunk.s_init`, save packed weight as `trunk.s_init.weight`, scales as `trunk.s_init.scales`, affine biases as `trunk.s_init.biases`, and an optional original Linear bias as `trunk.s_init.linear_bias`. Store logical and physical input widths in the manifest and keep unquantized non-matrix parameters under their stable Swift names.

- [ ] **Step 4: Add a synthetic checkpoint integration test**

The test writes a state dictionary containing one eligible Linear, one LayerNorm, and one excluded confidence key, invokes `export_state_dict`, reloads SafeTensors, and asserts packed shapes, tensor names, dtypes, checksums, and exclusion behavior.

- [ ] **Step 5: Run tests and lint**

Run: `python3.12 -m pytest tests/mlx_export/test_model_export.py tests/mlx_export/test_quantization.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/boltz_mlx_export tests/mlx_export
git commit -m "feat: export Boltz structure weights as int8 MLX artifacts"
```

### Task 3: Offline feature bundles and PyTorch parity fixtures

**Files:**
- Create: `src/boltz_mlx_export/feature_export.py`
- Create: `src/boltz_mlx_export/fixtures.py`
- Create: `src/boltz_mlx_export/tensor_io.py`
- Test: `tests/mlx_export/test_feature_export.py`
- Test: `tests/mlx_export/test_fixtures.py`

**Interfaces:**
- Produces: `export_feature_batch(feats, directory)`, `load_feature_bundle(directory)`, and `FixtureRecorder` forward hooks.
- Feature arrays are detached, moved to CPU, made contiguous, and saved without pickle.

- [ ] **Step 1: Write failing deterministic export tests**

```python
def test_feature_bundle_is_deterministic(tmp_path):
    feats = {"token_pad_mask": torch.tensor([[1, 1]]), "atom_pad_mask": torch.tensor([[1]])}
    export_feature_batch(feats, tmp_path / "a")
    export_feature_batch(feats, tmp_path / "b")
    assert file_sha256(tmp_path / "a/features.safetensors") == file_sha256(
        tmp_path / "b/features.safetensors"
    )
```

- [ ] **Step 2: Implement dtype normalization, metadata, and fixture hooks**

Preserve float32/float16/bfloat16, map PyTorch `long` to int64, and preserve bool. Fixture records include input tensors, output tensors, `atol`, `rtol`, source module name, and a monotonically increasing call index so recycled modules remain distinguishable.

- [ ] **Step 3: Connect CLI commands to model/feature/fixture exporters**

`export-features` invokes the pinned Boltz preprocessing pipeline, obtains the single-batch feature dictionary, and writes it. `make-fixtures` uses a user-supplied feature bundle and checkpoint, injects exported noise, disables custom kernels, and records module boundaries.

- [ ] **Step 4: Run tests and commit**

Run: `python3.12 -m pytest tests/mlx_export -q`

Expected: all exporter tests pass without downloading a production checkpoint.

```bash
git add src/boltz_mlx_export tests/mlx_export
git commit -m "feat: export Boltz features and parity fixtures"
```

### Task 4: Swift package, errors, configuration, and SafeTensors loader

**Files:**
- Create: `Package.swift`
- Create: `Sources/BoltzMLX/Artifact/BoltzArtifact.swift`
- Create: `Sources/BoltzMLX/Artifact/ArtifactManifest.swift`
- Create: `Sources/BoltzMLX/Artifact/FeatureBundle.swift`
- Create: `Sources/BoltzMLX/BoltzError.swift`
- Create: `Sources/BoltzMLX/BoltzTypes.swift`
- Test: `tests/BoltzMLXTests/ArtifactTests.swift`
- Fixture: `tests/BoltzMLXTests/Fixtures/artifact-v1/`

**Interfaces:**
- Produces: `BoltzArtifact.load(from:)`, `FeatureBundle.load(from:)`, `BoltzPredictionOptions`, `BoltzInputLimits`, `BoltzStructure`, and typed `BoltzError`.
- Pins `.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6")` and links `MLX` and `MLXNN`.

- [ ] **Step 1: Add a failing artifact-loader XCTest**

```swift
func testRejectsUnknownSchema() async throws {
    let url = try fixtureURL("unknown-schema")
    await XCTAssertThrowsErrorAsync(try await BoltzArtifact.load(from: url)) { error in
        XCTAssertEqual(error as? BoltzError, .unsupportedSchema(found: 99, supported: 1))
    }
}
```

- [ ] **Step 2: Implement Codable manifests and strict tensor validation**

Use MLX `loadArrays(url:)`/`loadArraysAndMetadata(url:)`, compare every loaded array against its declared name, shape, and dtype, and reject undeclared tensors.

- [ ] **Step 3: Resolve and run Swift tests**

Run: `xcodebuild test -scheme BoltzMLX -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation`

Expected: artifact tests pass and MLX Metal resources link.

- [ ] **Step 4: Commit**

```bash
git add Package.swift Package.resolved Sources/BoltzMLX tests/BoltzMLXTests
git commit -m "feat: add MLX Swift artifact loader"
```

### Task 5: MLX Swift primitive layers

**Files:**
- Create: `Sources/BoltzMLX/Layers/AffineLinear.swift`
- Create: `Sources/BoltzMLX/Layers/AttentionPairBias.swift`
- Create: `Sources/BoltzMLX/Layers/Transition.swift`
- Create: `Sources/BoltzMLX/Layers/TriangleMultiplication.swift`
- Create: `Sources/BoltzMLX/Layers/OuterProductMean.swift`
- Create: `Sources/BoltzMLX/Layers/PairWeightedAveraging.swift`
- Create: `Sources/BoltzMLX/Layers/TensorOps.swift`
- Test: `tests/BoltzMLXTests/PrimitiveParityTests.swift`
- Fixture: `tests/BoltzMLXTests/Fixtures/primitives/`

**Interfaces:**
- `AffineLinear` loads packed weights/scales/biases and calls `quantizedMM(x, packedWeight, scales: scales, biases: affineBiases, transpose: true, groupSize: 64, bits: 8, mode: .affine)` before adding the optional Linear bias.
- Tensor conventions match PyTorch fixtures: batch-first and feature-last.

- [ ] **Step 1: Generate small deterministic PyTorch fixtures**

Run: `python3.12 -m boltz_mlx_export make-fixtures --checkpoint "$BOLTZ_CHECKPOINT" --features "$BOLTZ_FEATURES" --output tests/BoltzMLXTests/Fixtures/primitives`

Expected: SafeTensors and metadata appear for every primitive.

- [ ] **Step 2: Add failing float and quantized parity tests**

Each test loads fixture inputs, executes one Swift layer, calls `eval(output)`, and compares flattened float values with the fixture's `atol` and `rtol`.

- [ ] **Step 3: Implement primitives using MLX operations**

Port equations directly from the pinned PyTorch files. Keep einsum labels and reshape/transpose steps adjacent to comments containing the expected shapes. Evaluate chunked outer-product and transition outputs at chunk boundaries.

- [ ] **Step 4: Run parity tests and commit**

Run: `xcodebuild test -scheme BoltzMLX -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -only-testing:BoltzMLXTests/PrimitiveParityTests`

Expected: all float fixtures pass; quantized fixtures meet their declared tolerances.

```bash
git add Sources/BoltzMLX/Layers tests/BoltzMLXTests
git commit -m "feat: port Boltz tensor primitives to MLX Swift"
```

### Task 6: Input embedding, MSA, templates, and Pairformer trunk

**Files:**
- Create: `Sources/BoltzMLX/Trunk/InputEmbedder.swift`
- Create: `Sources/BoltzMLX/Trunk/RelativePositionEncoder.swift`
- Create: `Sources/BoltzMLX/Trunk/ContactConditioning.swift`
- Create: `Sources/BoltzMLX/Trunk/TemplateModule.swift`
- Create: `Sources/BoltzMLX/Trunk/MSAModule.swift`
- Create: `Sources/BoltzMLX/Trunk/Pairformer.swift`
- Create: `Sources/BoltzMLX/Trunk/BoltzTrunk.swift`
- Test: `tests/BoltzMLXTests/TrunkParityTests.swift`

**Interfaces:**
- `BoltzTrunk.callAsFunction(features:recyclingSteps:) -> TrunkOutput` returns `s`, `z`, `sInputs`, and relative position encoding.
- Consumes Task 5 primitives and Task 4 feature bundle.

- [ ] **Step 1: Record module-boundary fixtures from PyTorch**

Record embedding outputs, template contribution, MSA contribution, every Pairformer block, and each recycling result for a tiny padded feature bundle.

- [ ] **Step 2: Write failing boundary parity tests**

Use zero recycling and one recycling step fixtures. Assert shapes before numeric comparisons so layout mistakes are localized.

- [ ] **Step 3: Port the trunk modules in upstream execution order**

Do not port `torch.compile` or custom CUDA kernel branches. Use the upstream non-kernel equations. Call `eval` and `Memory.clearCache()` only at explicit recycling boundaries.

- [ ] **Step 4: Run tests, measure peak memory, and commit**

Run: `xcodebuild test -scheme BoltzMLX -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -only-testing:BoltzMLXTests/TrunkParityTests`

Expected: every recorded boundary passes its fixture tolerance.

```bash
git add Sources/BoltzMLX/Trunk tests/BoltzMLXTests
git commit -m "feat: port the Boltz-2 trunk to MLX Swift"
```

### Task 7: Diffusion conditioning, score model, and sampler

**Files:**
- Create: `Sources/BoltzMLX/Diffusion/DiffusionConditioning.swift`
- Create: `Sources/BoltzMLX/Diffusion/AtomEncoder.swift`
- Create: `Sources/BoltzMLX/Diffusion/AtomTransformer.swift`
- Create: `Sources/BoltzMLX/Diffusion/DiffusionTransformer.swift`
- Create: `Sources/BoltzMLX/Diffusion/AtomDecoder.swift`
- Create: `Sources/BoltzMLX/Diffusion/DiffusionScoreModel.swift`
- Create: `Sources/BoltzMLX/Diffusion/AtomDiffusion.swift`
- Test: `tests/BoltzMLXTests/DiffusionParityTests.swift`
- Test: `tests/BoltzMLXTests/SamplerParityTests.swift`

**Interfaces:**
- `DiffusionConditioning` returns the six arrays consumed by the score model.
- `AtomDiffusion.sample(trunk:inputs:features:diffusionSteps:noise:cancellationCheck:) async throws -> MLXArray` accepts optional injected noise for parity tests and generated MLX noise for production.
- Cancellation is checked between diffusion steps.

- [ ] **Step 1: Export conditioning, score, schedule, and sampler-step fixtures**

Fixtures include window-index arrays and externally supplied initial/step noise so stochastic differences cannot mask equation errors.

- [ ] **Step 2: Add failing diffusion boundary tests**

Test Fourier embedding, atom encoder/decoder window gathering, conditioned transitions, one score evaluation, one sampler update, and a complete three-step deterministic sample.

- [ ] **Step 3: Port non-kernel diffusion equations**

Preserve upstream float32 regions for coordinates, noise scaling, and schedule arithmetic. Use float16 activations elsewhere. Evaluate after each diffusion step, check cancellation, and clear cache according to runtime configuration.

- [ ] **Step 4: Run parity tests and commit**

Run: `xcodebuild test -scheme BoltzMLX -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation -only-testing:BoltzMLXTests/DiffusionParityTests -only-testing:BoltzMLXTests/SamplerParityTests`

Expected: all deterministic fixture comparisons pass.

```bash
git add Sources/BoltzMLX/Diffusion tests/BoltzMLXTests
git commit -m "feat: port Boltz-2 diffusion sampling to MLX Swift"
```

### Task 8: Predictor actor and macOS CLI

**Files:**
- Create: `Sources/BoltzMLX/BoltzPredictor.swift`
- Create: `Sources/BoltzMLX/MemoryPlanner.swift`
- Create: `Sources/BoltzMLXCLI/main.swift`
- Test: `tests/BoltzMLXTests/PredictorTests.swift`
- Modify: `Package.swift`

**Interfaces:**
- Implements the public API in the design spec.
- CLI syntax: `BoltzMLXCLI predict --model <dir> --features <dir> --output <file> [--recycling-steps N] [--diffusion-steps N] [--seed N]`.

- [ ] **Step 1: Write failing predictor validation tests**

Test token/atom/MSA limits, estimated-memory rejection, cancellation, deterministic same-seed results, and output atom-mask unpadding.

- [ ] **Step 2: Implement memory planning and predictor orchestration**

Set `Memory.cacheLimit` and `Memory.memoryLimit` before allocation, validate dimensions, execute trunk/conditioning/sample, and serialize coordinates with metadata.

- [ ] **Step 3: Implement CLI argument parsing and typed exit codes**

Exit `2` for usage, `3` for artifact/schema errors, `4` for input limits, and `5` for execution failures.

- [ ] **Step 4: Run tests and a tiny CLI fixture**

Run: `xcodebuild test -scheme BoltzMLX -destination 'platform=macOS,arch=arm64' -skipPackagePluginValidation`

Run: `swift run BoltzMLXCLI predict --model tests/BoltzMLXTests/Fixtures/model-tiny --features tests/BoltzMLXTests/Fixtures/features-tiny --output /tmp/boltz-mlx-prediction.safetensors`

Expected: command exits zero and writes coordinates plus metadata.

- [ ] **Step 5: Commit**

```bash
git add Package.swift Sources/BoltzMLX Sources/BoltzMLXCLI tests/BoltzMLXTests
git commit -m "feat: add native Boltz MLX predictor and CLI"
```

### Task 9: iOS demo and device memory instrumentation

**Files:**
- Create: `examples/BoltzMLXDemo/README.md`
- Create: `examples/BoltzMLXDemo/project.yml`
- Create: `examples/BoltzMLXDemo/BoltzMLXDemoApp.swift`
- Create: `examples/BoltzMLXDemo/PredictionView.swift`
- Create: `examples/BoltzMLXDemo/PredictionViewModel.swift`
- Create: `examples/BoltzMLXDemo/Resources/README.md`
- Create: `scripts/build_ios_demo.sh`

**Interfaces:**
- Demo loads user-imported model/features directories or bundled tiny fixtures.
- Shows phase, elapsed time, token/atom counts, MLX peak memory, and cancellation.

- [ ] **Step 1: Add the minimal SwiftUI app and generate the Xcode project**

Use XcodeGen only to generate the project; commit `project.yml`, not user-specific Xcode state. Link the local `BoltzMLX` package and target iOS 17.

- [ ] **Step 2: Build for simulator**

Run: `scripts/build_ios_demo.sh simulator`

Expected: `xcodebuild` succeeds for an available iPhone simulator destination.

- [ ] **Step 3: Run a device benchmark when an iPhone 15 Pro is connected**

Record model bytes, active/cache/peak MLX memory, wall time, token count, atom count, MSA depth, recycling steps, and diffusion steps as JSON under `validation/device/`.

- [ ] **Step 4: Commit**

```bash
git add examples/BoltzMLXDemo scripts/build_ios_demo.sh validation/device
git commit -m "feat: add Boltz MLX iOS demo"
```

### Task 10: Production conversion, quality report, and documentation

**Files:**
- Create: `docs/mlx-ios.md`
- Create: `validation/quality/report.json`
- Create: `validation/quality/report.md`
- Create: `scripts/validate_mlx_port.sh`
- Modify: `README.md`

**Interfaces:**
- Validation script runs Python unit tests, Swift parity tests, production artifact checks, CLI smoke inference, and report generation.

- [ ] **Step 1: Convert the production checkpoint**

Run: `boltz-mlx export-model --checkpoint "$BOLTZ_CACHE/boltz2_conf.ckpt" --output "$BOLTZ_MLX_MODEL"`

Expected: exporter reports zero omitted eligible matrices and writes a checksum-verified int8 artifact.

- [ ] **Step 2: Export the frozen protein and protein-ligand corpus**

Use `examples/prot_no_msa.yaml` and `examples/ligand.yaml`, with fixed MSA/template inputs and fixed sampling noise, to produce feature bundles and upstream reference outputs.

- [ ] **Step 3: Run float16 and int8 quality comparisons**

Rigidly align predicted structures, calculate all-atom RMSD and lDDT-style agreement, and store exact thresholds and measured values in `report.json`. Fail validation if either artifact exceeds its declared threshold.

- [ ] **Step 4: Document installation, export, CLI, iOS integration, schemas, limits, and exclusions**

Include exact commands, artifact sizes, measured iPhone limits, and the scientific-validation caveat from the design.

- [ ] **Step 5: Run full verification**

Run: `scripts/validate_mlx_port.sh`

Expected: Python tests, Swift tests, simulator build, artifact validation, CLI smoke test, and quality thresholds all pass.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/mlx-ios.md scripts/validate_mlx_port.sh validation/quality
git commit -m "docs: validate and document Boltz MLX iOS inference"
```
