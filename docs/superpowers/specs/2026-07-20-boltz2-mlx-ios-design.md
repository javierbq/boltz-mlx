# Boltz-2 MLX iOS Structure Inference Design

## Goal

Port the structure-prediction path of Boltz `v2.2.1` to MLX Swift and run it on an iPhone 15 Pro or newer with iOS 17 or newer. Model weights are stored with MLX 8-bit affine weight-only quantization. Activations remain floating point. The first milestone consumes features generated offline by the upstream Boltz Python pipeline.

The implementation is inference-only. Training, confidence prediction, affinity prediction, B-factor prediction, inference-time potentials, and on-device YAML/MSA/ligand preprocessing are outside this milestone.

## Chosen approach

Implement the network directly in MLX Swift, backed by a Python export and parity harness.

This is preferred over two alternatives:

1. A Python MLX port followed by a separate Swift rewrite would provide a convenient intermediate runtime, but it would create two ports whose behavior could diverge.
2. Core ML conversion would simplify application integration, but Boltz's dynamic recycling, diffusion loop, atom-window attention, pairwise tensors, and triangular operations do not map reliably to a single converted graph. Debugging conversion failures would also be harder than comparing explicit MLX layers.

The direct Swift approach gives the iOS runtime the same module boundaries as the PyTorch source and makes every boundary independently testable against exported fixtures.

## Frozen references

- Boltz source and architecture: tag `v2.2.1`, commit `cb04aeccdd480fd4db707f0bbafde538397fa2ac`.
- Default structure weights: the structure-related parameters from `boltz2_conf.ckpt` published for Boltz-2.
- MLX Swift: exact package version `0.31.6`.
- Deployment floor: iOS 17.
- Baseline device: iPhone 15 Pro (A17 Pro).

The exporter records these identifiers in the artifact manifest and refuses an unsupported Boltz checkpoint schema.

## Deliverables

The repository will contain:

- `src/boltz_mlx_export`: Python code that loads the pinned PyTorch checkpoint, exports structure-only weights, exports precomputed features, and creates deterministic parity fixtures.
- `Package.swift` and `Sources/BoltzMLX`: a native Swift package containing the MLX implementation.
- `Sources/BoltzMLXCLI`: a macOS command-line runner for the same Swift inference library.
- `Tests/BoltzMLXTests`: layer, artifact-loading, and deterministic sampling parity tests.
- `examples/BoltzMLXDemo`: a minimal iOS sample target that loads bundled artifacts and runs a prediction asynchronously.

The original `boltz predict` command remains the offline parser/featurizer. The exporter adds a `boltz-mlx` Python command with these operations:

```text
boltz-mlx export-model --checkpoint boltz2_conf.ckpt --output Model/
boltz-mlx export-features input.yaml --output Features/
boltz-mlx make-fixtures --checkpoint boltz2_conf.ckpt --output Tests/Fixtures/
```

The native macOS executable accepts the same artifacts:

```text
BoltzMLXCLI predict --model Model/ --features Features/ --output prediction.safetensors
```

## Runtime architecture

### Artifact loader

`BoltzArtifact` loads `config.json`, `manifest.json`, and sharded SafeTensors files. It validates format version, model revision, tensor names, shapes, and quantization metadata before constructing modules. Files are loaded from the application bundle or an application-controlled directory; the package performs no network downloads.

### Feature contract

Offline preprocessing emits a feature bundle rather than Python objects. `features.safetensors` contains every tensor read by the structure path, with stable names, explicit dtypes, and batch dimension one. `metadata.json` contains schema version, source input identity, token and atom counts, atom annotations required to interpret returned coordinates, and deterministic sampling settings.

The iOS runtime supports one complex per invocation. It rejects unknown schema versions, missing tensors, invalid ranks, inconsistent token/atom dimensions, and inputs above configured limits before allocating pairwise tensors.

### Network modules

The Swift package mirrors the upstream structure path:

1. Input embedding and relative/contact/bond features.
2. Recycling loop with optional template conditioning, MSA stack, and Pairformer.
3. Diffusion conditioning.
4. Atom diffusion score network and deterministic sampler.
5. Coordinate unpadding and result packaging.

Confidence, affinity, distogram-only output, and B-factor heads are not instantiated or exported.

Shared primitives—linear/quantized linear, layer normalization, attention with pair bias, transitions, pair-weighted averaging, outer-product mean, triangular multiplication, and atom-window indexing—live in focused files and are used by the trunk and diffusion modules.

### Sampling

Sampling follows the upstream `AtomDiffusion.sample` schedule and update equations. A caller supplies a `UInt64` seed, number of recycling steps, number of diffusion steps, and sample count. The milestone supports one sample at a time on iOS to bound peak memory; multiple samples are run sequentially by the application.

Random values are exported as fixtures for numerical unit tests because PyTorch and MLX random generators are not expected to produce identical streams from the same seed. End-to-end quality tests compare aligned structures rather than raw stochastic coordinates.

## Weight conversion and int8 quantization

The Python exporter performs these steps:

1. Load the pinned checkpoint on CPU using the upstream Boltz class and inference configuration.
2. Select only parameters required by the structure path.
3. Rename PyTorch keys to the stable Swift module schema and transpose matrices where MLX layout requires it.
4. Save a float16 reference artifact.
5. Zero-pad matrix input dimensions to a multiple of 64 where required, recording logical and physical dimensions in the manifest.
6. Quantize every eligible Linear and Embedding weight using MLX affine quantization with `bits=8` and `group_size=64`.
6. Store packed weights, scales, biases, original shapes, and per-tensor quantization settings in sharded SafeTensors files.

Non-matrix parameters, normalization parameters, scalar schedules, and coordinate constants remain float16 or float32 as appropriate. This is still an int8 weight-only model: every operation supported by MLX quantized matrix multiplication uses int8-packed weights, while parameters for operations that are not matrix multiplication retain their required dtype.

The Swift quantized layer pads its input to the recorded physical dimension before matrix multiplication. This allows narrow learned projections, including the one-channel bond projection, to remain int8. The exporter never silently leaves an eligible layer unquantized.

## Numerical and quality validation

Validation proceeds from small deterministic units to the stochastic full model:

- Primitive and module fixtures contain PyTorch inputs and outputs for float32 computation. MLX Swift output must meet a per-fixture absolute/relative tolerance declared in the fixture metadata.
- The unquantized float16 Swift network is compared at input embedding, each trunk stage, diffusion conditioning, and score-network boundaries.
- Quantized tests compare dequantized weights and module outputs against the float16 Swift reference.
- End-to-end deterministic tests inject the same exported initial noise and compare final coordinates after rigid alignment.
- Quantized quality tests use a small frozen corpus containing a protein-only input and a protein-ligand input. They report backbone/atom RMSD and lDDT-style agreement against the unquantized Swift reference and the upstream PyTorch output.

No claim of scientific equivalence is made merely because the code builds. The artifact is marked `validated: true` only after all required fixtures and corpus thresholds pass.

## Memory and execution policy

Pairwise and MSA activations, not the int8 weights, are expected to dominate memory as input size grows. The runtime therefore:

- loads weight shards lazily where module boundaries allow;
- evaluates and releases intermediate MLX graphs at recycling and diffusion-step boundaries;
- runs diffusion samples sequentially;
- exposes token, atom, and MSA-depth limits in the artifact configuration;
- checks estimated peak allocation before inference and returns a typed limit error rather than allowing an operating-system termination.

Initial validated limits are measured on an iPhone 15 Pro and recorded in the artifact manifest. Larger devices may opt into higher limits without changing model semantics.

## Public Swift API

```swift
public struct BoltzPredictionOptions: Sendable {
    public var recyclingSteps: Int
    public var diffusionSteps: Int
    public var seed: UInt64
}

public struct BoltzStructure: Sendable {
    public let coordinates: [SIMD3<Float>]
    public let atomMask: [Bool]
}

public actor BoltzPredictor {
    public init(modelDirectory: URL, limits: BoltzInputLimits = .artifactDefaults) async throws
    public func predict(
        featureDirectory: URL,
        options: BoltzPredictionOptions
    ) async throws -> BoltzStructure
}
```

Errors are typed as artifact, feature-schema, input-limit, unsupported-operation, or execution failures. Cancellation is checked between recycling iterations and diffusion steps.

## Compatibility and evolution

Both model and feature artifacts use an integer schema version. The loader accepts only versions implemented by the package. Tensor names are part of the schema and are not inferred from arbitrary PyTorch state dictionaries. Later milestones may add on-device preprocessing, confidence, affinity, or alternative group sizes behind new schema capabilities without weakening v1 validation.

## Acceptance criteria

Milestone one is complete when:

1. The exporter converts the pinned Boltz-2 checkpoint into float16 and int8 structure-only artifacts without omitted eligible matrices.
2. The Swift package and macOS CLI build with pinned MLX Swift `0.31.6`.
3. Primitive, trunk, diffusion-conditioning, score-network, and deterministic sampler parity fixtures pass.
4. The iOS demo performs a prediction from bundled precomputed features on an iPhone 15 Pro or simulator where Metal execution is available.
5. The int8 artifact stays within the measured device memory limit and passes the frozen-corpus quality thresholds recorded by the validation report.
6. Documentation explains artifact generation, macOS reference execution, iOS integration, supported limits, and known exclusions.
