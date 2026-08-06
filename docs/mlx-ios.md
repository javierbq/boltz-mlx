# Boltz-2 structure inference with MLX Swift

This port targets iPhone 15 Pro or newer on iOS 17+. Learned inference is native Swift/MLX; Python and PyTorch are used only to preprocess inputs, convert the upstream checkpoint, and generate parity fixtures.

## Scope

- Boltz-2 structure prediction, pinned to upstream `v2.2.1` (`cb04aeccdd480fd4db707f0bbafde538397fa2ac`)
- Affine 8-bit weight-only quantization, group size 64, floating-point activations
- Quantized Linear and Embedding weights, including narrow matrices zero-padded to a physical 64-column group
- Input embedding, templates, MSA stack, 64-block Pairformer trunk, diffusion conditioning, score network, and sampler
- SafeTensors plus strict versioned JSON manifests
- Native Swift API, macOS CLI, and SwiftUI iOS demo

The initial milestone does not include confidence, affinity, B-factor prediction, potentials, training, or on-device feature generation.

## Offline setup and export

Use Python 3.12 on Apple silicon. A virtual environment outside an iCloud-optimized directory is recommended for large checkpoints.

```bash
python3.12 -m venv /tmp/boltz-mlx-venv
/tmp/boltz-mlx-venv/bin/pip install -e '.[mlx-export]'

/tmp/boltz-mlx-venv/bin/boltz-mlx export-model \
  --checkpoint ~/.boltz/boltz2_conf.ckpt \
  --output ~/BoltzArtifacts/boltz2-int8
```

The model directory contains `model.safetensors`, `manifest.json`, and `config.json`. Conversion retains only the structure path and records the exact logical and padded physical shape of each quantized matrix.

Feature preprocessing currently reuses upstream Boltz:

```bash
/tmp/boltz-mlx-venv/bin/boltz-mlx export-features examples/prot_no_msa.yaml \
  --output ~/BoltzArtifacts/protein-features
```

The Boltz molecule cache must already exist under `$BOLTZ_CACHE` or `~/.boltz`. Feature artifacts contain `features.safetensors`, `manifest.json`, and `metadata.json`.

## macOS CLI

```bash
swift run BoltzMLXCLI predict \
  --model ~/BoltzArtifacts/boltz2-int8 \
  --features ~/BoltzArtifacts/protein-features \
  --output /tmp/prediction.safetensors \
  --recycling-steps 0 \
  --diffusion-steps 20 \
  --seed 0
```

The output contains unpadded `coordinates` and `atom_mask` arrays. A sorted JSON sidecar records the sampling settings.

## Swift API

```swift
let predictor = try BoltzPredictor(modelDirectory: modelURL)
let features = try FeatureBundle.load(from: featureURL)
let structure = try await predictor.predict(
  features: features,
  options: BoltzPredictionOptions(
    recyclingSteps: 0,
    diffusionSteps: 20,
    seed: 0
  )
)
```

`BoltzPredictor` is an actor, so only one large inference graph executes at a time. The default `MemoryPlanner` caps tokens at 256, padded atoms at 2,048, and MSA depth at 1,024, sets a 64 MiB MLX cache, and rejects estimated activation use above 6 GiB before building pair tensors. These are conservative starting limits, not validated device maxima.

## iOS demo

Generate and build the demo project:

```bash
scripts/build_ios_demo.sh simulator
```

For a connected device use `scripts/build_ios_demo.sh device`, then sign from Xcode. The app imports model and feature folders with the document picker and reports phase, token and atom counts, elapsed time, MLX peak memory, and cancellation state.

## Artifact contract

Schema version 1 rejects unknown versions, kinds, tensor names, shapes, and dtypes before graph construction. Model tensors use their upstream module paths. Quantized matrices are stored as packed `weight`, `scales`, and affine `biases`; ordinary Linear bias is stored separately as `bias`.

The architecture configuration is exported from Lightning checkpoint hyperparameters instead of being hard-coded in Swift. It includes model widths, window sizes, feature flags, MSA/template/Pairformer depths, diffusion transformer layout, and inference schedule.

## Validation status

Run `scripts/validate_mlx_port.sh` for Python tests, Swift tests, strict formatting, CLI build, and simulator build. Production structural quality remains a scientific validation gate: do not treat int8 output as equivalent to upstream Boltz until the frozen protein and protein-ligand corpus has recorded aligned RMSD and lDDT-style metrics in `validation/quality/report.json`.
