# Boltz MLX iOS Demo

This SwiftUI app targets iOS 17 and runs the native MLX Swift structure path. It consumes two folders produced offline by `boltz-mlx`: the affine-int8 model artifact and a precomputed feature bundle.

Generate and build the project from the repository root:

```bash
scripts/build_ios_demo.sh simulator
```

On device, choose both folders through the document picker, then start prediction. The screen reports phase, elapsed time, input tokens, output atoms, and peak MLX memory. Inference is intentionally serialized and can be cancelled between diffusion steps.
