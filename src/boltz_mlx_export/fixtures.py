"""PyTorch forward-hook fixtures used to validate the MLX Swift port."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from boltz_mlx_export.feature_export import load_feature_bundle
from boltz_mlx_export.model_export import _load_boltz2
from boltz_mlx_export.tensor_io import flatten_tensors, save_torch_tensors

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType
    from typing import Self

DEFAULT_BOUNDARIES = (
    "input_embedder",
    "msa_module",
    "pairformer_module",
    "diffusion_conditioning",
    "structure_module.score_model",
)


class FixtureRecorder:
    """Record deterministic tensor inputs and outputs from module forward hooks."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._calls: list[dict[str, int | float | str]] = []

    def __enter__(self) -> Self:
        """Create the output directory and begin a recording scope."""
        self.output.mkdir(parents=True, exist_ok=True)
        return self

    def watch(
        self,
        name: str,
        module: torch.nn.Module,
        *,
        atol: float = 1e-4,
        rtol: float = 1e-4,
    ) -> None:
        """Attach one named forward hook with declared comparison tolerances."""

        def record(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
        ) -> None:
            call_index = self._counts[name]
            self._counts[name] += 1
            arrays = flatten_tensors(inputs, "input")
            arrays.update(flatten_tensors(output, "output"))
            filename = f"{name}__{call_index:03d}.safetensors"
            save_torch_tensors(arrays, self.output / filename)
            self._calls.append(
                {
                    "atol": atol,
                    "call_index": call_index,
                    "file": filename,
                    "module": name,
                    "rtol": rtol,
                },
            )

        self._handles.append(module.register_forward_hook(record))

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Remove hooks and write the deterministic call index."""
        for handle in self._handles:
            handle.remove()
        payload = {"calls": self._calls, "schema_version": 1}
        (self.output / "fixtures.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def make_fixtures(*, checkpoint: Path, features: Path, output: Path) -> None:
    """Run a small deterministic forward pass and record major model boundaries."""
    model = _load_boltz2(checkpoint)
    model.confidence_prediction = False
    model.affinity_prediction = False
    feature_tensors = load_feature_bundle(features)
    modules = dict(model.named_modules())
    torch.manual_seed(0)
    with FixtureRecorder(output) as recorder:
        for name in DEFAULT_BOUNDARIES:
            module = modules.get(name)
            if module is None:
                message = f"checkpoint model is missing fixture boundary {name}"
                raise ValueError(message)
            recorder.watch(name, module)
        with torch.inference_mode():
            model(
                feature_tensors,
                recycling_steps=0,
                num_sampling_steps=3,
                diffusion_samples=1,
                max_parallel_samples=1,
            )
