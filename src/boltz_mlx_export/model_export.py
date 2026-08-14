"""Convert pinned Boltz-2 structure weights into MLX Swift artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_torch_file

from boltz_mlx_export.names import select_structure_values, swift_tensor_name
from boltz_mlx_export.quantization import (
    INT8_BITS,
    INT8_GROUP_SIZE,
    quantize_affine_int8,
)
from boltz_mlx_export.schema import ArtifactManifest, TensorSpec

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet
    from pathlib import Path

    from torch import Tensor


#: Manifest dtype spellings. These must match ArtifactIO.dtypeName on the Swift side
#: exactly: the loader compares the string per tensor and throws on any mismatch.
_TORCH_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


@dataclass(frozen=True)
class ModelConfiguration:
    """Versioned Boltz-2 architecture values required by the Swift runtime."""

    schema_version: int
    source_revision: str
    source_commit: str
    atom_s: int
    atom_z: int
    token_s: int
    token_z: int
    num_bins: int
    atom_feature_dim: int
    atoms_per_window_queries: int
    atoms_per_window_keys: int
    fix_sym_check: bool
    cyclic_pos_enc: bool
    bond_type_feature: bool
    use_templates: bool
    use_templates_v2: bool
    conditioning_cutoff_min: float
    conditioning_cutoff_max: float
    embedder_args: dict[str, object]
    msa_args: dict[str, object]
    pairformer_args: dict[str, object]
    # Confidence architecture. REQUIRED for a runnable confidence head, and NOT inferable from
    # defaults: this checkpoint sets add_s_to_z_prod / add_s_input_to_s / add_z_input_to_z all True,
    # each of which changes WHICH TENSORS EXIST, and confidence_args.use_separate_heads True means
    # there is no `to_pae_logits` at all — PAE comes from to_pae_intra_logits + to_pae_inter_logits
    # combined via asym_id. A port written from the obvious reading of confidencev2.py fails its
    # weight lookup. None when the checkpoint has no confidence head.
    confidence_model_args: dict[str, object] | None
    score_model_args: dict[str, object]
    diffusion_process_args: dict[str, object]
    template_args: dict[str, object] | None

    @classmethod
    def from_hparams(cls, hparams: Mapping[str, object]) -> ModelConfiguration:
        """Select and normalize the architecture contract from Lightning hparams."""

        def required(name: str) -> object:
            if name not in hparams:
                message = f"checkpoint is missing required architecture value: {name}"
                raise ValueError(message)
            return hparams[name]

        def mapping(name: str) -> dict[str, object]:
            value = required(name)
            if not _is_mapping(value):
                message = f"checkpoint architecture value {name} must be a mapping"
                raise TypeError(message)
            return _json_mapping(value)

        template_value = hparams.get("template_args")
        if template_value is not None and not _is_mapping(template_value):
            message = "checkpoint architecture value template_args must be a mapping"
            raise TypeError(message)

        diffusion_process_args = mapping("diffusion_process_args")
        diffusion_process_args.setdefault("num_sampling_steps", 5)

        return cls(
            schema_version=1,
            source_revision="v2.2.1",
            source_commit="cb04aeccdd480fd4db707f0bbafde538397fa2ac",
            atom_s=int(required("atom_s")),
            atom_z=int(required("atom_z")),
            token_s=int(required("token_s")),
            token_z=int(required("token_z")),
            num_bins=int(required("num_bins")),
            atom_feature_dim=int(hparams.get("atom_feature_dim", 128)),
            atoms_per_window_queries=int(
                hparams.get("atoms_per_window_queries", 32)
            ),
            atoms_per_window_keys=int(hparams.get("atoms_per_window_keys", 128)),
            fix_sym_check=bool(hparams.get("fix_sym_check", False)),
            cyclic_pos_enc=bool(hparams.get("cyclic_pos_enc", False)),
            bond_type_feature=bool(hparams.get("bond_type_feature", False)),
            use_templates=bool(hparams.get("use_templates", False)),
            use_templates_v2=bool(hparams.get("use_templates_v2", False)),
            conditioning_cutoff_min=float(
                hparams.get("conditioning_cutoff_min", 4.0)
            ),
            conditioning_cutoff_max=float(
                hparams.get("conditioning_cutoff_max", 20.0)
            ),
            embedder_args=mapping("embedder_args"),
            msa_args=mapping("msa_args"),
            pairformer_args=mapping("pairformer_args"),
            confidence_model_args=(
                _json_mapping(hparams["confidence_model_args"])
                if _is_mapping(hparams.get("confidence_model_args"))
                else None
            ),
            score_model_args=mapping("score_model_args"),
            diffusion_process_args=diffusion_process_args,
            template_args=(
                _json_mapping(template_value) if template_value is not None else None
            ),
        )

    def write(self, path: Path) -> None:
        """Write deterministic configuration JSON."""
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> ModelConfiguration:
        """Read a configuration written by :meth:`write`."""
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(**value)


def _is_mapping(value: object) -> bool:
    """Accept standard mappings and OmegaConf's mapping-compatible DictConfig."""
    return isinstance(value, Mapping) or callable(getattr(value, "items", None))


def _json_mapping(value: object) -> dict[str, object]:
    """Copy a nested checkpoint mapping into deterministic JSON-compatible data."""
    if not _is_mapping(value):
        message = f"architecture value is not a mapping: {type(value)}"
        raise TypeError(message)
    normalized: dict[str, object] = {}
    for key, item in value.items():  # type: ignore[union-attr]
        name = str(key)
        normalized[name] = _json_value(name, item)
    return normalized


def _json_value(name: str, value: object) -> object:
    """Normalize one OmegaConf or Python value for stable JSON output."""
    if _is_mapping(value):
        return _json_mapping(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Iterable):
        return [
            _json_value(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        ]
    message = f"architecture value {name} is not JSON serializable: {type(value)}"
    raise TypeError(message)


def select_structure_state_dict(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Select only state required by the Boltz-2 structure path."""
    return select_structure_values(state)


class Precision(StrEnum):
    """Weight representation for an exported model artifact.

    ``INT8`` packs every eligible matrix with MLX affine int8 (group 64) and stores
    everything else as float16. ``FLOAT16`` and ``BFLOAT16`` store every float tensor
    dense at that width, matrices included, and emit no quantization block.

    A pack is single-dtype on purpose: MLX promotes a mixed float16/bfloat16 operation
    to float32, so a pack that mixed widths would silently change both numerics and
    speed relative to either pure pack.
    """

    INT8 = "int8"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    @property
    def torch_dtype(self) -> torch.dtype:
        """The torch dtype float tensors are stored at under this precision."""
        return (
            torch.bfloat16 if self is Precision.BFLOAT16 else torch.float16
        )


def _numpy_parameter(tensor: Tensor) -> np.ndarray:
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.to(torch.float16)
    array = value.numpy()
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float16, copy=False)
    return array


def _dense_parameter(tensor: Tensor, dtype: torch.dtype) -> Tensor:
    """Return one contiguous tensor, floats narrowed to the pack's dtype."""
    value = tensor.detach().cpu().contiguous()
    if value.dtype.is_floating_point:
        return value.to(dtype)
    return value


def _export_dense(
    selected: Mapping[str, Tensor],
    *,
    output: Path,
    source_checkpoint_sha256: str,
    precision: Precision,
) -> ArtifactManifest:
    """Write an unquantized artifact directory at one uniform float width.

    Matrices carry no ``logical_shape``/``physical_shape``: quantization padded the
    input width up to the group size, and dense weights need no padding, so the shape
    is its own single source of truth.
    """
    dtype = precision.torch_dtype
    tensors: dict[str, Tensor] = {}
    specs: list[TensorSpec] = []
    for torch_name, tensor in selected.items():
        name = swift_tensor_name(torch_name)
        value = _dense_parameter(tensor, dtype)
        tensors[name] = value
        specs.append(
            TensorSpec(
                name=name,
                shape=tuple(int(dimension) for dimension in value.shape),
                dtype=_TORCH_DTYPE_NAMES[value.dtype],
            ),
        )

    output.mkdir(parents=True, exist_ok=True)
    # safetensors.numpy cannot round-trip bfloat16 -- numpy has no such dtype -- so
    # dense packs are written through the torch backend regardless of width.
    save_torch_file(dict(sorted(tensors.items())), output / "model.safetensors")
    manifest = ArtifactManifest.model_v1(
        source_checkpoint_sha256=source_checkpoint_sha256,
        tensors=tuple(sorted(specs, key=lambda spec: spec.name)),
        quantization=None,
    )
    manifest.write(output / "manifest.json")
    return manifest


def export_state_dict(
    state: Mapping[str, Tensor],
    *,
    output: Path,
    eligible_matrix_names: AbstractSet[str],
    source_checkpoint_sha256: str,
    precision: Precision = Precision.INT8,
) -> ArtifactManifest:
    """Write a deterministic, structure-only model artifact directory."""
    selected = select_structure_state_dict(state)
    missing = set(eligible_matrix_names).difference(selected)
    if missing:
        message = (
            f"eligible matrix names are absent from structure state: {sorted(missing)}"
        )
        raise ValueError(message)

    if precision is not Precision.INT8:
        return _export_dense(
            selected,
            output=output,
            source_checkpoint_sha256=source_checkpoint_sha256,
            precision=precision,
        )

    arrays: dict[str, np.ndarray] = {}
    specs: list[TensorSpec] = []
    for torch_name, tensor in selected.items():
        name = swift_tensor_name(torch_name)
        if torch_name in eligible_matrix_names:
            quantized = quantize_affine_int8(_numpy_parameter(tensor))
            module_name = name.removesuffix(".weight")
            quantized_arrays = {
                f"{module_name}.weight": quantized.weight,
                f"{module_name}.scales": quantized.scales,
                f"{module_name}.biases": quantized.biases,
            }
            for quantized_name, array in quantized_arrays.items():
                arrays[quantized_name] = array
                is_weight = quantized_name.endswith(".weight")
                specs.append(
                    TensorSpec(
                        name=quantized_name,
                        shape=tuple(int(value) for value in array.shape),
                        dtype=str(array.dtype),
                        logical_shape=quantized.logical_shape if is_weight else None,
                        physical_shape=quantized.physical_shape if is_weight else None,
                    ),
                )
        else:
            array = _numpy_parameter(tensor)
            arrays[name] = array
            specs.append(
                TensorSpec(
                    name=name,
                    shape=tuple(int(value) for value in array.shape),
                    dtype=str(array.dtype),
                ),
            )

    output.mkdir(parents=True, exist_ok=True)
    save_file(dict(sorted(arrays.items())), output / "model.safetensors")
    manifest = ArtifactManifest.model_v1(
        source_checkpoint_sha256=source_checkpoint_sha256,
        tensors=tuple(sorted(specs, key=lambda spec: spec.name)),
        quantization={
            "bits": INT8_BITS,
            "group_size": INT8_GROUP_SIZE,
            "mode": "affine",
        },
    )
    manifest.write(output / "manifest.json")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_boltz2(checkpoint: Path) -> torch.nn.Module:
    from boltz.main import (  # noqa: PLC0415
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2  # noqa: PLC0415

    return Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 20,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        # Deliberately NOT overridden: the checkpoint's own confidence_model_args must be used, or
        # Boltz2.__init__ does ConfidenceModule(**None) and the load fails. It was previously
        # overridden to None because only structure weights were exported.
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(BoltzSteeringParams()),
    ).eval()


def _eligible_matrix_names(model: torch.nn.Module) -> set[str]:
    eligible: set[str] = set()
    # Lightning checkpoints retain aliases for shared modules. Enumerate those aliases
    # too so no duplicate Linear matrix silently falls back to float16.
    for module_name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            eligible.add(f"{module_name}.weight")
    return set(select_structure_values(dict.fromkeys(eligible)))


def export_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    precision: Precision = Precision.INT8,
) -> ArtifactManifest:
    """Load and export the pinned production Boltz-2 checkpoint."""
    if not checkpoint.is_file():
        message = f"checkpoint does not exist: {checkpoint}"
        raise FileNotFoundError(message)
    model = _load_boltz2(checkpoint)
    manifest = export_state_dict(
        model.state_dict(),
        output=output,
        eligible_matrix_names=_eligible_matrix_names(model),
        source_checkpoint_sha256=_sha256(checkpoint),
        precision=precision,
    )
    ModelConfiguration.from_hparams(dict(model.hparams)).write(
        output / "config.json"
    )
    return manifest
