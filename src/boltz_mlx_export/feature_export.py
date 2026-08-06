"""Export precomputed upstream Boltz features for native inference."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file

from boltz_mlx_export.schema import (
    SCHEMA_VERSION,
    ArtifactManifest,
    FeatureMetadata,
    TensorSpec,
)
from boltz_mlx_export.tensor_io import save_torch_tensors

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

MIN_MSA_MASK_RANK = 2


def _tensor_dtype(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _required_count(features: Mapping[str, Any], name: str) -> int:
    value = features.get(name)
    if not isinstance(value, torch.Tensor):
        message = f"feature batch is missing required tensor {name!r}"
        raise TypeError(message)
    return int(value.to(dtype=torch.int64).sum().item())


def export_feature_batch(
    features: Mapping[str, Any],
    output: Path,
    *,
    sample_id: str,
) -> ArtifactManifest:
    """Write one already-collated feature dictionary as a v1 bundle."""
    tensors = {
        name: value
        for name, value in features.items()
        if isinstance(value, torch.Tensor)
    }
    if not tensors:
        message = "feature batch contains no tensors"
        raise ValueError(message)

    token_count = _required_count(tensors, "token_pad_mask")
    atom_count = _required_count(tensors, "atom_pad_mask")
    msa_mask = tensors.get("msa_mask")
    if msa_mask is None or msa_mask.ndim < MIN_MSA_MASK_RANK:
        message = "feature batch is missing a ranked msa_mask tensor"
        raise ValueError(message)
    msa_depth = int(msa_mask.shape[1])

    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "features.safetensors"
    save_torch_tensors(tensors, feature_path)
    tensor_specs = tuple(
        TensorSpec(
            name=name,
            shape=tuple(int(value) for value in tensor.shape),
            dtype=_tensor_dtype(tensor),
            shard=feature_path.name,
        )
        for name, tensor in sorted(tensors.items())
    )
    manifest = ArtifactManifest.features_v1(tensors=tensor_specs)
    manifest.write(output / "manifest.json")
    FeatureMetadata(
        schema_version=SCHEMA_VERSION,
        sample_id=sample_id,
        token_count=token_count,
        atom_count=atom_count,
        msa_depth=msa_depth,
    ).write(output / "metadata.json")
    return manifest


def load_feature_bundle(directory: Path) -> dict[str, torch.Tensor]:
    """Load a feature bundle after validating its v1 manifest."""
    manifest = ArtifactManifest.read(directory / "manifest.json")
    if manifest.artifact_kind.value != "features":
        message = f"expected features artifact, found {manifest.artifact_kind.value}"
        raise ValueError(message)
    arrays = load_file(directory / "features.safetensors", device="cpu")
    expected = {tensor.name for tensor in manifest.tensors}
    if set(arrays) != expected:
        message = "feature tensor names do not match manifest"
        raise ValueError(message)
    return arrays


def export_input_features(*, input_path: Path, output: Path) -> ArtifactManifest:
    """Run pinned Boltz preprocessing offline and export its single batch."""
    from boltz.data.module.inferencev2 import (  # noqa: PLC0415
        PredictionDataset,
        collate,
    )
    from boltz.data.types import Manifest  # noqa: PLC0415
    from boltz.main import (  # noqa: PLC0415
        check_inputs,
        get_cache_path,
        process_inputs,
    )

    cache = Path(get_cache_path()).expanduser()
    mol_dir = cache / "mols"
    if not mol_dir.is_dir():
        message = (
            f"Boltz-2 molecule cache is missing at {mol_dir}; "
            "run one upstream `boltz predict` preprocessing job first"
        )
        raise FileNotFoundError(message)
    inputs = check_inputs(input_path.expanduser())
    if len(inputs) != 1:
        message = f"feature export requires exactly one input, found {len(inputs)}"
        raise ValueError(message)

    with tempfile.TemporaryDirectory(prefix="boltz-mlx-features-") as temporary:
        processed_root = Path(temporary)
        process_inputs(
            data=inputs,
            out_dir=processed_root,
            ccd_path=cache / "ccd.pkl",
            mol_dir=mol_dir,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            boltz2=True,
            use_msa_server=False,
            preprocessing_threads=1,
        )
        processed = processed_root / "processed"
        manifest = Manifest.load(processed / "manifest.json")
        if len(manifest.records) != 1:
            message = "Boltz preprocessing did not produce exactly one record"
            raise RuntimeError(message)
        dataset = PredictionDataset(
            manifest=manifest,
            target_dir=processed / "structures",
            msa_dir=processed / "msa",
            mol_dir=mol_dir,
            constraints_dir=processed / "constraints",
            template_dir=processed / "templates",
            extra_mols_dir=processed / "mols",
        )
        batch = collate([dataset[0]])
        return export_feature_batch(
            batch,
            output,
            sample_id=manifest.records[0].id,
        )
