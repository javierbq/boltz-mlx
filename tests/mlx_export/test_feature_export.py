from pathlib import Path

import torch
from safetensors.torch import load_file

from boltz_mlx_export.feature_export import export_feature_batch
from boltz_mlx_export.schema import ArtifactKind, ArtifactManifest, FeatureMetadata

EXPECTED_TOKEN_COUNT = 2
EXPECTED_ATOM_COUNT = 3
EXPECTED_MSA_DEPTH = 5


def test_feature_bundle_is_deterministic(tmp_path: Path) -> None:
    """Equivalent batches produce byte-identical SafeTensors and JSON."""
    features = {
        "token_pad_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "atom_pad_mask": torch.tensor([[1, 0]], dtype=torch.bool),
        "msa_mask": torch.ones((1, 3, 2), dtype=torch.float32),
        "record": object(),
    }

    export_feature_batch(features, tmp_path / "a", sample_id="tiny")
    export_feature_batch(features, tmp_path / "b", sample_id="tiny")

    assert (tmp_path / "a/features.safetensors").read_bytes() == (
        tmp_path / "b/features.safetensors"
    ).read_bytes()
    assert (tmp_path / "a/manifest.json").read_bytes() == (
        tmp_path / "b/manifest.json"
    ).read_bytes()
    assert (tmp_path / "a/metadata.json").read_bytes() == (
        tmp_path / "b/metadata.json"
    ).read_bytes()


def test_feature_bundle_records_dimensions_and_tensor_dtypes(tmp_path: Path) -> None:
    """Swift can validate allocation dimensions before loading the network."""
    features = {
        "token_pad_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "atom_pad_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
        "msa_mask": torch.ones((1, 5, 3), dtype=torch.float32),
        "token_index": torch.tensor([[0, 1, 2]], dtype=torch.int64),
    }

    manifest = export_feature_batch(features, tmp_path, sample_id="dimensions")

    arrays = load_file(tmp_path / "features.safetensors")
    metadata = FeatureMetadata.read(tmp_path / "metadata.json")
    assert set(arrays) == set(features)
    assert arrays["token_index"].dtype is torch.int64
    assert metadata.token_count == EXPECTED_TOKEN_COUNT
    assert metadata.atom_count == EXPECTED_ATOM_COUNT
    assert metadata.msa_depth == EXPECTED_MSA_DEPTH
    assert metadata.sample_id == "dimensions"
    assert manifest.artifact_kind is ArtifactKind.FEATURES
    assert ArtifactManifest.read(tmp_path / "manifest.json") == manifest
