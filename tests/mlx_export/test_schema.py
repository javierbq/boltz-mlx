from pathlib import Path

import pytest

from boltz_mlx_export.schema import (
    ArtifactKind,
    ArtifactManifest,
    ManifestValidationError,
    TensorSpec,
)


def test_model_manifest_round_trip(tmp_path: Path) -> None:
    """A model manifest preserves enum and tuple values through JSON."""
    manifest = ArtifactManifest.model_v1(
        source_checkpoint_sha256="a" * 64,
        tensors=(
            TensorSpec(
                name="trunk.s_init.weight",
                shape=(64, 2),
                dtype="uint32",
            ),
        ),
    )
    path = tmp_path / "manifest.json"

    manifest.write(path)

    assert ArtifactManifest.read(path) == manifest
    assert manifest.artifact_kind is ArtifactKind.MODEL


def test_manifest_json_is_deterministic(tmp_path: Path) -> None:
    """Equivalent manifests produce byte-identical JSON files."""
    manifest = ArtifactManifest.model_v1(source_checkpoint_sha256="b" * 64)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    manifest.write(first)
    manifest.write(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text().endswith("\n")


def test_manifest_rejects_unknown_schema(tmp_path: Path) -> None:
    """Readers fail explicitly instead of guessing at unknown schemas."""
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"artifact_kind":"model","schema_version":99,'
        '"source_revision":"v2.2.1","tensors":[]}\n',
    )

    with pytest.raises(ManifestValidationError, match="schema version 99"):
        ArtifactManifest.read(path)
