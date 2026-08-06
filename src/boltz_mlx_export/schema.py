"""Stable JSON schemas shared by the offline exporter and Swift runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1
SHA256_HEX_LENGTH = 64
BOLTZ_SOURCE_REVISION = "v2.2.1"
BOLTZ_SOURCE_COMMIT = "cb04aeccdd480fd4db707f0bbafde538397fa2ac"


class ManifestValidationError(ValueError):
    """Raised when an artifact manifest violates the supported schema."""


class ArtifactKind(StrEnum):
    """Kinds of directories understood by Boltz MLX."""

    MODEL = "model"
    FEATURES = "features"
    FIXTURE = "fixture"


@dataclass(frozen=True)
class TensorSpec:
    """Serializable declaration for one tensor in an artifact."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str = "model.safetensors"
    logical_shape: tuple[int, ...] | None = None
    physical_shape: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TensorSpec:
        """Decode a tensor declaration from JSON-compatible values."""
        return cls(
            name=str(value["name"]),
            shape=tuple(int(dimension) for dimension in value["shape"]),
            dtype=str(value["dtype"]),
            shard=str(value.get("shard", "model.safetensors")),
            logical_shape=(
                tuple(int(dimension) for dimension in value["logical_shape"])
                if value.get("logical_shape") is not None
                else None
            ),
            physical_shape=(
                tuple(int(dimension) for dimension in value["physical_shape"])
                if value.get("physical_shape") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """Top-level manifest for a model, feature bundle, or parity fixture."""

    supported_schema_version: ClassVar[int] = SCHEMA_VERSION

    schema_version: int
    artifact_kind: ArtifactKind
    source_revision: str
    source_commit: str
    source_checkpoint_sha256: str | None
    tensors: tuple[TensorSpec, ...]
    quantization: dict[str, int | str] | None = None

    def __post_init__(self) -> None:
        """Validate invariants that both exporters and runtimes rely on."""
        if self.schema_version != self.supported_schema_version:
            message = (
                f"unsupported schema version {self.schema_version}; "
                f"expected {self.supported_schema_version}"
            )
            raise ManifestValidationError(message)
        names = [tensor.name for tensor in self.tensors]
        if len(names) != len(set(names)):
            message = "tensor names must be unique"
            raise ManifestValidationError(message)
        if self.source_checkpoint_sha256 is not None and (
            len(self.source_checkpoint_sha256) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef"
                for character in self.source_checkpoint_sha256
            )
        ):
            message = (
                "source_checkpoint_sha256 must be 64 lowercase hexadecimal characters"
            )
            raise ManifestValidationError(message)

    @classmethod
    def model_v1(
        cls,
        *,
        source_checkpoint_sha256: str,
        tensors: tuple[TensorSpec, ...] = (),
        quantization: dict[str, int | str] | None = None,
    ) -> ArtifactManifest:
        """Construct a version-one model manifest pinned to Boltz 2.2.1."""
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_kind=ArtifactKind.MODEL,
            source_revision=BOLTZ_SOURCE_REVISION,
            source_commit=BOLTZ_SOURCE_COMMIT,
            source_checkpoint_sha256=source_checkpoint_sha256,
            tensors=tensors,
            quantization=quantization,
        )

    @classmethod
    def features_v1(
        cls,
        *,
        tensors: tuple[TensorSpec, ...],
    ) -> ArtifactManifest:
        """Construct a version-one precomputed feature manifest."""
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_kind=ArtifactKind.FEATURES,
            source_revision=BOLTZ_SOURCE_REVISION,
            source_commit=BOLTZ_SOURCE_COMMIT,
            source_checkpoint_sha256=None,
            tensors=tensors,
        )

    @classmethod
    def fixture_v1(
        cls,
        *,
        source_checkpoint_sha256: str,
        tensors: tuple[TensorSpec, ...],
    ) -> ArtifactManifest:
        """Construct a version-one PyTorch parity-fixture manifest."""
        return cls(
            schema_version=SCHEMA_VERSION,
            artifact_kind=ArtifactKind.FIXTURE,
            source_revision=BOLTZ_SOURCE_REVISION,
            source_commit=BOLTZ_SOURCE_COMMIT,
            source_checkpoint_sha256=source_checkpoint_sha256,
            tensors=tensors,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible mapping."""
        value = asdict(self)
        value["artifact_kind"] = self.artifact_kind.value
        value["tensors"] = [asdict(tensor) for tensor in self.tensors]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactManifest:
        """Validate and decode a JSON-compatible mapping."""
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                artifact_kind=ArtifactKind(value["artifact_kind"]),
                source_revision=str(value["source_revision"]),
                source_commit=str(value.get("source_commit", "")),
                source_checkpoint_sha256=value.get("source_checkpoint_sha256"),
                tensors=tuple(TensorSpec.from_dict(item) for item in value["tensors"]),
                quantization=value.get("quantization"),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ManifestValidationError):
                raise
            message = f"invalid artifact manifest: {error}"
            raise ManifestValidationError(message) from error

    def write(self, path: Path) -> None:
        """Write deterministic UTF-8 JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ArtifactManifest:
        """Read and validate a manifest from disk."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            message = f"cannot read artifact manifest: {error}"
            raise ManifestValidationError(message) from error
        if not isinstance(value, dict):
            message = "artifact manifest must contain a JSON object"
            raise ManifestValidationError(message)
        return cls.from_dict(value)


@dataclass(frozen=True)
class FeatureMetadata:
    """Non-tensor dimensions and identity for a feature bundle."""

    schema_version: Literal[1]
    sample_id: str
    token_count: int
    atom_count: int
    msa_depth: int

    def write(self, path: Path) -> None:
        """Write deterministic feature metadata JSON."""
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> FeatureMetadata:
        """Read and validate feature metadata JSON."""
        value = json.loads(path.read_text(encoding="utf-8"))
        schema_version = int(value["schema_version"])
        if schema_version != SCHEMA_VERSION:
            message = f"unsupported feature schema version {schema_version}"
            raise ManifestValidationError(message)
        return cls(
            schema_version=1,
            sample_id=str(value["sample_id"]),
            token_count=int(value["token_count"]),
            atom_count=int(value["atom_count"]),
            msa_depth=int(value["msa_depth"]),
        )
