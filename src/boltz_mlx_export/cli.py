"""Command-line interface for offline Boltz MLX artifact generation."""

from pathlib import Path

import click

from boltz_mlx_export.schema import SCHEMA_VERSION


@click.group()
@click.version_option(version=f"schema {SCHEMA_VERSION}", prog_name="boltz-mlx")
def cli() -> None:
    """Export Boltz-2 model and feature artifacts for MLX Swift."""


@cli.command("export-model")
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--precision",
    type=click.Choice(["int8", "float16", "bfloat16"]),
    default="int8",
    show_default=True,
    help="int8 quantizes every matrix; float16/bfloat16 store them dense (~2x larger).",
)
def export_model_command(checkpoint: Path, output: Path, precision: str) -> None:
    """Export structure-only weights from a Boltz-2 checkpoint."""
    from boltz_mlx_export.model_export import (  # noqa: PLC0415
        Precision,
        export_checkpoint,
    )

    export_checkpoint(
        checkpoint=checkpoint,
        output=output,
        precision=Precision(precision),
    )


@cli.command("export-features")
@click.argument("input_path", type=click.Path(path_type=Path, exists=True))
@click.option("--output", type=click.Path(path_type=Path), required=True)
def export_features_command(input_path: Path, output: Path) -> None:
    """Precompute and export a Boltz-2 input feature bundle."""
    from boltz_mlx_export.feature_export import export_input_features  # noqa: PLC0415

    export_input_features(input_path=input_path, output=output)


@cli.command("make-fixtures")
@click.option("--checkpoint", type=click.Path(path_type=Path), required=True)
@click.option("--features", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def make_fixtures_command(checkpoint: Path, features: Path, output: Path) -> None:
    """Record deterministic PyTorch boundary fixtures for Swift tests."""
    from boltz_mlx_export.fixtures import make_fixtures  # noqa: PLC0415

    make_fixtures(checkpoint=checkpoint, features=features, output=output)
