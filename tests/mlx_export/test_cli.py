from click.testing import CliRunner

from boltz_mlx_export.cli import cli


def test_cli_lists_export_commands() -> None:
    """The root help advertises every artifact-generation operation."""
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "export-model" in result.output
    assert "export-features" in result.output
    assert "make-fixtures" in result.output


def test_cli_reports_version() -> None:
    """The command exposes the artifact schema version."""
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "schema 1" in result.output
