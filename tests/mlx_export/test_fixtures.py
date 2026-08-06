import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from boltz_mlx_export.fixtures import FixtureRecorder

FIRST_OUTPUT = 2
SECOND_OUTPUT = 3
TEST_ATOL = 1e-5
TEST_RTOL = 1e-4


class AddOne(torch.nn.Module):
    """Small real module used to exercise PyTorch forward hooks."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add one to the input."""
        return value + 1


def test_fixture_recorder_numbers_repeated_module_calls(tmp_path: Path) -> None:
    """Recycling calls to the same module remain independently addressable."""
    module = AddOne()

    with FixtureRecorder(tmp_path) as recorder:
        recorder.watch("recycle.block", module, atol=TEST_ATOL, rtol=TEST_RTOL)
        module(torch.tensor([1.0]))
        module(torch.tensor([2.0]))

    first = tmp_path / "recycle.block__000.safetensors"
    second = tmp_path / "recycle.block__001.safetensors"
    assert load_file(first)["output"].item() == FIRST_OUTPUT
    assert load_file(second)["output"].item() == SECOND_OUTPUT
    metadata = json.loads((tmp_path / "fixtures.json").read_text())
    assert [call["call_index"] for call in metadata["calls"]] == [0, 1]
    assert metadata["calls"][0]["atol"] == TEST_ATOL
    assert metadata["calls"][0]["rtol"] == TEST_RTOL


def test_fixture_recorder_flattens_tuple_and_mapping_outputs(tmp_path: Path) -> None:
    """Nested module boundaries use stable tensor keys in SafeTensors."""

    class Structured(torch.nn.Module):
        def forward(
            self,
            value: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return value, {"double": value * 2}

    module = Structured()
    with FixtureRecorder(tmp_path) as recorder:
        recorder.watch("structured", module)
        module(torch.tensor([3.0]))

    arrays = load_file(tmp_path / "structured__000.safetensors")
    assert set(arrays) == {"input.0", "output.0", "output.1.double"}
