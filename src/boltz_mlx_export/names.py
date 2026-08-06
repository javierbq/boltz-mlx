"""Tensor-name rules shared by checkpoint export and parity fixtures."""

from collections.abc import Mapping
from typing import TypeVar

STRUCTURE_PREFIXES = (
    "input_embedder",
    "s_init",
    "z_init_1",
    "z_init_2",
    "rel_pos",
    "token_bonds",
    "token_bonds_type",
    "contact_conditioning",
    "s_norm",
    "z_norm",
    "s_recycle",
    "z_recycle",
    "template_module",
    "msa_module",
    "pairformer_module",
    "diffusion_conditioning",
    "structure_module",
    # Confidence. `distogram_module` is NOT optional here: boltz2.py feeds
    # dict_out["pdistogram"] into confidence_module as pred_distogram_logits, so exporting the
    # confidence head without it produces a graph that cannot run.
    "confidence_module",
    "distogram_module",
)

Value = TypeVar("Value")


def is_structure_tensor(name: str) -> bool:
    """Return whether a state-dictionary key belongs to structure inference."""
    root = name.split(".", maxsplit=1)[0]
    return root in STRUCTURE_PREFIXES


def select_structure_values(values: Mapping[str, Value]) -> dict[str, Value]:
    """Copy structure-only values in deterministic name order."""
    return {name: values[name] for name in sorted(values) if is_structure_tensor(name)}


def swift_tensor_name(torch_name: str) -> str:
    """Normalize wrapper-only path components out of a PyTorch tensor name."""
    return ".".join(
        component for component in torch_name.split(".") if component != "_orig_mod"
    )
