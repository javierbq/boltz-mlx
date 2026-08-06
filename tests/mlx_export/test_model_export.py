import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from boltz_mlx_export.model_export import (
    ModelConfiguration,
    _eligible_matrix_names,
    export_state_dict,
    select_structure_state_dict,
)
from boltz_mlx_export.schema import ArtifactManifest


def test_selection_includes_confidence_and_distogram_but_not_affinity() -> None:
    """Confidence and distogram weights ARE exported; affinity and b-factor are not.

    Confidence is required for PAE (hence min_ipSAE, the design-loop gate), and distogram is a
    required input to it — boltz2.py feeds dict_out["pdistogram"] in as pred_distogram_logits, so
    exporting confidence without distogram yields a head that cannot run. Affinity needs a separate
    checkpoint and b-factor is unused, so both stay out.
    """
    state = {
        "s_init.weight": torch.ones(4, 4),
        "structure_module.score_model.out.weight": torch.ones(4, 4),
        "confidence_module.confidence_heads.to_pae_intra_logits.weight": torch.ones(4, 4),
        "distogram_module.distogram.weight": torch.ones(4, 4),
        "affinity_module.head.weight": torch.ones(4, 4),
        "bfactor_module.weight": torch.ones(4, 4),
    }

    selected = select_structure_state_dict(state)

    assert set(selected) == {
        "s_init.weight",
        "structure_module.score_model.out.weight",
        "confidence_module.confidence_heads.to_pae_intra_logits.weight",
        "distogram_module.distogram.weight",
    }


def test_shared_linear_aliases_are_all_quantized() -> None:
    """Lightning state dict aliases must not leave duplicate fp16 matrices."""
    model = torch.nn.Module()
    shared = torch.nn.Linear(4, 4)
    model.s_init = shared
    model.s_recycle = shared

    assert _eligible_matrix_names(model) == {
        "s_init.weight",
        "s_recycle.weight",
    }


def test_export_state_dict_writes_quantized_and_plain_tensors(tmp_path: Path) -> None:
    """A synthetic state produces a strict int8 model directory."""
    state = {
        "s_init.weight": torch.arange(128, dtype=torch.float32).reshape(2, 64),
        "s_norm.weight": torch.ones(2),
        "s_norm.bias": torch.zeros(2),
        "token_bonds.weight": torch.arange(2, dtype=torch.float32).reshape(2, 1),
        "bfactor_module.weight": torch.ones(2, 64),
    }

    manifest = export_state_dict(
        state,
        output=tmp_path,
        eligible_matrix_names={"s_init.weight", "token_bonds.weight"},
        source_checkpoint_sha256="c" * 64,
    )

    arrays = load_file(tmp_path / "model.safetensors")
    assert set(arrays) == {
        "s_init.biases",
        "s_init.scales",
        "s_init.weight",
        "s_norm.bias",
        "s_norm.weight",
        "token_bonds.biases",
        "token_bonds.scales",
        "token_bonds.weight",
    }
    assert arrays["s_init.weight"].dtype == np.uint32
    assert arrays["token_bonds.weight"].shape == (2, 16)
    assert arrays["s_norm.weight"].dtype == np.float16
    assert not any(name.startswith("confidence_module") for name in arrays)
    assert ArtifactManifest.read(tmp_path / "manifest.json") == manifest

    tensor_specs = {tensor.name: tensor for tensor in manifest.tensors}
    assert tensor_specs["token_bonds.weight"].logical_shape == (2, 1)
    assert tensor_specs["token_bonds.weight"].physical_shape == (2, 64)
    assert json.loads((tmp_path / "manifest.json").read_text())["quantization"] == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }


def test_model_configuration_round_trip_preserves_runtime_architecture(
    tmp_path: Path,
) -> None:
    """The Swift runtime receives architecture values rather than guessing them."""
    configuration = ModelConfiguration.from_hparams(
        {
            "atom_s": 128,
            "atom_z": 16,
            "token_s": 384,
            "token_z": 128,
            "num_bins": 64,
            "atom_feature_dim": 128,
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
            "fix_sym_check": True,
            "cyclic_pos_enc": False,
            "bond_type_feature": True,
            "use_templates": True,
            "use_templates_v2": True,
            "conditioning_cutoff_min": 4.0,
            "conditioning_cutoff_max": 20.0,
            "embedder_args": {
                "atom_encoder_depth": 3,
                "atom_encoder_heads": 4,
                "add_method_conditioning": True,
            },
            "msa_args": {
                "msa_s": 64,
                "msa_blocks": 4,
                "pairwise_head_width": 32,
                "pairwise_num_heads": 4,
                "use_paired_feature": True,
            },
            "pairformer_args": {"num_blocks": 64, "num_heads": 16, "v2": True},
            "score_model_args": {
                "dim_fourier": 256,
                "atom_encoder_depth": 3,
                "atom_encoder_heads": 4,
                "token_transformer_depth": 24,
                "token_transformer_heads": 8,
                "atom_decoder_depth": 3,
                "atom_decoder_heads": 4,
                "conditioning_transition_layers": 2,
            },
            "diffusion_process_args": {
                "num_sampling_steps": 5,
                "sigma_min": 0.0001,
                "sigma_max": 160.0,
                "sigma_data": 16.0,
                "rho": 7.0,
                "gamma_0": 0.8,
                "gamma_min": 1.0,
                "noise_scale": 1.003,
                "step_scale": 1.5,
            },
            "template_args": {
                "template_dim": 64,
                "template_blocks": 2,
                "pairwise_head_width": 32,
                "pairwise_num_heads": 4,
            },
        }
    )

    path = tmp_path / "config.json"
    configuration.write(path)

    assert ModelConfiguration.read(path) == configuration
    payload = json.loads(path.read_text())
    assert payload["pairformer_args"]["num_blocks"] == configuration.pairformer_args[
        "num_blocks"
    ]
    assert payload["score_model_args"]["token_transformer_depth"] == (
        configuration.score_model_args["token_transformer_depth"]
    )
