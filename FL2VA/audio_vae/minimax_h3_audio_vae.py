# SPDX-License-Identifier: Apache-2.0
# Remote entry: self-contained MiniMax H3 audio VAE (DAC-lineage encoder + BigVGAN decoder).
# Loaded via config.json:auto_map with trust_remote_code; weights are safetensors-only.
from __future__ import annotations

import json
from pathlib import Path

import torch.nn as nn

# --- dependency manifest ---
# diffusers' dynamic-module loader only copies ONE level of relative
# imports into its cache; list every bundle module here so all files
# are copied, letting their own second-level imports resolve.
from .dac_activations import SnakeBeta as _dep_dac_activations  # noqa: F401
from .dac_alias_free_act import Activation1d as _dep_dac_alias_free_act  # noqa: F401
from .dac_alias_free_filter import kaiser_sinc_filter1d as _dep_dac_alias_free_filter  # noqa: F401
from .dac_alias_free_resample import UpSample1d as _dep_dac_alias_free_resample  # noqa: F401
from .dac_attn_proj import GeGluMlp as _dep_dac_attn_proj  # noqa: F401
from .dac_bigvgan import AttrDict as _dep_dac_bigvgan  # noqa: F401
from .dac_audio_vae import AttrDict as _dep_dac_audio_vae  # noqa: F401
from .dac_utils import init_weights as _dep_dac_utils  # noqa: F401
# --- end dependency manifest ---
from safetensors.torch import load_file

from .dac_audio_vae import DacAudioVAE


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("MiniMax H3 audio VAE requires PyYAML.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MiniMaxH3AudioVAE(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        component_dir = Path(pretrained_model_name_or_path)
        with (component_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        audio_config = _load_yaml(component_dir / config["source_config_path"])
        if "source_safetensors_path" not in config:
            raise KeyError(
                "source_safetensors_path is required; pickle checkpoints are not supported"
            )
        if "source_metadata_path" not in config:
            raise KeyError(
                "source_metadata_path is required when source_safetensors_path is set"
            )
        state_dict = load_file(
            component_dir / config["source_safetensors_path"], device="cpu"
        )
        with (component_dir / config["source_metadata_path"]).open(
            "r", encoding="utf-8"
        ) as f:
            metadata_doc = json.load(f)
        metadata = metadata_doc["metadata"]["kwargs"]

        model = DacAudioVAE(
            encoder_rates=metadata["encoder_rates"],
            decoder_rates=metadata["decoder_rates"],
            attn_proj=metadata["attn_proj"],
            decoder_type=metadata["decoder_type"],
            decoder_dim=audio_config["model_config"]["decoder_dim"],
            vae_latent_channels=audio_config["model_config"]["vae_latent_channels"],
            sample_rate=metadata["sample_rate"],
        )
        model.load_state_dict(state_dict, strict=True)
        return cls(model.eval())

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
