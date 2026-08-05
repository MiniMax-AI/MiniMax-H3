# SPDX-License-Identifier: Apache-2.0
# Remote entry: self-contained MiniMax H3 visual VAE (3D CNN encoder + ViT3D decoder).
# Loaded via config.json:auto_map with trust_remote_code.
from __future__ import annotations

import json
from pathlib import Path

import safetensors.torch
import torch.nn as nn

# --- dependency manifest ---
# diffusers' dynamic-module loader only copies ONE level of relative
# imports into its cache; list every bundle module here so all files
# are copied, letting their own second-level imports resolve.
from .attention import Attention as _dep_attention  # noqa: F401
from .base_module import FeedForward as _dep_base_module  # noqa: F401
from .conv import SpatialParallelConv3d as _dep_conv  # noqa: F401
from .flash import make_block_causal_mask_mod as _dep_flash  # noqa: F401
from .func import create_token_ids as _dep_func  # noqa: F401
from .klvae import AutoencoderKL as _dep_klvae  # noqa: F401
from .norm import FusedGroupNorm3D as _dep_norm  # noqa: F401
from .normalize import get_norm_constants as _dep_normalize  # noqa: F401
from .parallel import get_parallel_state as _dep_parallel  # noqa: F401
from .utils import apply_spatial_parallel as _dep_utils  # noqa: F401
from .vae_cnn import EncoderFCN3D as _dep_vae_cnn  # noqa: F401
from .vae_module import DiagonalGaussianDistribution as _dep_vae_module  # noqa: F401
from .vae_processor import VAEProcessor as _dep_vae_processor  # noqa: F401
from .vae_vit import ViTBase as _dep_vae_vit  # noqa: F401
# --- end dependency manifest ---

from .klvae import AutoencoderKLLegacy
from .parallel import get_parallel_state

_SOURCE_CLASSES = {
    "AutoencoderKLLegacy": AutoencoderKLLegacy,
}


def _ensure_vae_parallel_state() -> None:
    """Seed the bundled VAE parallel state for single-process inference."""
    state = get_parallel_state()
    if not isinstance(state, dict):
        raise TypeError("get_parallel_state() must return a dict")
    if state:
        return
    state.update(
        {
            "group_size": 1,
            "group_rank": 0,
            "local_process_group": None,
            "sp_size": 1,
            "sp_rank": 0,
            "sp_enabled": False,
            "sp_process_group": None,
            "tp_size": 1,
            "tp_rank": 0,
        }
    )


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        component_dir = Path(pretrained_model_name_or_path)
        with (component_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        source_path = component_dir / config["source_path"]
        source_class_name = config["source_class_name"]
        if source_class_name not in _SOURCE_CLASSES:
            raise ValueError(
                f"unsupported source_class_name {source_class_name!r}; "
                f"bundled: {sorted(_SOURCE_CLASSES)}"
            )
        source_cls = _SOURCE_CLASSES[source_class_name]
        if "source_safetensors_path" not in config:
            raise ValueError(
                "source_safetensors_path is required; pickle checkpoints are "
                "not supported"
            )
        weights_path = source_path / config["source_safetensors_path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"source weights not found: {weights_path}")
        if bool(config["vae_parallel_tiling"]):
            _ensure_vae_parallel_state()
        load_kwargs = {
            "clip_length": int(config["vae_clip_length"]),
            "token_drop": int(config["vae_token_drop"]),
            "encoder_tiling": int(config["vae_encoder_tiling"]),
            "decoder_tiling": int(config["vae_decoder_tiling"]),
            "parallel_tiling": int(config["vae_parallel_tiling"]),
            "tile_size": int(config["vae_tile_size"]),
            "tile_overlap_min": int(config["vae_tile_overlap_min"]),
            "encoder_parallel": int(config["vae_encoder_parallel"]),
            "decoder_parallel": int(config["vae_decoder_parallel"]),
            "chunk_dim": int(config["vae_chunk_dim"]),
        }
        # Mirror diffusers ModelMixin.from_pretrained instantiation semantics
        # (config-driven init via from_config) but load the state dict from an
        # explicitly named safetensors file instead of the diffusers default
        # weight filename.
        source_config = source_cls.load_config(str(source_path))
        model, _unused = source_cls.from_config(
            source_config, return_unused_kwargs=True, **load_kwargs
        )
        state_dict = safetensors.torch.load_file(str(weights_path))
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return cls(model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
