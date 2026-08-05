# SPDX-License-Identifier: Apache-2.0
# ViT3D decoder for the MiniMax H3 visual VAE (inference-only bundle).
import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging

from .attention import maybe_checkpoint
from .base_module import TransformerBlock, RotaryEmbeddingND
from .flash import make_block_causal_mask_mod
from .func import create_token_ids
from .parallel import get_subseq, gather_subseq, get_parallel_state

logger = logging.get_logger(__name__)


def _linear_with_module_dtype(linear, tensor, out_dtype=None):
    weight = getattr(linear, "weight", None)
    target_dtype = getattr(weight, "dtype", tensor.dtype)
    output = linear(tensor.to(target_dtype))
    if out_dtype is not None and output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output


def _make_seq_len_mask_mod(seq_len, base_mask_mod=None):
    if base_mask_mod is None:
        def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
            return (q_idx < seq_len) & (kv_idx < seq_len)

        mask_mod.block_sparse_cache_key = ("seq_len", seq_len)
        return mask_mod

    def mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors):
        return (
            (q_idx < seq_len)
            & (kv_idx < seq_len)
            & base_mask_mod(b, h, q_idx, kv_idx, seqlen_info, aux_tensors)
        )

    base_cache_key = getattr(base_mask_mod, "block_sparse_cache_key", None)
    if base_cache_key is not None:
        mask_mod.block_sparse_cache_key = ("seq_len", seq_len, base_cache_key)
    if hasattr(base_mask_mod, "use_fast_sampling"):
        mask_mod.use_fast_sampling = base_mask_mod.use_fast_sampling
    return mask_mod






def _pack_tensors_3d(tensors, patch_size, patch_size_t):
    batch_size, num_channels_tensors, temporal, height, width = tensors.shape

    tensors = tensors.view(
        batch_size,
        num_channels_tensors,
        temporal // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    tensors = tensors.permute(0, 2, 4, 6, 1, 3, 5, 7)
    tensors = tensors.reshape(
        batch_size,
        (temporal // patch_size_t) * (height // patch_size) * (width // patch_size),
        num_channels_tensors * patch_size_t * patch_size * patch_size,
    )
    return tensors


def _unpack_tensors_3d(tensors, patch_size, patch_size_t, temporal, height, width):
    batch_size, num_patches, channels = tensors.shape
    num_channels_tensors = channels // (patch_size_t * patch_size * patch_size)

    tensors = tensors.view(
        batch_size,
        temporal // patch_size_t,
        height // patch_size,
        width // patch_size,
        num_channels_tensors,
        patch_size_t,
        patch_size,
        patch_size,
    )
    tensors = tensors.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    tensors = tensors.reshape(batch_size, num_channels_tensors, temporal, height, width)
    return tensors


class ViTBase(ModelMixin, ConfigMixin):
    """Base class for ViT Encoder and Decoder with common functionality."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    gradient_checkpointing_mode = "full"

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def set_spatial_parallel(self, enabled):
        self.spatial_parallel = enabled
        if hasattr(self, "transformer_blocks"):
            for block in self.transformer_blocks:
                block.attn.spatial_parallel = enabled

    def _init_weights(self):
        def basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(basic_init)

    def init_mask_config(self, dim, is_3d=False):
        self._mask_dim = dim
        self._mask_is_3d = is_3d
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))

    def set_mask_config(self, mask_config):
        self.mask_prob = mask_config.get("mask_prob", 0.0)
        self.mask_enabled = self.mask_prob > 0
        self.mask_style = mask_config.get("mask_style", "replace")
        if self.mask_enabled and self.mask_style == "drop" and self.mask_prob < 1.0:
            logger.warning("mask_style='drop' with mask_prob < 1.0")
        if self._mask_is_3d:
            self.temporal_scale_range = mask_config.get("temporal_scale_range", (0.3, 0.5))
            self.spatial_scale_range = mask_config.get("spatial_scale_range", (0.1, 0.25))
            self.min_mask_ratio = mask_config.get("min_mask_ratio", 0.75)
            self.max_mask_ratio = mask_config.get("max_mask_ratio", 0.95)
        else:
            self.spatial_scale_range = mask_config.get("spatial_scale_range", (0.15, 0.15))
            self.min_mask_ratio = mask_config.get("min_mask_ratio", 0.5)
            self.max_mask_ratio = mask_config.get("max_mask_ratio", 0.75)
        self.aspect_ratio_range = mask_config.get("aspect_ratio_range", (0.75, 1.5))
        self.max_retries = mask_config.get("max_retries", 100)
        if self.mask_enabled and self.mask_style == "drop" and getattr(self, "t_causal", False):
            logger.warning("mask_style='drop' with t_causal may cause issues")
        if self.mask_enabled and "mask_token" in self._buffers:
            del self._buffers["mask_token"]
            self.mask_token = nn.Parameter(torch.randn(1, 1, self._mask_dim) * 0.02)

    def init_suffix_tokens(self, dim, num_register_tokens, has_cls_token=True):
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, dim) * 0.02)
        else:
            self.register_tokens = None
        if has_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def apply_mask_preprocess(self, hidden_states, img_ids, patch_dims, num_suffix):
        if self.training and self.mask_enabled:
            raise NotImplementedError(
                "mask modeling is not supported in this inference-only bundle"
            )
        return hidden_states, img_ids

    def forward_transformer_blocks(self, hidden_states, rotary_pos_emb, pack_info=None):
        if pack_info is None:
            pack_info = {}
        for block in self.transformer_blocks:
            hidden_states = maybe_checkpoint(
                self, block, hidden_states, rotary_pos_emb, pack_info
            )
        return hidden_states

    def _pad_for_sp(self, hidden_states, img_ids, pack_info=None):
        if pack_info is None:
            pack_info = {}
        if not self.spatial_parallel:
            return hidden_states, img_ids, pack_info, 0

        seq_len = hidden_states.shape[1]
        sp_size = get_parallel_state().get("sp_size", 1)
        pad_len = (-seq_len) % sp_size
        if pad_len == 0:
            return hidden_states, img_ids, pack_info, 0

        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len))
        img_ids = torch.nn.functional.pad(img_ids, (0, 0, 0, pad_len))

        pack_info = dict(pack_info)
        base_mask_mod = pack_info.get("mask_mod")
        pack_info["mask_mod"] = _make_seq_len_mask_mod(seq_len, base_mask_mod)
        pack_info.pop("block_sparse", None)
        return hidden_states, img_ids, pack_info, pad_len

    @staticmethod
    def _unpad_for_sp(hidden_states, pad_len):
        if pad_len == 0:
            return hidden_states
        return hidden_states[:, :-pad_len, :]

    def apply_mask_postprocess(self, hidden_states, num_patches):
        if self.training and self.mask_enabled and self.mask_style == "drop":
            raise NotImplementedError(
                "mask modeling is not supported in this inference-only bundle"
            )
        return hidden_states








class ViT3DDecoder(ViTBase):
    """Vision Transformer Video Decoder using TransformerBlock."""

    @register_to_config
    def __init__(
        self,
        patch_size: int = 16,
        patch_size_t: int = 4,
        t_causal: bool = False,
        in_channels: int = 16,
        out_channels: int = 3,
        num_layers: int = 24,
        heads: int = 16,
        dim_head: int = 64,
        norm_type: str = "layer_norm",
        norm_affine: bool = True,
        qk_norm_type: str = None,
        qk_norm_affine: bool = False,
        ffn_activation_fn: str = "gelu",
        ffn_use_gated: bool = False,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 1.0,
        bias: bool = True,
        eps: float = 1e-5,
        num_register_tokens: int = 4,
        mask_config: dict = {},
        **kwargs,
    ):
        super().__init__()

        dim = heads * dim_head
        rope_apply_dim = int(dim_head * rope_dim_ratio)

        self.pos_embed = RotaryEmbeddingND(rope_apply_dim, rope_theta, n_dim=3, use_angle=True)

        self.x_embedder = nn.Linear(in_channels, dim)

        self.init_suffix_tokens(dim, num_register_tokens, has_cls_token=False)

        self.t_causal = t_causal

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    heads=heads,
                    dim_head=dim_head,
                    norm_type=norm_type,
                    norm_affine=norm_affine,
                    qk_norm_type=qk_norm_type,
                    qk_norm_affine=qk_norm_affine,
                    ffn_activation_fn=ffn_activation_fn,
                    ffn_use_gated=ffn_use_gated,
                    bias=bias,
                    eps=eps,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )

        self.spatial_parallel = False
        for block in self.transformer_blocks:
            block.attn.spatial_parallel = False

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=norm_affine, eps=eps)
        patch_dim = out_channels * patch_size_t * patch_size * patch_size
        self.proj_out = nn.Linear(dim, patch_dim)

        self.init_mask_config(dim, is_3d=True)
        self.set_mask_config(mask_config)

        self._init_weights()
        self.gradient_checkpointing = False

        if len(kwargs) > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.warning(f"Unused kwargs: {kwargs}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss_info = {}

        B, C, latent_T, latent_H, latent_W = x.shape
        patch_size = self.config.patch_size
        patch_size_t = self.config.patch_size_t
        num_suffix = 1 + self.num_register_tokens

        hidden_states = _pack_tensors_3d(x, 1, 1)
        latent_size = (latent_T, latent_H, latent_W)

        with torch.autocast("cuda", enabled=False):
            hidden_states = _linear_with_module_dtype(self.x_embedder, hidden_states, hidden_states.dtype)

        num_patches = hidden_states.shape[1]

        tokens = [hidden_states]

        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(B, -1, -1)
            tokens.append(register_tokens)

        cls_token = torch.zeros_like(hidden_states[:, 0:1, :])
        tokens.append(cls_token)
        hidden_states = torch.cat(tokens, dim=1)

        patch_dims = [latent_T, latent_H, latent_W]
        img_ids = create_token_ids(latent_size, x.device, x.dtype).expand(B, -1, -1)
        suffix_ids = torch.zeros((B, num_suffix, 3), device=x.device, dtype=img_ids.dtype)
        img_ids = torch.cat([img_ids, suffix_ids], dim=1)

        hidden_states, img_ids = self.apply_mask_preprocess(hidden_states, img_ids, patch_dims, num_suffix)

        pack_info = {}
        if self.t_causal:
            spatial_size = latent_H * latent_W
            mask_mod = make_block_causal_mask_mod(
                num_tokens=num_patches,
                block_size=spatial_size,
                suffix=True,
            )
            pack_info["mask_mod"] = mask_mod

        hidden_states, img_ids, pack_info, sp_pad_len = self._pad_for_sp(hidden_states, img_ids, pack_info)

        rotary_pos_emb = self.pos_embed(img_ids)

        if self.spatial_parallel:
            hidden_states = get_subseq(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = maybe_checkpoint(
                self, block, hidden_states, rotary_pos_emb, pack_info
            )

        if self.spatial_parallel:
            hidden_states = gather_subseq(hidden_states)
        hidden_states = self._unpad_for_sp(hidden_states, sp_pad_len)

        hidden_states = self.norm_out(hidden_states)

        hidden_states = self.apply_mask_postprocess(hidden_states, num_patches)

        with torch.autocast("cuda", enabled=False):
            output = _linear_with_module_dtype(self.proj_out, hidden_states, hidden_states.dtype)

        output = output[:, :num_patches, :]

        video_t = latent_size[0] * patch_size_t
        video_h = latent_size[1] * patch_size
        video_w = latent_size[2] * patch_size
        output = _unpack_tensors_3d(output, patch_size, patch_size_t, video_t, video_h, video_w)

        return output














