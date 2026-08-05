# SPDX-License-Identifier: Apache-2.0
# Attention module for the MiniMax H3 visual VAE (inference-only bundle).
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from diffusers.utils import logging

from .parallel import all_to_all_4D, get_parallel_state
from .func import apply_rotary_pos_emb
from .flash import flash_attn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _env_flag(name, default="0"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    weight = getattr(module, "weight", None)
    return hidden_states.to(getattr(weight, "dtype", hidden_states.dtype))


def maybe_checkpoint(owner, function, *args):
    if owner.training and getattr(owner, "gradient_checkpointing", False):
        raise NotImplementedError(
            "gradient checkpointing is not supported in this inference-only bundle"
        )
    return function(*args)


class Attention(nn.Module):
    def __init__(
        self,
        heads,
        dim_head,
        embed_dim: Optional[int] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_affine: bool = False,
        bias: bool = True,
        out_bias: Optional[bool] = None,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.attn_inner_dim = dim_head * heads
        self.embed_dim = embed_dim if embed_dim is not None else self.attn_inner_dim

        out_bias = out_bias if out_bias is not None else bias

        if qk_norm_type is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm_type == "layer_norm":
            self.norm_q = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.LayerNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        elif qk_norm_type == "rms_norm":
            self.norm_q = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
            self.norm_k = nn.RMSNorm(
                dim_head, eps=eps, elementwise_affine=qk_norm_affine
            )
        else:
            raise ValueError(
                f"unknown qk_norm_type: {qk_norm_type}. Should be None,'layer_norm','rms_norm'"
            )

        self.to_qkv = nn.Linear(self.embed_dim, self.attn_inner_dim * 3, bias=bias)

        self.to_out = nn.Linear(self.attn_inner_dim, self.embed_dim, bias=out_bias)

        self.spatial_parallel = get_parallel_state().get("sp_enabled", False)

        state = get_parallel_state()
        sp_size = state.get("sp_size", 1)
        tp_size = state.get("tp_size", 1)
        parallel_size = sp_size * tp_size
        if parallel_size > 1 and self.heads % parallel_size != 0:
            raise ValueError(
                f"num_heads {self.heads} must be divisible by sp_size * tp_size ({sp_size} * {tp_size} = {parallel_size})"
            )

        if len(kwargs) > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            logger.warning(f"Unused kwargs: {kwargs}")

    def _perform_attention(self, query, key, value, pack_info):
        cu_seqlens = pack_info.get("cu_seqlens", None)
        mask_mod = pack_info.get("mask_mod", None)
        block_sparse = pack_info.get("block_sparse", None)

        if cu_seqlens is not None:
            raise NotImplementedError(
                "varlen attention is not supported in this inference-only bundle"
            )

        if mask_mod is not None:
            hidden_states = flash_attn(
                query,
                key,
                value,
                mask_mod=mask_mod,
                block_sparse=block_sparse,
            )
        else:
            hidden_states = flash_attn(
                query,
                key,
                value,
            )

        return hidden_states

    def perform_attention(self, query, key, value, pack_info={}):
        return self._perform_attention(query, key, value, pack_info)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        pack_info: dict = {},
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.to_qkv(hidden_states)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        if self.spatial_parallel:
            local_process_group = get_parallel_state()["sp_process_group"]
            query = all_to_all_4D(query, 2, 1, group=local_process_group)
            key = all_to_all_4D(key, 2, 1, group=local_process_group)
            value = all_to_all_4D(value, 2, 1, group=local_process_group)

        if self.norm_q is not None:
            query = self.norm_q(_vit_norm_input(self.norm_q, query)).to(query.dtype)
        if self.norm_k is not None:
            key = self.norm_k(_vit_norm_input(self.norm_k, key)).to(key.dtype)

        if rotary_pos_emb is not None:
            query = apply_rotary_pos_emb(query, rotary_pos_emb)
            key = apply_rotary_pos_emb(key, rotary_pos_emb)

        hidden_states = self.perform_attention(query, key, value, pack_info)

        if self.spatial_parallel:
            hidden_states = all_to_all_4D(hidden_states, 1, 2, group=local_process_group)

        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states
