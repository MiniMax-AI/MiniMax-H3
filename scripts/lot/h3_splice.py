"""Target-video tail for an H3 DiT: embed, position and project the LoT tokens.

This module imports no model code. ``h3_musubi.predict(..., lot=splice)`` calls
it at three points of the forward (embedding, rotary positions, output head).

Latent space. The adapter works in y-space: each extent's clean patch is the
pretrained latent divided by that extent's scale ``s`` (eq. 16). Until a
Procrustes fit every ``s`` is 1 and y-space is the pretrained latent. After
``fit_extent`` it is not. Then the sampler state handed to the DiT must be
``y_t``, built from ``to_y(x0)``, and the clean estimate goes back through
``to_x``. A splice built with ``y_space=False`` refuses fitted scales instead
of silently feeding x-space rows to y-space heads.
"""

from __future__ import annotations

import torch

from adapter import LotVisualAdapter
from h3 import patchify, unpatchify
from h3_positions import video_positions
from layout import LotLayout


class LotSplice:
    """One layout and the adapter that embeds and scatters its video tail."""

    def __init__(self, adapter: LotVisualAdapter, layout: LotLayout, *, y_space: bool = False):
        self.adapter = adapter
        self.layout = layout
        self.y_space = bool(y_space)

    def unit_scales(self) -> bool:
        """True while every extent in this layout still has ``s = 1``."""
        used = {rect.extent for rect in self.layout.rects}
        return all(float(self.adapter.bank.scale(extent)) == 1.0 for extent in used)

    def to_y(self, x0: torch.Tensor) -> torch.Tensor:
        """Pretrained clean latent ``(B, C, T, H, W)`` to y-space, same shape."""
        tokens = patchify(x0)
        return unpatchify(self.adapter.bank.scale_clean(tokens, self.layout)).to(x0.dtype)

    def to_x(self, y0: torch.Tensor) -> torch.Tensor:
        """Inverse of ``to_y``: a y-space clean estimate back to the pretrained latent."""
        tokens = patchify(y0)
        return unpatchify(self.adapter.bank.unscale(tokens, self.layout)).to(y0.dtype)

    def video_tokens(self, video_rows: torch.Tensor, latent_t: int, lat_h: int, lat_w: int) -> torch.Tensor:
        """``(n, 96)`` patch rows from ``patchify_video`` to ``(1, T, H, W, 96)``."""
        if not self.y_space and not self.unit_scales():
            raise ValueError(
                "extent scales are fitted (s != 1), so the DiT input must be y_t. "
                "Build the state with to_y and pass LotSplice(..., y_space=True)."
            )
        token_h, token_w = lat_h // 2, lat_w // 2
        if lat_h % 2 or lat_w % 2:
            raise ValueError("latent sides must be divisible by the 2x2 patch")
        expected = latent_t * token_h * token_w
        if video_rows.ndim != 2 or video_rows.shape[0] != expected:
            raise ValueError(f"video rows {tuple(video_rows.shape)} != ({expected}, D)")
        if video_rows.shape[-1] != self.adapter.token_dim:
            raise ValueError(f"token dim {video_rows.shape[-1]} != {self.adapter.token_dim}")
        if self.layout.time != latent_t or self.layout.height != token_h or self.layout.width != token_w:
            raise ValueError(
                f"layout {(self.layout.time, self.layout.height, self.layout.width)} "
                f"does not match tokens {(latent_t, token_h, token_w)}"
            )
        return video_rows.reshape(1, latent_t, token_h, token_w, video_rows.shape[-1])

    def embed_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(L, hidden)`` for the packed sequence. The batch dimension is 1."""
        hidden, _centers = self.adapter.embed(tokens, self.layout)
        return hidden[0]

    def replace_video_positions(
        self,
        dense_pos: torch.Tensor,
        latent_t: int,
        lat_h: int,
        lat_w: int,
    ) -> torch.Tensor:
        """Keep the text, condition, and audio prefix. Replace the dense video tail."""
        n_dense = latent_t * (lat_h // 2) * (lat_w // 2)
        if dense_pos.shape[0] < n_dense:
            raise ValueError("position tensor is shorter than the dense video tail")
        origin = float(dense_pos[-n_dense, 0].item())
        tail = video_positions(self.layout, lat_h, lat_w, origin)
        return torch.cat([dense_pos[:-n_dense], tail.to(dtype=dense_pos.dtype)], dim=0)

    def modulate(self, final_layer, states: torch.Tensor, t_emb: torch.Tensor, t_index: int) -> torch.Tensor:
        """RMSNorm and AdaLN, the shared part of ``FinalLayer``. Not ``video_out``."""
        shift, scale = final_layer.adaln_proj(t_emb)
        return final_layer.norm(states) * (1.0 + scale[t_index]) + shift[t_index]

    def project_rows(self, modulated: torch.Tensor, tokens: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
        """Extent linears plus the dense velocity, flattened in patchify order.

        The result is H3's head convention, ``x0 - eps``, so H3's samplers
        steps it unchanged. It is in the same space as ``tokens``: y-space once
        scales are fitted.
        """
        weight = next(iter(self.adapter.out_proj.parameters()))
        states = modulated.to(dtype=weight.dtype, device=weight.device).unsqueeze(0)
        tokens = tokens.to(dtype=weight.dtype, device=weight.device)
        velocity = self.adapter.velocity_from_states(
            states, tokens, sigma, self.layout, x0_minus_eps=True)
        return velocity.reshape(-1, velocity.shape[-1])
