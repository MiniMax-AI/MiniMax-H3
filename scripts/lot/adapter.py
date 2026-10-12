"""Level-of-Token DiT adapter.

The backbone still runs attention. This module replaces only the visual
token pack, the positional centers, and the velocity head:

    tokens (B, T, H, W, D) in scaled y-space
        -> per-extent input head + shape MLP          (eqs. 11–12)
        -> backbone(hidden, centers, sigma) -> hidden
        -> per-extent output head                      (eq. 10)
        -> LakonLab asymmetric-velocity recovery       (eq. 9)
        -> dense velocity on the uniform lattice

Text, audio, and the transformer's own blocks stay inside ``backbone``.
It must return one hidden vector per LoT token, in ``layout.rects`` order.
"""

from __future__ import annotations

import torch
from torch import nn

from flow import (
    apply_extent_scales,
    asymmetric_target,
    fit_procrustes,
    gather_extent,
    mean_basis,
    recover_dense_velocity,
    scatter_extent,
)
from layout import LotLayout, shape_features


def _key(extent: tuple[int, int, int]) -> str:
    et, eh, ew = extent
    return f"{et}x{eh}x{ew}"


class ExtentBank(nn.Module):
    """Fixed per-extent lifts ``A`` and data scales ``s``.

    Bases start as the mean lift and stay frozen. ``fit_`` replaces one
    extent with a Procrustes fit; training does not update these buffers.
    """

    def __init__(self, token_dim: int, extents: list[tuple[int, int, int]]):
        super().__init__()
        if not extents:
            raise ValueError("at least one extent is required")
        self.token_dim = int(token_dim)
        self.extents = [tuple(map(int, extent)) for extent in extents]
        seen = set()
        for extent in self.extents:
            if extent in seen:
                raise ValueError(f"duplicate extent {extent}")
            seen.add(extent)
            if min(extent) < 1:
                raise ValueError(f"bad extent {extent}")
            key = _key(extent)
            self.register_buffer(f"A_{key}", mean_basis(self.token_dim, extent))
            self.register_buffer(f"s_{key}", torch.ones(()))

    def basis(self, extent: tuple[int, int, int]) -> torch.Tensor:
        return getattr(self, f"A_{_key(extent)}")

    def scale(self, extent: tuple[int, int, int]) -> torch.Tensor:
        return getattr(self, f"s_{_key(extent)}")

    def scales(self) -> dict[tuple[int, int, int], torch.Tensor]:
        return {extent: self.scale(extent) for extent in self.extents}

    def dense_dim(self, extent: tuple[int, int, int]) -> int:
        et, eh, ew = extent
        return self.token_dim * et * eh * ew

    @torch.no_grad()
    def fit_(self, extent: tuple[int, int, int], dense: torch.Tensor, reference: torch.Tensor) -> None:
        """Replace ``A`` and ``s`` for one extent. Inputs are detached."""
        if tuple(extent) not in self.extents:
            raise KeyError(f"extent {extent} is not in this bank")
        if dense.shape[-1] != self.dense_dim(extent) or reference.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected dense (N, {self.dense_dim(extent)}) and reference (N, {self.token_dim})"
            )
        basis, scale = fit_procrustes(dense, reference)
        slot = self.basis(extent)
        slot.copy_(basis.to(device=slot.device, dtype=slot.dtype))
        scale_slot = self.scale(extent)
        scale_slot.copy_(scale.to(device=scale_slot.device, dtype=scale_slot.dtype))

    def scale_clean(self, x0: torch.Tensor, layout: LotLayout) -> torch.Tensor:
        """Eq. 16. ``x0`` is ``(B, T, H, W, D)`` in the pretrained latent space."""
        self._check_layout(layout)
        return apply_extent_scales(x0, layout, self.scales(), invert=False)

    def unscale(self, y0: torch.Tensor, layout: LotLayout) -> torch.Tensor:
        """Map a y-space clean estimate back to the pretrained latent."""
        self._check_layout(layout)
        return apply_extent_scales(y0, layout, self.scales(), invert=True)

    def _check_layout(self, layout: LotLayout) -> None:
        missing = sorted({rect.extent for rect in layout.rects} - set(self.extents))
        if missing:
            raise KeyError(f"layout uses extents with no basis: {missing}")


class LotVisualAdapter(nn.Module):
    """Extent heads and shape MLP wrapped around an external backbone."""

    def __init__(
        self,
        token_dim: int,
        hidden_size: int,
        extents: list[tuple[int, int, int]],
        *,
        include_time: bool = False,
        shape_hidden: int = 256,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_size = int(hidden_size)
        self.include_time = bool(include_time)
        self.bank = ExtentBank(self.token_dim, extents)
        feat_dim = 5 if self.include_time else 4
        width = max(feat_dim, int(shape_hidden))
        self.shape_mlp = nn.Sequential(
            nn.Linear(feat_dim, width),
            nn.SiLU(),
            nn.Linear(width, self.hidden_size),
        )
        nn.init.zeros_(self.shape_mlp[-1].weight)
        nn.init.zeros_(self.shape_mlp[-1].bias)
        self.in_proj = nn.ModuleDict()
        self.out_proj = nn.ModuleDict()
        for extent in self.bank.extents:
            dense_dim = self.bank.dense_dim(extent)
            self.in_proj[_key(extent)] = nn.Linear(dense_dim, self.hidden_size)
            self.out_proj[_key(extent)] = nn.Linear(self.hidden_size, dense_dim)
        self.register_buffer("pretrained_in", torch.empty(self.hidden_size, self.token_dim))
        self.register_buffer("pretrained_out", torch.empty(self.token_dim, self.hidden_size))
        self.register_buffer("pretrained_bias_in", torch.empty(self.hidden_size))
        self.register_buffer("pretrained_bias_out", torch.empty(self.token_dim))
        self.register_buffer("pretrained_ready", torch.zeros((), dtype=torch.int32))

    @torch.no_grad()
    def init_from_pretrained(
        self,
        weight_in: torch.Tensor,
        weight_out: torch.Tensor,
        bias_in: torch.Tensor | None = None,
        bias_out: torch.Tensor | None = None,
    ) -> None:
        """Initialize heads as ``W_in,e = W_in A^T`` and ``W_e = A W_out``.

        ``weight_in`` is ``(hidden, D)`` and ``weight_out`` is ``(D, hidden)``,
        matching ``nn.Linear`` layout. Output bias is lifted by ``A`` so a
        unit extent reproduces the pretrained bias and a larger extent starts
        on the same subspace.
        """
        hidden, token_dim = self.hidden_size, self.token_dim
        if tuple(weight_in.shape) != (hidden, token_dim):
            raise ValueError(f"weight_in shape {tuple(weight_in.shape)} != {(hidden, token_dim)}")
        if tuple(weight_out.shape) != (token_dim, hidden):
            raise ValueError(f"weight_out shape {tuple(weight_out.shape)} != {(token_dim, hidden)}")
        self.pretrained_in.copy_(weight_in.detach().to(dtype=self.pretrained_in.dtype, device=self.pretrained_in.device).clone())
        self.pretrained_out.copy_(weight_out.detach().to(dtype=self.pretrained_out.dtype, device=self.pretrained_out.device).clone())
        if bias_in is None:
            self.pretrained_bias_in.zero_()
        else:
            self.pretrained_bias_in.copy_(bias_in.detach().to(dtype=self.pretrained_bias_in.dtype, device=self.pretrained_bias_in.device).clone())
        if bias_out is None:
            self.pretrained_bias_out.zero_()
        else:
            self.pretrained_bias_out.copy_(bias_out.detach().to(dtype=self.pretrained_bias_out.dtype, device=self.pretrained_bias_out.device).clone())
        self.pretrained_ready.fill_(1)
        for extent in self.bank.extents:
            key = _key(extent)
            in_layer = self.in_proj[key]
            out_layer = self.out_proj[key]
            basis = self.bank.basis(extent).to(dtype=in_layer.weight.dtype, device=in_layer.weight.device)
            weight_in_m = weight_in.to(dtype=in_layer.weight.dtype, device=in_layer.weight.device)
            weight_out_m = weight_out.to(dtype=out_layer.weight.dtype, device=out_layer.weight.device)
            in_layer.weight.copy_(weight_in_m @ basis.T)
            out_layer.weight.copy_(basis @ weight_out_m)
            if bias_in is None:
                in_layer.bias.zero_()
            else:
                in_layer.bias.copy_(bias_in.to(dtype=in_layer.bias.dtype, device=in_layer.bias.device))
            if bias_out is None:
                out_layer.bias.zero_()
            else:
                lifted = basis @ bias_out.to(dtype=basis.dtype, device=basis.device)
                out_layer.bias.copy_(lifted.to(dtype=out_layer.bias.dtype))

    @torch.no_grad()
    def fit_extent(self, extent: tuple[int, int, int], dense: torch.Tensor, reference: torch.Tensor) -> None:
        """Fit one basis, then rebuild every head from the stored pretrained maps.

        ``fit_`` on the bank alone updates ``A`` and leaves the heads on the
        previous subspace. Call this instead once ``init_from_pretrained`` has run.
        """
        self.bank.fit_(extent, dense, reference)
        if int(self.pretrained_ready) != 1:
            return
        self.init_from_pretrained(
            self.pretrained_in,
            self.pretrained_out,
            self.pretrained_bias_in,
            self.pretrained_bias_out,
        )

    def embed(self, tokens: torch.Tensor, layout: LotLayout) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(B, L, hidden)`` and ``(L, 3)`` finest-grid centers."""
        self._check_tokens(tokens, layout)
        hidden = tokens.new_zeros(tokens.shape[0], layout.count, self.hidden_size)
        for extent, indices in layout.groups():
            rects = [layout.rects[index] for index in indices]
            dense = gather_extent(tokens, rects)
            projected = self.in_proj[_key(extent)](dense)
            features = shape_features(extent, self.include_time).to(device=tokens.device, dtype=tokens.dtype)
            projected = projected + self.shape_mlp(features)
            hidden[:, indices] = projected
        centers = layout.centers().to(device=tokens.device)
        return hidden, centers

    def velocity_from_states(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
        sigma: torch.Tensor | float,
        layout: LotLayout,
        *,
        x0_minus_eps: bool = False,
    ) -> torch.Tensor:
        """Extent heads plus eq. 9, scattered onto the uniform lattice.

        Eq. 9 assumes the head predicts ``P eps - x0`` and returns ``eps - x0``.
        H3's head predicts the negation (``x0 - eps``; H3's scheduler steps ``x0 = x_t + sigma * v``).
        ``x0_minus_eps=True`` flips the head before recovery and the velocity
        after, so the result stays in the caller's convention. A unit extent has
        no complement, so 1x1 parity cannot see a wrong sign here.
        """
        sign = -1.0 if x0_minus_eps else 1.0
        self._check_tokens(tokens, layout)
        if states.shape != (tokens.shape[0], layout.count, self.hidden_size):
            raise ValueError(
                f"backbone returned {tuple(states.shape)}, expected "
                f"{(tokens.shape[0], layout.count, self.hidden_size)}"
            )
        velocity = tokens.new_zeros(tokens.shape)
        for extent, indices in layout.groups():
            rects = [layout.rects[index] for index in indices]
            u_a = sign * self.out_proj[_key(extent)](states[:, indices])
            x_t = gather_extent(tokens, rects)
            recovered = recover_dense_velocity(u_a, x_t, self.bank.basis(extent), sigma)
            scatter_extent(velocity, sign * recovered, rects)
        return velocity

    def forward(self, tokens, sigma, layout: LotLayout, backbone, backbone_kwargs=None):
        """Dense y-space velocity. ``backbone(hidden, centers, sigma, **kwargs)``."""
        hidden, centers = self.embed(tokens, layout)
        kwargs = {} if backbone_kwargs is None else backbone_kwargs
        states = backbone(hidden, centers, sigma, **kwargs)
        return self.velocity_from_states(states, tokens, sigma, layout)

    def asymmetric_targets(
        self,
        y0: torch.Tensor,
        noise: torch.Tensor,
        layout: LotLayout,
    ) -> list[tuple[list[int], torch.Tensor]]:
        """Per-extent eq. 8 targets, for tests and for a direct head loss."""
        self.bank._check_layout(layout)
        targets = []
        for extent, indices in layout.groups():
            rects = [layout.rects[index] for index in indices]
            clean = gather_extent(y0, rects)
            eps = gather_extent(noise, rects)
            targets.append((indices, asymmetric_target(clean, eps, self.bank.basis(extent))))
        return targets

    def _check_tokens(self, tokens: torch.Tensor, layout: LotLayout) -> None:
        expected = (layout.time, layout.height, layout.width, self.token_dim)
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"tokens shape {tuple(tokens.shape)} != (B, {expected[0]}, {expected[1]}, {expected[2]}, {expected[3]})")
        self.bank._check_layout(layout)
