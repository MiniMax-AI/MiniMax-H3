"""Mixed-resolution additions to LoT, adapted from MPEG-Diffusion (Menn et al. 2026).

MPEG-Diffusion: Mixed-Resolution Video Generation, Menn, Gu, He, V S, Song, Liu,
Xie, Huang (Roblox), Xu and Marculescu (UT Austin), arXiv:2611.12069. That paper
keeps two token grids (HR inside a mask, LR elsewhere). LoT has one sequence of
mixed extents. The three pieces here carry its ideas over to that sequence:

1. ``DualKV`` (their dual-resolution attention, Sec. 3). Their Proposition 1:
   with one shared K/V, a coarse token can stand in for ``n`` fine tokens under
   softmax only when those fine tokens are identical. So 1x1 queries here attend
   to a *dense* key/value view: every coarse token's key and value are expanded
   to its ``n`` cells by ``f_up(k) = k + U V^T k`` (their "linear delta":
   exact copy plus a rank-``r`` correction, zero at init), before RoPE, and the
   expanded keys are rotated at the cells' dense positions. Every other query
   (text, audio, coarse tokens) keeps LoT's native attention, which already sees
   each fine token individually, so no ``f_down`` is needed. With zero deltas a
   fine query sees a coarse token as ``n`` copies at its cells.
2. ``RegionHead`` (their self-selected HR region, Sec. 3.4): ``s = w^T LN(h)``
   on a middle block's video tokens, smoothed in space and time, thresholded at
   ``m + alpha (p - m)`` per video (median ``m``, 99th percentile ``p``), all
   coarse when ``p`` is under a floor. The mask becomes a per-latent-frame LoT
   layout. Labels here are self-supervised in latent space (``region_labels``):
   tokens where the coarse layout's clean estimate is furthest from the dense
   model's, a latent stand-in for their per-patch LPIPS label.
3. ``SeamMixer`` (their mixed-resolution decoder's mixer ``G``, eq. 6): a small
   3D-conv residual on the final latent, applied outside the fine region only,
   zero-initialised so it starts as the identity. LoT already returns a dense
   latent, so there is no upscaler; the residual targets coarse-token blocking
   and the seams between extents.

Nothing here imports model code; ``h3_musubi`` wires it into H3's forward.
"""

from __future__ import annotations

import contextlib
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from layout import LotLayout, smooth_detail
from layout_sources import layout_from_desired_any


# --------------------------------------------------------------------- 2. dual K/V


class DualKVPlan:
    """Index tensors for one layout: which video rows are 1x1, and how to assemble the dense view.

    The expanded order is: the 1x1 tokens (in layout order), then for each larger extent its
    tokens, each followed by its ``eh * ew`` cells in row-major order. ``dense_order`` permutes
    that into dense ``(t, u, v)`` order, the order of H3's dense video positions.
    """

    def __init__(self, layout: LotLayout):
        self.layout = layout
        fine, groups = [], []
        cells = []
        for index, rect in enumerate(layout.rects):
            if rect.extent == (1, 1, 1):
                fine.append(index)
        width, plane = layout.width, layout.height * layout.width
        for index in fine:
            rect = layout.rects[index]
            cells.append(rect.t * plane + rect.u * width + rect.v)
        for extent, indices in layout.groups():
            if extent == (1, 1, 1):
                continue
            groups.append((extent, torch.tensor(indices, dtype=torch.long)))
            for index in indices:
                rect = layout.rects[index]
                for du in range(rect.eh):
                    for dv in range(rect.ew):
                        cells.append(rect.t * plane + (rect.u + du) * width + rect.v + dv)
        cells = torch.tensor(cells, dtype=torch.long)
        if cells.numel() != layout.dense_count or torch.unique(cells).numel() != layout.dense_count:
            raise ValueError("layout does not cover the dense grid exactly once")
        self.fine_rows = torch.tensor(fine, dtype=torch.long)
        fine_mask = torch.zeros(layout.count, dtype=torch.bool)
        fine_mask[self.fine_rows] = True
        self.other_rows = torch.nonzero(~fine_mask).squeeze(1)
        self.groups = groups
        self.dense_order = torch.argsort(cells)

    def to(self, device) -> "DualKVPlan":
        self.fine_rows = self.fine_rows.to(device)
        self.other_rows = self.other_rows.to(device)
        self.dense_order = self.dense_order.to(device)
        self.groups = [(extent, rows.to(device)) for extent, rows in self.groups]
        return self


def _key(layer: int, extent: tuple[int, int, int], which: str) -> str:
    return f"l{layer}_{extent[1]}x{extent[2]}_{which}"


class UpMaps(nn.Module):
    """``f_up`` per block, extent, and K or V: ``k -> k + (k V) U`` reshaped to ``n`` cells.

    ``V`` is ``(d, r)``, ``U`` is ``(r, n * d)``, shared by all heads of a block (the paper fits
    one map per layer). ``U`` starts at zero, so the map starts as an exact copy.
    """

    def __init__(self, layers: int, head_dim: int, extents, rank: int = 16):
        super().__init__()
        self.head_dim, self.rank = int(head_dim), int(rank)
        self.extents = [tuple(map(int, e)) for e in extents if tuple(e) != (1, 1, 1)]
        self.down = nn.ParameterDict()
        self.up = nn.ParameterDict()
        for layer in range(layers):
            for extent in self.extents:
                cells = extent[1] * extent[2]
                for which in ("k", "v"):
                    name = _key(layer, extent, which)
                    down = torch.empty(self.head_dim, self.rank)
                    nn.init.normal_(down, std=1.0 / math.sqrt(self.head_dim))
                    self.down[name] = nn.Parameter(down)
                    self.up[name] = nn.Parameter(torch.zeros(self.rank, cells * self.head_dim))

    def expand(self, layer: int, extent: tuple[int, int, int], which: str, rows: torch.Tensor) -> torch.Tensor:
        """``(B, N, H, d)`` -> ``(B, N * n, H, d)``: each token, then its cells row-major."""
        name = _key(layer, extent, which)
        cells = extent[1] * extent[2]
        batch, count, heads, dim = rows.shape
        delta = (rows @ self.down[name].to(rows.dtype)) @ self.up[name].to(rows.dtype)   # (B, N, H, n*d)
        delta = delta.reshape(batch, count, heads, cells, dim).transpose(2, 3)           # (B, N, n, H, d)
        expanded = rows.unsqueeze(2) + delta
        return expanded.reshape(batch, count * cells, heads, dim)

    @torch.no_grad()
    def fit_(self, layer: int, extent: tuple[int, int, int], which: str,
             coarse: torch.Tensor, cells: torch.Tensor, ridge: float = 1e-2) -> float:
        """Ridge fit of the delta (their init), truncated to rank ``r`` by SVD.

        ``coarse`` is ``(N, d)`` keys (or values) of coarse tokens, ``cells`` is ``(N, n, d)`` the
        dense model's keys at the cells they cover. Solves ``min |coarse W - (cells - coarse)|^2 +
        ridge |W|^2`` for ``W`` ``(d, n*d)``, keeps the top ``r`` singular directions. Returns
        the relative residual of the copy-plus-delta fit."""
        name = _key(layer, extent, which)
        x = coarse.float()
        target = (cells.float() - x.unsqueeze(1)).reshape(x.shape[0], -1)
        self.fit_moments_(layer, extent, which, x.T @ x, x.T @ target, x.shape[0], ridge)
        fitted = x.unsqueeze(1) + ((x @ self.down[name].float()) @ self.up[name].float()).reshape_as(cells)
        return float((fitted - cells.float()).norm() / cells.float().norm().clamp(min=1e-8))

    @torch.no_grad()
    def fit_moments_(self, layer: int, extent: tuple[int, int, int], which: str,
                     xtx: torch.Tensor, xty: torch.Tensor, count: int, ridge: float = 1e-2) -> None:
        """``fit_`` from accumulated ``X^T X`` ``(d, d)`` and ``X^T (cells - X)`` ``(d, n*d)`` over ``count`` rows."""
        name = _key(layer, extent, which)
        gram = xtx.float() + ridge * count * torch.eye(xtx.shape[0], device=xtx.device)
        weight = torch.linalg.solve(gram, xty.float())                                     # (d, n*d)
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        r = self.rank
        self.down[name].copy_((u[:, :r] * s[:r].sqrt()).to(self.down[name].dtype))
        self.up[name].copy_((s[:r].sqrt().unsqueeze(1) * vh[:r]).to(self.up[name].dtype))


class DualKV:
    """``UpMaps`` plus a per-layout plan cache; what ``h3_musubi.predict(..., dual=)`` takes."""

    def __init__(self, maps: UpMaps):
        self.maps = maps
        self._plans: dict = {}

    def plan(self, layout: LotLayout, device) -> DualKVPlan:
        key = (layout, str(device))
        if key not in self._plans:
            if len(self._plans) > 8:
                self._plans.clear()
            self._plans[key] = DualKVPlan(layout).to(device)
        return self._plans[key]


def _rope(states: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """H3's split-half RoPE (``model._apply_rope_split_half``), repeated so this file needs no model import."""
    pairs = table.shape[-3]
    rotary = torch.stack((states[..., :pairs], states[..., pairs:2 * pairs]), dim=-1)
    rotary = torch.matmul(table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, states[..., 2 * pairs:]), dim=-1)


def _sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``(B, L, H, d)`` in, ``(B, L, H * d)`` out."""
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)


def dual_attention(attn, hidden: torch.Tensor, rotation: torch.Tensor, *, layer: int, plan: DualKVPlan,
                   maps: UpMaps | None, video_start: int, dense_rotation: torch.Tensor,
                   capture: dict | None = None) -> torch.Tensor:
    """H3 ``Attention.forward`` with the dense K/V view for 1x1 queries (see module docstring).

    ``hidden`` is the packed ``(1, S, D)`` sequence whose tail from ``video_start`` is the LoT video;
    ``rotation`` its rotary table; ``dense_rotation`` the table of the dense video positions.
    ``capture`` (optional) receives this layer's pre-RoPE video keys/values for ``UpMaps.fit_``.
    ``maps=None`` is H3's own attention (capture only).
    """
    batch, length, _ = hidden.shape
    query, key, value = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
    query = attn.q_norm(query.reshape(batch, length, attn.num_heads, attn.head_dim))
    key = attn.k_norm(key.reshape(batch, length, attn.num_heads, attn.head_dim))
    value = value.reshape(batch, length, attn.num_heads, attn.head_dim).contiguous()
    if capture is not None:
        capture[layer] = (key[:, video_start:].detach(), value[:, video_start:].detach())
    query_r = _rope(query, rotation).to(value)
    key_r = _rope(key, rotation).to(value)
    if maps is None:
        return attn.out_proj(_sdpa(query_r, key_r, value))

    # Native LoT attention for text, audio and coarse queries.
    other = torch.cat([torch.arange(video_start, device=hidden.device), video_start + plan.other_rows])
    output = hidden.new_zeros(batch, length, attn.inner_dim, dtype=value.dtype)
    output[:, other] = _sdpa(query_r[:, other], key_r, value)

    if plan.fine_rows.numel():
        # Dense view for 1x1 queries: fine keys as they are, coarse keys expanded to their cells.
        video_key, video_value = key[:, video_start:], value[:, video_start:]
        keys = [video_key[:, plan.fine_rows]]
        values = [video_value[:, plan.fine_rows]]
        for extent, rows in plan.groups:
            keys.append(maps.expand(layer, extent, "k", video_key[:, rows]))
            values.append(maps.expand(layer, extent, "v", video_value[:, rows]))
        dense_key = torch.cat(keys, dim=1)[:, plan.dense_order]
        dense_value = torch.cat(values, dim=1)[:, plan.dense_order]
        dense_key = _rope(dense_key, dense_rotation).to(value)
        fine = video_start + plan.fine_rows
        output[:, fine] = _sdpa(query_r[:, fine],
                                torch.cat([key_r[:, :video_start], dense_key], dim=1),
                                torch.cat([value[:, :video_start], dense_value], dim=1))
    return attn.out_proj(output)


@contextlib.contextmanager
def patched_attention(blocks, attention=None, **context):
    """Route every block's ``attn.forward`` through ``dual_attention`` for one forward pass.

    musubi-tuner's code is unchanged: the instance attribute shadows the class method and is
    removed on exit, so checkpointing and block swap see the same modules."""
    patched = []
    try:
        for layer, block in enumerate(blocks):
            attn = block.attn

            def forward(self, hidden, rotation=None, _layer=layer):
                return (attention or dual_attention)(self, hidden, rotation, layer=_layer, **context)

            attn.forward = types.MethodType(forward, attn)
            patched.append(attn)
        yield
    finally:
        for attn in patched:
            del attn.forward


# ------------------------------------------------------------- 4. two native tracks


class TwoTrackPlan:
    """Index tensors for the paper's own arrangement: a complete native LR grid plus HR tokens on a mask.

    ``mask`` is ``(T, h, w)`` bool on the LR token grid; every masked LR token has ``scale * scale``
    HR tokens. ``hr_index`` are their rows in the HR grid's raster order (the order of the HR
    track's rows), and ``cell_order`` permutes "LR token, then its cells row-major" into HR raster.
    """

    def __init__(self, mask: torch.Tensor, scale: int = 2):
        mask = mask.bool().cpu()
        time, height, width = mask.shape
        self.mask, self.scale = mask, int(scale)
        self.lr_count = time * height * width
        self.hr_shape = (time, height * scale, width * scale)
        up = mask.repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        self.hr_index = torch.nonzero(up.flatten()).squeeze(1)
        t, u, v = torch.meshgrid(torch.arange(time), torch.arange(height), torch.arange(width), indexing="ij")
        du, dv = torch.meshgrid(torch.arange(scale), torch.arange(scale), indexing="ij")
        cells = (t.reshape(-1, 1) * (height * scale) + u.reshape(-1, 1) * scale + du.reshape(1, -1)) * (width * scale) \
            + v.reshape(-1, 1) * scale + dv.reshape(1, -1)                                   # (lr, scale^2)
        self.cell_order = torch.argsort(cells.flatten())
        self.extent = (1, self.scale, self.scale)

    @property
    def hr_count(self) -> int:
        return int(self.hr_index.numel())

    def to(self, device) -> "TwoTrackPlan":
        self.hr_index = self.hr_index.to(device)
        self.cell_order = self.cell_order.to(device)
        return self


def two_track_attention(attn, hidden: torch.Tensor, rotation: torch.Tensor, *, layer: int, plan: TwoTrackPlan,
                        maps: UpMaps | None, video_start: int, hr_rotation: torch.Tensor,
                        capture: dict | None = None) -> torch.Tensor:
    """Dual-resolution attention (their eq. 3) on ``[text, audio, LR video (whole grid), HR video (mask)]``.

    Text, audio and LR queries see only text, audio and LR keys: that part is the unmodified model
    at LR, so it needs no ``f_down``. HR queries see text, audio and a full HR grid of keys/values:
    their own inside the mask, ``f_up`` of the LR token's outside it (``maps=None`` copies), rotated
    at the HR grid's positions (``hr_rotation``, the table of the whole HR grid).
    """
    batch, length, _ = hidden.shape
    query, key, value = attn.qkv_proj(hidden).split(attn.inner_dim, dim=-1)
    query = attn.q_norm(query.reshape(batch, length, attn.num_heads, attn.head_dim))
    key = attn.k_norm(key.reshape(batch, length, attn.num_heads, attn.head_dim))
    value = value.reshape(batch, length, attn.num_heads, attn.head_dim).contiguous()
    split = video_start + plan.lr_count
    if capture is not None:
        capture[layer] = (key[:, video_start:split].detach(), value[:, video_start:split].detach())
    query_r = _rope(query, rotation).to(value)
    key_r = _rope(key, rotation).to(value)
    output = hidden.new_zeros(batch, length, attn.inner_dim, dtype=value.dtype)
    output[:, :split] = _sdpa(query_r[:, :split], key_r[:, :split], value[:, :split])
    if plan.hr_count:
        cells = plan.scale * plan.scale
        lr_key, lr_value = key[:, video_start:split], value[:, video_start:split]
        if maps is None:
            grid_key = lr_key.repeat_interleave(cells, dim=1)
            grid_value = lr_value.repeat_interleave(cells, dim=1)
        else:
            grid_key = maps.expand(layer, plan.extent, "k", lr_key).to(key.dtype)
            grid_value = maps.expand(layer, plan.extent, "v", lr_value).to(value.dtype)
        grid_key = grid_key[:, plan.cell_order].index_copy(1, plan.hr_index, key[:, split:])
        grid_value = grid_value[:, plan.cell_order].index_copy(1, plan.hr_index, value[:, split:])
        grid_key = _rope(grid_key, hr_rotation).to(value)
        output[:, split:] = _sdpa(query_r[:, split:],
                                  torch.cat([key_r[:, :video_start], grid_key], dim=1),
                                  torch.cat([value[:, :video_start], grid_value], dim=1))
    return attn.out_proj(output)


# ------------------------------------------------------------------ 1. region head


class RegionHead(nn.Module):
    """``s = w^T LN(h)`` per video token, on a middle block's states (their eq. 7)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.score = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """``(..., L, D)`` -> ``(..., L)`` logits."""
        return self.score(self.norm(states.float())).squeeze(-1)


def scores_to_cells(scores: torch.Tensor, layout: LotLayout) -> torch.Tensor:
    """Per-token scores ``(L,)`` -> ``(T, H, W)``: every cell takes its token's score."""
    field = scores.new_zeros(layout.time, layout.height, layout.width)
    for value, rect in zip(scores, layout.rects):
        field[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew] = value
    return field


def region_mask(field: torch.Tensor, *, alpha: float = 0.5, floor: float | None = None,
                sigma: float = 1.0, temporal: bool = True) -> torch.Tensor:
    """Their selection rule: smooth, then ``M = s~ > m + alpha (p - m)`` per video.

    ``field`` is ``(T, H, W)`` scores (logits or probabilities). Smoothing is a spatial
    Gaussian and, with ``temporal``, a ``[1, 2, 1] / 4`` filter across latent frames. With
    ``floor``, a video whose 99th percentile is under it stays fully coarse."""
    smoothed = smooth_detail(field.float(), sigma)
    if temporal and smoothed.shape[0] > 1:
        padded = torch.cat([smoothed[:1], smoothed, smoothed[-1:]])
        smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    flat = smoothed.flatten()
    median, top = float(flat.median()), float(torch.quantile(flat, 0.99))
    if floor is not None and top < floor:
        return torch.zeros_like(smoothed, dtype=torch.bool)
    return smoothed > median + alpha * (top - median)


def region_layout(mask: torch.Tensor, *, background: int = 4, ring: int = 1) -> LotLayout:
    """Mask -> per-latent-frame LoT layout: 1x1 inside, 2x2 on a ``ring``-token border, ``background`` elsewhere."""
    desired = torch.full(mask.shape, int(background), dtype=torch.long)
    if ring > 0 and background > 2:
        grown = F.max_pool2d(mask.float()[:, None], 2 * ring + 1, stride=1, padding=ring)[:, 0] > 0.5
        desired[grown] = 2
    desired[mask] = 1
    return layout_from_desired_any(desired)


def region_labels(x0_coarse: torch.Tensor, x0_dense: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """Self-supervised label ``(T, H, W)``: the ``fraction`` of tokens per latent frame where the
    coarse layout's clean estimate is furthest from the dense model's (a latent-space stand-in
    for their per-patch LPIPS label). Latents are ``(1, C, T, 2H, 2W)``."""
    error = (x0_coarse.float() - x0_dense.float()).abs().mean(dim=1)[0]               # (T, 2H, 2W)
    error = F.avg_pool2d(error[:, None], 2)[:, 0]                                       # token grid
    flat = error.flatten(1)
    cut = torch.quantile(flat, 1.0 - fraction, dim=1, keepdim=True)
    return (flat >= cut).reshape(error.shape)


def token_targets(labels: torch.Tensor, layout: LotLayout) -> torch.Tensor:
    """Cell labels ``(T, H, W)`` -> per-token targets ``(L,)``: the labelled share of its cells."""
    return torch.stack([labels[r.t, r.u:r.u + r.eh, r.v:r.v + r.ew].float().mean() for r in layout.rects])


# ------------------------------------------------------------------- 3. seam mixer


def extent_maps(layout: LotLayout) -> torch.Tensor:
    """``(2, T, H, W)``: is-1x1 mask and ``log2`` token area, on the token grid."""
    fine = torch.zeros(layout.time, layout.height, layout.width)
    area = torch.zeros_like(fine)
    for rect in layout.rects:
        fine[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew] = float(rect.extent == (1, 1, 1))
        area[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew] = math.log2(rect.eh * rect.ew)
    return torch.stack([fine, area])


class SeamMixer(nn.Module):
    """``z = M z_lot + (1 - M)(z_lot + G(z_lot, maps))`` on the latent (their eq. 6, adapted).

    ``G`` is three 3x3x3 convs; the last is zero-initialised, so the mixer starts as the identity.
    ``M`` is the 1x1 region upsampled to latent pixels."""

    def __init__(self, channels: int = 24, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels + 2, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor, layout: LotLayout) -> torch.Tensor:
        maps = extent_maps(layout).to(latent.device, latent.dtype)
        maps = maps.repeat_interleave(2, -2).repeat_interleave(2, -1).unsqueeze(0)       # (1, 2, T, 2H, 2W)
        residual = self.net(torch.cat([latent, maps], dim=1))
        return latent + (1.0 - maps[:, :1]) * residual


def mixer_loss(mixed: torch.Tensor, target: torch.Tensor, layout: LotLayout) -> torch.Tensor:
    """Latent L1 outside the 1x1 region (their latent term; pixel L1 + LPIPS need the VAE decoder)."""
    outside = 1.0 - extent_maps(layout)[0].to(mixed.device)
    outside = outside.repeat_interleave(2, -2).repeat_interleave(2, -1)[None, None]
    return ((mixed - target).abs() * outside).sum() / (outside.sum() * mixed.shape[1]).clamp(min=1.0)


# ------------------------------------------------- 5. two-track decoder (their Sec. 4.2)


class LatentUpscaler(nn.Module):
    """``U``: LR latent ``(B, C, T, h, w)`` -> ``(B, C, T, 2h, 2w)``, bicubic plus a learned residual.

    The last conv is zero-initialised, so an untrained upscaler is bicubic."""

    def __init__(self, channels: int = 24, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, 4 * width, 3, padding=1), nn.PixelShuffle(2), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = latent.shape
        frames = latent.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width).float()
        up = F.interpolate(frames, scale_factor=2, mode="bicubic", align_corners=False) + self.net(frames)
        return up.reshape(batch, time, channels, 2 * height, 2 * width).permute(0, 2, 1, 3, 4)


class TwoTrackMixer(nn.Module):
    """``z = M z_h + (1 - M)(z_l_up + G(z_h, z_l_up, M))`` (their eq. 6). ``G`` is a small 3D-conv net,
    zero-initialised, so an untrained mixer is a hard paste."""

    def __init__(self, channels: int = 24, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2 * channels + 1, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, width, 3, padding=1), nn.SiLU(),
            nn.Conv3d(width, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hr: torch.Tensor, lr_up: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``hr``, ``lr_up`` ``(B, C, T, H, W)``; ``mask`` ``(B, 1, T, H, W)`` in [0, 1] on the HR latent grid."""
        hr, lr_up, mask = hr.float(), lr_up.float(), mask.float()
        delta = self.net(torch.cat([hr * mask, lr_up, mask], dim=1))
        return mask * hr + (1.0 - mask) * (lr_up + delta)


__all__ = [
    "DualKVPlan", "UpMaps", "DualKV", "dual_attention", "patched_attention", "TwoTrackPlan", "two_track_attention", "RegionHead", "scores_to_cells",
    "region_mask", "region_layout", "region_labels", "token_targets", "SeamMixer", "extent_maps",
    "mixer_loss", "LatentUpscaler", "TwoTrackMixer",
]
