"""MiniMax-H3 geometry for a Level-of-Token visual stream.

H3's native patch is ``(t, h, w) = (1, 2, 2)`` on 24-channel video latents,
so the uniform token dimension is ``24 * 1 * 2 * 2 = 96``. LoT extents are
counted in those tokens, not in latent pixels. A paper video extent of
``2 x 2`` therefore covers a ``1 x 4 x 4`` block of VAE latents.

The Wan training setup in Appendix A.10 keeps ``e_t = 1`` and draws spatial
extents from ``{1, 2, 4}``, including rectangles. That is the H3 preset.
"""

from __future__ import annotations

import itertools

import torch

from adapter import LotVisualAdapter
from layout import TokenRect, dense_layout, layout_from_rects


H3_LATENT_CHANNELS = 24
H3_PATCH = (1, 2, 2)
H3_TOKEN_DIM = H3_LATENT_CHANNELS * H3_PATCH[0] * H3_PATCH[1] * H3_PATCH[2]
H3_HIDDEN = 5376
H3_HEADS = 56
H3_HEAD_DIM = 128

# Appendix A.10. Temporal extent stays 1; spatial sides are 1, 2, or 4.
H3_SPATIAL_SIDES = (1, 2, 4)
H3_EXTENTS = tuple((1, eh, ew) for eh, ew in itertools.product(H3_SPATIAL_SIDES, repeat=2))

# Image quadtree in Appendix A.3. Squares only, up to 8.
IMAGE_EXTENTS = ((1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8))

# Wan's per-frame token cap. H3 resolutions differ; this is a budget check,
# not a hard limit of the adapter.
WAN_MAX_TOKENS_PER_FRAME = 3072


def patchify(latent: torch.Tensor, patch: tuple[int, int, int] = H3_PATCH) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, T', H', W', C*pt*ph*pw)``.

    Channel order inside the token is C-order ``(channel, pt, ph, pw)`` with
    ``pw`` fastest.
    """
    if latent.ndim != 5:
        raise ValueError("latent must be (B, C, T, H, W)")
    pt, ph, pw = patch
    batch, channels, time, height, width = latent.shape
    if time % pt or height % ph or width % pw:
        raise ValueError(f"latent {(time, height, width)} is not divisible by patch {patch}")
    tt, hh, ww = time // pt, height // ph, width // pw
    tokens = latent.reshape(batch, channels, tt, pt, hh, ph, ww, pw)
    tokens = tokens.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return tokens.reshape(batch, tt, hh, ww, channels * pt * ph * pw)


def unpatchify(
    tokens: torch.Tensor,
    patch: tuple[int, int, int] = H3_PATCH,
    channels: int = H3_LATENT_CHANNELS,
) -> torch.Tensor:
    """Inverse of ``patchify``."""
    pt, ph, pw = patch
    if tokens.ndim != 5 or tokens.shape[-1] != channels * pt * ph * pw:
        raise ValueError("token shape does not match channels and patch")
    batch, tt, hh, ww, _dense = tokens.shape
    tokens = tokens.reshape(batch, tt, hh, ww, channels, pt, ph, pw)
    tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return tokens.reshape(batch, channels, tt * pt, hh * ph, ww * pw)


def make_h3_adapter(
    hidden_size: int = H3_HIDDEN,
    *,
    include_time: bool = False,
    shape_hidden: int = 256,
    extents: tuple[tuple[int, int, int], ...] = H3_EXTENTS,
) -> LotVisualAdapter:
    """Adapter whose token dim matches H3. Weights are not loaded here."""
    return LotVisualAdapter(
        H3_TOKEN_DIM,
        hidden_size,
        list(extents),
        include_time=include_time,
        shape_hidden=shape_hidden,
    )


def tokens_per_frame(layout_count: int, time: int) -> float:
    if time < 1:
        raise ValueError("time must be positive")
    return layout_count / time


def over_wan_frame_budget(layout_count: int, time: int) -> bool:
    return tokens_per_frame(layout_count, time) > WAN_MAX_TOKENS_PER_FRAME


def gate_frame_layout(height: int = 24, width: int = 42):
    """One frame of three horizontal bands: 1×1, then 2×2, then 4×2.

    The default is the 768×1344 timing clip: 24×42 tokens, 462 rectangles.
    Width 42 is not divisible by 4, so the coarse band is 4×2, not 4×4.
    Rows 0–7 are 1×1, rows 8–15 are 2×2, rows 16–23 are 4×2. The phase-4
    smoke canvas (384×640) is ``gate_frame_layout(12, 20)``.
    """
    band = height // 3
    if height % 3 or band % 4 or width % 2:
        raise ValueError(f"token grid {(height, width)} needs height a multiple of 12 and an even width")
    rects = []
    for u in range(band):
        for v in range(width):
            rects.append(TokenRect(0, u, v, 1, 1, 1))
    for u in range(band, 2 * band, 2):
        for v in range(0, width, 2):
            rects.append(TokenRect(0, u, v, 1, 2, 2))
    for u in range(2 * band, height, 4):
        for v in range(0, width, 2):
            rects.append(TokenRect(0, u, v, 1, 4, 2))
    return layout_from_rects(1, height, width, rects)


def clip_layout(frames: int, height: int = 24, width: int = 42):
    """``gate_frame_layout`` repeated on every latent frame."""
    frame = gate_frame_layout(height, width)
    rects = [
        TokenRect(t, rect.u, rect.v, 1, rect.eh, rect.ew)
        for t in range(frames)
        for rect in frame.rects
    ]
    return layout_from_rects(frames, height, width, rects)


GRID_LAYOUTS = ("dense", "bands", "center", "uniform2", "uniform4")


def grid_layout(name: str, time: int, height: int, width: int):
    """A LoT layout for any token grid, in 4×4 super-cells; edge leftovers stay 1×1.

    ``bands``: top third 1×1, middle 2×2, bottom 4×2 (by super-cell row).
    ``center``: 1×1 near the middle, 2×2 around it, 4×4 at the edges; meant for
    portraits and talking heads, where the face sits in the middle.
    ``uniformN``: every full super-cell N×N. ``dense``: all 1×1.
    """
    if name not in GRID_LAYOUTS:
        raise ValueError(f"layout {name!r} is not one of {GRID_LAYOUTS}")
    if name == "dense":
        return dense_layout(time, height, width)   # raster order: bit-identical to the stock forward
    rows, cols = height // 4, width // 4

    def extent(cell_u: int, cell_v: int) -> tuple[int, int, int]:
        if name == "dense":
            return (1, 1, 1)
        if name == "uniform2":
            return (1, 2, 2)
        if name == "uniform4":
            return (1, 4, 4)
        if name == "bands":
            third = cell_u * 3 // max(rows, 1)
            return ((1, 1, 1), (1, 2, 2), (1, 4, 2))[min(third, 2)]
        du = (cell_u + 0.5) / max(rows, 1) - 0.5
        dv = (cell_v + 0.5) / max(cols, 1) - 0.5
        radius = (du * du + dv * dv) ** 0.5 / 0.7071
        return (1, 1, 1) if radius < 0.35 else (1, 2, 2) if radius < 0.7 else (1, 4, 4)

    rects = []
    for t in range(time):
        for u in range(height):
            for v in range(width):
                if u >= rows * 4 or v >= cols * 4:
                    rects.append(TokenRect(t, u, v, 1, 1, 1))     # edge leftovers
        for cu in range(rows):
            for cv in range(cols):
                _et, eh, ew = extent(cu, cv)
                for u in range(cu * 4, cu * 4 + 4, eh):
                    for v in range(cv * 4, cv * 4 + 4, ew):
                        rects.append(TokenRect(t, u, v, 1, eh, ew))
    return layout_from_rects(time, height, width, rects)
