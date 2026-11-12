"""Content-aware LoT layouts: semantic mask, bounding boxes, depth of field, texture.

The LoT paper's supplementary (layout-adaptive generation) derives per-frame
layouts from these four sources with one trained model: fine tokens on the
primary action region, coarser ones on secondary objects and background. Each
source here becomes a per-token *desired extent* map (the coarsest square a
token may join: 1, 2 or 4), and ``layout_from_desired_any`` turns that into a
quadtree layout per latent frame, so the layout follows the content over time.

Pixel inputs are ``(F, H, W)`` (or ``(H, W)`` for a single frame) in [0, 1] at
pixel resolution; they are area-averaged onto the token lattice (one H3 token =
32×32 pixels) and mapped from pixel frames onto H3's latent frames
(``(1, 4, 4, 4, 4)`` pixel frames per latent frame, repeating).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from h3_positions import FRAME_PER_TOKEN
from layout import LotLayout, TokenRect, area_average, blur_radius, layout_from_desired

EXTENTS = (1, 2, 4)


def latent_frame_spans(latent_t: int) -> list[tuple[int, int]]:
    """``[start, stop)`` pixel frames covered by each H3 latent frame."""
    spans, start = [], 0
    for k in range(latent_t):
        n = FRAME_PER_TOKEN[k % 5]
        spans.append((start, start + n))
        start += n
    return spans


def to_latent_frames(field: torch.Tensor, latent_t: int, reduce: str = "max") -> torch.Tensor:
    """``(F, h, w)`` per pixel frame -> ``(latent_t, h, w)`` per latent frame.

    A single frame is broadcast. Any other frame count is stretched onto the
    expected pixel-frame count first, so a mask from a shorter or longer clip
    still lines up in time.
    """
    if field.ndim == 2:
        field = field[None]
    spans = latent_frame_spans(latent_t)
    total = spans[-1][1]
    if field.shape[0] == 1:
        return field.expand(latent_t, *field.shape[1:]).clone()
    if field.shape[0] != total:
        index = torch.linspace(0, field.shape[0] - 1, total).round().long()
        field = field[index]
    pool = (lambda x: x.amax(0)) if reduce == "max" else (lambda x: x.mean(0))
    return torch.stack([pool(field[a:b]) for a, b in spans])


def layout_from_desired_any(desired: torch.Tensor) -> LotLayout:
    """Quadtree (4 → 2 → 1) for any ``(T, H, W)`` grid.

    A 4×4 cell cut by the far edge keeps the clipped rectangle when its cells
    allow it (4×2 on the 2-token strip of H3's 24×42, 2×4 on a 2-row strip), so
    edges coarsen like the interior instead of falling back to 1×1. Every
    clipped extent is in H3's bank (FINE + MID + COARSE).
    """
    return layout_from_desired(desired.long().clamp(1, 4), root=4, edge_rects=True)


def desired_from_levels(fine: torch.Tensor, mid: torch.Tensor | None, background: int) -> torch.Tensor:
    """Boolean token maps -> desired extents: fine 1, mid 2, everything else ``background``."""
    if background not in EXTENTS:
        raise ValueError(f"background extent must be one of {EXTENTS}")
    desired = torch.full(fine.shape, background, dtype=torch.long)
    if mid is not None:
        desired[mid & (desired > 2)] = 2
    desired[fine] = 1
    return desired


def _tokens(pixel: torch.Tensor, latent_t: int, token_h: int, token_w: int, reduce: str = "max") -> torch.Tensor:
    pixel = pixel.float()
    if pixel.ndim == 2:
        pixel = pixel[None]
    return to_latent_frames(area_average(pixel, token_h, token_w), latent_t, reduce)


def _dilate(field: torch.Tensor, tokens: int) -> torch.Tensor:
    if tokens <= 0:
        return field
    k = 2 * tokens + 1
    return F.max_pool2d(field.float()[:, None], k, stride=1, padding=tokens)[:, 0] > 0.5


def mask_layout(mask: torch.Tensor, latent_t: int, token_h: int, token_w: int, *,
                secondary: torch.Tensor | None = None, background: int = 4,
                coverage: float = 0.05, dilate: int = 1) -> LotLayout:
    """Semantic mask (or rasterised boxes): 1×1 on the mask, 2×2 on ``secondary``, ``background`` elsewhere.

    A token counts as covered when ``coverage`` of its 32×32 pixels are inside
    the mask; ``dilate`` grows the fine region by that many tokens so edges stay sharp.
    """
    fine = _dilate(_tokens(mask, latent_t, token_h, token_w) > coverage, dilate)
    mid = None
    if secondary is not None:
        mid = _dilate(_tokens(secondary, latent_t, token_h, token_w) > coverage, dilate)
    return layout_from_desired_any(desired_from_levels(fine, mid, background))


def boxes_to_mask(boxes: list[tuple[float, float, float, float]], height: int, width: int,
                  frames: int = 1) -> torch.Tensor:
    """Normalised ``(x0, y0, x1, y1)`` boxes -> ``(frames, height, width)`` mask (same boxes every frame)."""
    mask = torch.zeros(frames, height, width)
    for x0, y0, x1, y1 in boxes:
        a, b = sorted((x0, x1))
        c, d = sorted((y0, y1))
        mask[:, int(c * height):max(int(d * height), int(c * height) + 1),
             int(a * width):max(int(b * width), int(a * width) + 1)] = 1.0
    return mask


# lot/layout.py layout_from_blur. A cell may join extent b only when its circle of
# confusion is at least this radius. Coarser extents need a larger defocus.
_BLUR_SUPPORT = ((8, 3.0), (4, 1.0), (2, 0.2))


def _desired_from_radius(radius: torch.Tensor, background: int) -> torch.Tensor:
    """Blur radius -> coarsest extent each cell may join. In-focus cells stay at 1.

    ``background`` is the coarsest square this caller can emit. H3's video heads stop at 4,
    so a radius that would be an 8×8 square stays at 4.
    """
    desired = torch.ones(radius.shape, dtype=torch.long)
    for extent, support in reversed(_BLUR_SUPPORT):
        if extent > background:
            continue
        desired = torch.where(radius >= support, torch.full_like(desired, extent), desired)
    return desired


def depth_layout(depth: torch.Tensor, latent_t: int, token_h: int, token_w: int, *,
                 focus: float = 0.75, band: float = 0.15, background: int = 4,
                 r_target: float = 4.0) -> LotLayout:
    """Derived depth-of-field layout. Fine tokens on the focal plane, coarse squares on the defocus.

    The radius is the circle of confusion (eqs. 21–22): ``|1/z - 1/z_focus|``, scaled so the
    95th percentile of each latent frame equals ``r_target``. A square is kept only when every
    cell inside it is blurred enough for that extent. ``depth`` is non-negative and on the same
    scale as ``focus``. A Depth Anything map (brighter = nearer, typically in [0, 1]) is legal
    when ``focus`` is read from that same map. ``band`` stays in the signature for existing
    callers; the radius replaces that fixed band.
    """
    if background not in EXTENTS:
        raise ValueError(f"background extent must be one of {EXTENTS}")
    if focus <= 0 or r_target < 0 or band <= 0:
        raise ValueError("focus must be positive, on the depth map's scale; r_target and band must be positive")
    tokens = _tokens(depth, latent_t, token_h, token_w, reduce="mean")
    if not torch.isfinite(tokens).all() or float(tokens.min()) < 0:
        raise ValueError("depth must be finite and non-negative")
    desired = torch.stack([_desired_from_radius(blur_radius(frame, float(focus), r_target), background)
                           for frame in tokens])
    return layout_from_desired_any(desired)


def texture_scores(image: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """Per-token texture: std of luminance inside each token's pixels, ``(F, token_h, token_w)``.

    ``image`` is ``(F, H, W, C)`` or ``(F, H, W)`` in [0, 1].
    """
    if image.ndim == 4:
        image = (image[..., :3].float() * torch.tensor([0.299, 0.587, 0.114])).sum(-1)
    lum = image.float()
    mean = area_average(lum, token_h, token_w)
    sq = area_average(lum * lum, token_h, token_w)
    return (sq - mean * mean).clamp(min=0).sqrt()


def score_layout(score: torch.Tensor, *, fine_q: float = 0.7, mid_q: float = 0.4,
                 background: int = 4) -> LotLayout:
    """``(T, H, W)`` importance scores -> layout by per-frame quantiles: top ``1-fine_q`` 1×1, next band 2×2."""
    flat = score.flatten(1)
    hi = flat.quantile(fine_q, dim=1)[:, None, None]
    lo = flat.quantile(mid_q, dim=1)[:, None, None]
    # Strictly above the low quantile: perfectly flat tokens (ties at the bottom) may go coarse.
    mid = score > lo
    return layout_from_desired_any(desired_from_levels((score >= hi) & mid, mid, background))


def texture_layout(image: torch.Tensor, latent_t: int, token_h: int, token_w: int, *,
                   fine_q: float = 0.7, mid_q: float = 0.4, background: int = 4) -> LotLayout:
    """Texture variance (the paper's VRS source): fine where the image is busy, coarse where it is flat."""
    score = to_latent_frames(texture_scores(image, token_h, token_w), latent_t)
    return score_layout(score, fine_q=fine_q, mid_q=mid_q, background=background)


def compression(layout: LotLayout) -> float:
    return layout.dense_count / max(layout.count, 1)


__all__ = [
    "EXTENTS", "TokenRect", "boxes_to_mask", "compression", "depth_layout", "desired_from_levels",
    "latent_frame_spans", "layout_from_desired_any", "mask_layout", "score_layout", "texture_layout",
    "texture_scores", "to_latent_frames",
]
