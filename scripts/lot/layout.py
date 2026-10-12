"""Level-of-Token layouts.

A layout is a partition of the uniform token lattice into axis-aligned
rectangles. Extents are measured in uniform tokens, matching Nakayama et al.
2026, Sec. 3.1. Video layouts in the paper keep a temporal extent of 1 and
vary only the spatial rectangle (Appendix A.10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenRect:
    """One LoT token covering ``[t, t+et) x [u, u+eh) x [v, v+ew)``."""

    t: int
    u: int
    v: int
    et: int
    eh: int
    ew: int

    @property
    def extent(self) -> tuple[int, int, int]:
        return (self.et, self.eh, self.ew)

    def center(self) -> tuple[float, float, float]:
        """Finest-grid center. Unit extents recover the integer lattice (eq. 13)."""
        return (
            self.t + (self.et - 1) / 2,
            self.u + (self.eh - 1) / 2,
            self.v + (self.ew - 1) / 2,
        )


@dataclass(frozen=True)
class LotLayout:
    time: int
    height: int
    width: int
    rects: tuple[TokenRect, ...]

    def __post_init__(self) -> None:
        validate_layout(self)

    @property
    def count(self) -> int:
        return len(self.rects)

    @property
    def dense_count(self) -> int:
        return self.time * self.height * self.width

    @property
    def compression(self) -> float:
        """Uniform-token count divided by the LoT sequence length."""
        return self.dense_count / self.count

    def centers(self) -> torch.Tensor:
        """``(L, 3)`` centers in ``(t, h, w)`` order on the uniform lattice."""
        return torch.tensor([rect.center() for rect in self.rects], dtype=torch.float32)

    def groups(self) -> list[tuple[tuple[int, int, int], list[int]]]:
        """Rect indices grouped by extent, in order of first appearance."""
        order: list[tuple[int, int, int]] = []
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, rect in enumerate(self.rects):
            bucket = buckets.get(rect.extent)
            if bucket is None:
                order.append(rect.extent)
                bucket = []
                buckets[rect.extent] = bucket
            bucket.append(index)
        return [(extent, buckets[extent]) for extent in order]


def validate_layout(layout: LotLayout) -> None:
    if layout.time < 1 or layout.height < 1 or layout.width < 1:
        raise ValueError("layout axes must be positive")
    if not layout.rects:
        raise ValueError("layout has no tokens")
    cover = torch.zeros(layout.time, layout.height, layout.width, dtype=torch.int32)
    for rect in layout.rects:
        if min(rect.et, rect.eh, rect.ew) < 1:
            raise ValueError(f"non-positive extent on {rect}")
        t1 = rect.t + rect.et
        u1 = rect.u + rect.eh
        v1 = rect.v + rect.ew
        if rect.t < 0 or rect.u < 0 or rect.v < 0 or t1 > layout.time or u1 > layout.height or v1 > layout.width:
            raise ValueError(f"rect {rect} leaves the {layout.time}x{layout.height}x{layout.width} lattice")
        cover[rect.t:t1, rect.u:u1, rect.v:v1] += 1
    if int(cover.min()) != 1 or int(cover.max()) != 1:
        raise ValueError("rects are not a partition of the uniform token lattice")


def dense_layout(time: int, height: int, width: int) -> LotLayout:
    """One 1x1x1 token per uniform site, in t-then-row-major order."""
    rects = [
        TokenRect(t, u, v, 1, 1, 1)
        for t in range(time)
        for u in range(height)
        for v in range(width)
    ]
    return LotLayout(time, height, width, tuple(rects))


def layout_from_rects(time: int, height: int, width: int, rects: list[TokenRect]) -> LotLayout:
    return LotLayout(time, height, width, tuple(rects))


def _walk_squares(time: int, height: int, width: int, root: int, keep) -> LotLayout:
    if root < 1 or (root & (root - 1)) != 0:
        raise ValueError(f"root extent must be a power of two, got {root}")
    if height % root != 0 or width % root != 0:
        raise ValueError(f"token grid {(height, width)} is not divisible by root {root}")
    rects: list[TokenRect] = []

    def visit(t: int, u: int, v: int, size: int) -> None:
        if size == 1 or keep(t, u, v, size):
            rects.append(TokenRect(t, u, v, 1, size, size))
            return
        half = size // 2
        for du in (0, half):
            for dv in (0, half):
                visit(t, u + du, v + dv, half)

    for t in range(time):
        for u in range(0, height, root):
            for v in range(0, width, root):
                visit(t, u, v, root)
    return LotLayout(time, height, width, tuple(rects))


def _as_time_field(field: torch.Tensor) -> torch.Tensor:
    if field.ndim == 2:
        return field.unsqueeze(0)
    if field.ndim == 3:
        return field
    raise ValueError(f"expected (H, W) or (T, H, W), got {tuple(field.shape)}")


def layout_from_detail(
    detail: torch.Tensor,
    thresholds: dict[int, float],
    root: int = 8,
) -> LotLayout:
    """Quadtree of eq. 23.

    A block of extent ``b`` splits when its maximum detail is at least
    ``thresholds[b]``. Thresholds must be nondecreasing as blocks get finer:
    ``t_root <= ... <= t_2``. ``detail`` is already on the uniform token lattice.
    """
    field = _as_time_field(detail).detach().float()
    sizes = [size for size in (8, 4, 2) if size <= root]
    missing = [size for size in sizes if size not in thresholds]
    if missing:
        raise ValueError(f"thresholds missing extents {missing}")
    ordered = [float(thresholds[size]) for size in sizes]
    if ordered != sorted(ordered):
        raise ValueError("thresholds must satisfy t8 <= t4 <= t2")

    def keep(t: int, u: int, v: int, size: int) -> bool:
        block = field[t, u:u + size, v:v + size]
        return float(block.max()) < float(thresholds[size])

    return _walk_squares(field.shape[0], field.shape[1], field.shape[2], root, keep)


def layout_from_desired(desired: torch.Tensor, root: int = 8) -> LotLayout:
    """Keep a square only when every cell inside allows that extent.

    ``desired`` stores the coarsest square extent each finest cell may belong
    to. This is the merge rule used for mask levels and for blur radii: a
    block is coarsened only when its most demanding cell still allows it.
    """
    field = _as_time_field(desired).detach().float()

    def keep(t: int, u: int, v: int, size: int) -> bool:
        block = field[t, u:u + size, v:v + size]
        return float(block.min()) >= float(size)

    return _walk_squares(field.shape[0], field.shape[1], field.shape[2], root, keep)


def levels_to_desired(level: torch.Tensor, max_level: int = 3) -> torch.Tensor:
    """Map paper levels ``{0,1,2,3}`` to extents ``{8,4,2,1}`` (eq. 19)."""
    if int(level.min()) < 0 or int(level.max()) > max_level:
        raise ValueError(f"levels must lie in 0..{max_level}")
    return 2 ** (max_level - level.to(torch.int64))


def layout_from_regions(
    height: int,
    width: int,
    background_level: int,
    regions: list[tuple[torch.Tensor, int]],
    root: int = 8,
    time: int = 1,
) -> LotLayout:
    """Eq. 19. Overlapping regions keep the finer (larger) level.

    Each mask is ``(H, W)`` or ``(T, H, W)`` with true on the selected cells.
    The same masks are applied on every frame when ``time > 1`` and the mask
    has no time axis. Per-frame masks must already have length ``time``.
    """
    level = torch.full((time, height, width), int(background_level), dtype=torch.int64)
    for mask, region_level in regions:
        if region_level <= background_level:
            raise ValueError("selected regions must be finer than the background")
        mask_field = mask.to(dtype=torch.bool)
        if mask_field.ndim == 2:
            mask_field = mask_field.unsqueeze(0).expand(time, -1, -1)
        if tuple(mask_field.shape) != (time, height, width):
            raise ValueError(f"mask shape {tuple(mask_field.shape)} != {(time, height, width)}")
        picked = torch.full_like(level, int(region_level))
        level = torch.where(mask_field, torch.maximum(level, picked), level)
    return layout_from_desired(levels_to_desired(level), root=root)


def shape_features(extent: tuple[int, int, int], include_time: bool = False) -> torch.Tensor:
    """Eq. 11, optionally followed by ``log2(et)`` for temporal extents."""
    et, eh, ew = extent
    if min(et, eh, ew) < 1:
        raise ValueError(f"bad extent {extent}")
    values = [
        math.log2(eh),
        math.log2(ew),
        math.log2(eh * ew),
        math.log2(eh / ew),
    ]
    if include_time:
        values.append(math.log2(et))
    return torch.tensor(values, dtype=torch.float32)


def smooth_detail(detail: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian on the last two axes. ``sigma <= 0`` is a no-op."""
    if sigma <= 0:
        return detail
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=detail.device)
    kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    return _conv_last(detail.float(), kernel, radius).to(dtype=detail.dtype)


def _conv_last(field: torch.Tensor, kernel: torch.Tensor, radius: int) -> torch.Tensor:
    """Gaussian along height, then along width."""
    flat = field.reshape(-1, 1, field.shape[-2], field.shape[-1])
    weight_w = kernel.view(1, 1, 1, -1)
    weight_h = kernel.view(1, 1, -1, 1)
    flat = torch.nn.functional.pad(flat, (radius, radius, 0, 0), mode="reflect")
    flat = torch.nn.functional.conv2d(flat, weight_w)
    flat = torch.nn.functional.pad(flat, (0, 0, radius, radius), mode="reflect")
    flat = torch.nn.functional.conv2d(flat, weight_h)
    return flat.reshape(field.shape).to(dtype=field.dtype)


def prepare_detail(scores: torch.Tensor, gain: float = 1.0, sigma: float = 0.0) -> torch.Tensor:
    """Gain, optional smooth, clip to ``[0, 1]``. Scores are on the token lattice."""
    return smooth_detail(scores.float() * float(gain), sigma).clamp(0, 1)


def area_average(pixel_scores: torch.Tensor, token_height: int, token_width: int) -> torch.Tensor:
    """Area-average a pixel detail map onto the uniform token lattice."""
    if pixel_scores.ndim == 2:
        field = pixel_scores[None, None]
        squeeze = 2
    elif pixel_scores.ndim == 3:
        field = pixel_scores[:, None]
        squeeze = 1
    else:
        raise ValueError("pixel scores must be (H, W) or (T, H, W)")
    pooled = torch.nn.functional.adaptive_avg_pool2d(field.float(), (token_height, token_width))
    if squeeze == 2:
        return pooled[0, 0]
    return pooled[:, 0]


def blur_radius(depth: torch.Tensor, focal_depth: float, r_target: float) -> torch.Tensor:
    """Circle-of-confusion radius, eqs. 21–22.

    ``K`` is set so the 95th percentile of absolute inverse-depth deviation
    equals ``r_target``. Depth must be positive.
    """
    if focal_depth <= 0 or r_target < 0:
        raise ValueError("focal depth and r_target must be positive")
    deviation = (depth.float().clamp(min=1e-8).reciprocal() - (1.0 / float(focal_depth))).abs()
    flat = deviation.reshape(-1)
    percentile = torch.quantile(flat, 0.95).clamp(min=1e-8)
    return deviation * (float(r_target) / percentile)


def layout_from_blur(radius: torch.Tensor, support: dict[int, float], root: int = 8) -> LotLayout:
    """Merge a block only when its smallest blur radius supports that extent.

    ``support[b]`` is the minimum radius that still permits an extent-``b``
    token. ``support`` values must increase with ``b``. Cells with a small
    radius (in focus) force the block to split.
    """
    sizes = [size for size in (8, 4, 2) if size <= root]
    missing = [size for size in sizes if size not in support]
    if missing:
        raise ValueError(f"blur support missing extents {missing}")
    ordered = [float(support[size]) for size in sizes]
    if ordered != sorted(ordered, reverse=True):
        raise ValueError("blur support must be larger for coarser extents")
    field = _as_time_field(radius).detach().float()

    def keep(t: int, u: int, v: int, size: int) -> bool:
        block = field[t, u:u + size, v:v + size]
        return float(block.min()) >= float(support[size])

    return _walk_squares(field.shape[0], field.shape[1], field.shape[2], root, keep)


def directional_variation(luminance: torch.Tensor) -> torch.Tensor:
    """Per-cell max of absolute forward differences, padded at the far edge.

    This is the error field fed to the VRS tolerance in Appendix A.3. The
    paper cites Yang et al. 2019 for the shading rule and does not publish
    the finite-difference stencil; the forward difference is the stencil
    used here.
    """
    field = _as_time_field(luminance).float()
    dx = (field[..., :, 1:] - field[..., :, :-1]).abs()
    dy = (field[..., 1:, :] - field[..., :-1, :]).abs()
    dx = torch.nn.functional.pad(dx, (0, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    return torch.maximum(dx, dy).reshape(luminance.shape).to(dtype=luminance.dtype)


def layout_from_vrs(
    luminance: torch.Tensor,
    sensitivity: float,
    ambient: float,
    root: int = 8,
) -> LotLayout:
    """Coarsen a block when its variation stays under ``s * (mean(I) + a)``.

    Training in the paper draws ``s`` from ``[0.01, 0.1]`` and ``a`` from
    ``[0.01, 1]``. Larger values permit coarser tokens. The decision is
    square, on the same quadtree as the detail maps. Directional rectangles
    can still be passed through ``layout_from_rects``.
    """
    if sensitivity < 0 or ambient < 0:
        raise ValueError("VRS sensitivity and ambient must be non-negative")
    luma = _as_time_field(luminance).float()
    error = _as_time_field(directional_variation(luminance)).float()

    def keep(t: int, u: int, v: int, size: int) -> bool:
        block_luma = luma[t, u:u + size, v:v + size]
        block_error = error[t, u:u + size, v:v + size]
        tau = float(sensitivity) * (float(block_luma.mean()) + float(ambient))
        return float(block_error.max()) < tau

    return _walk_squares(luma.shape[0], luma.shape[1], luma.shape[2], root, keep)
