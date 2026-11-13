"""Rotary positions for an H3 target-video tail.

H3 does not feed integer token indices to RoPE. Each uniform token sits on a
spatial axis scaled by 32 and a non-uniform time axis whose steps are
``5/3 * (1, 4, 4, 4, 4)``. A LoT token is placed at its center on that same
grid. A 1x1 token lands on the pretrained coordinate.

The packed order here is text, then stereo audio, then target video. There
are no reference frames and no keyframe rows. This module does not import
the Separable Causal Diffusion port.
"""

from __future__ import annotations

import math

import torch

from layout import LotLayout


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
ROPE_SPATIAL_SCALE = 32.0
AUDIO_CHANNELS = 2


def spatial_axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """``dim // patch`` coordinates. A square canvas spans ``[0, 32)``."""
    if patch < 1 or dim % patch != 0:
        raise ValueError(f"dim {dim} is not divisible by patch {patch}")
    count = dim // patch
    ratio = float(dim) / float(sqrt_area)
    index = torch.arange(count, dtype=torch.float64)
    return (index * (ratio / count) + (1.0 - ratio) / 2.0) * ROPE_SPATIAL_SCALE


def sample_axis(axis: torch.Tensor, coord: float) -> torch.Tensor:
    """Linear sample. Integer coordinates return the stored sample."""
    last = axis.shape[0] - 1
    if last < 0:
        raise ValueError("empty rotary axis")
    if coord < -1e-6 or coord > last + 1e-6:
        raise ValueError(f"coordinate {coord} is outside 0..{last}")
    left = int(math.floor(coord))
    if left >= last:
        return axis[last]
    frac = coord - left
    return axis[left] * (1.0 - frac) + axis[left + 1] * frac


def video_time(num_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame, starting at ``origin``."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(num_frames)],
        dtype=torch.float64,
    )
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def video_positions(layout: LotLayout, latent_height: int, latent_width: int, origin: float) -> torch.Tensor:
    """``(L, 3)`` positions in ``(t, h, w)`` for ``layout.rects`` order.

    ``latent_height`` and ``latent_width`` are VAE-latent pixels. The uniform
    token grid is those sizes divided by the native spatial patch of 2.
    """
    token_h = latent_height // 2
    token_w = latent_width // 2
    if latent_height % 2 or latent_width % 2:
        raise ValueError("latent sides must be divisible by the 2x2 patch")
    if layout.height != token_h or layout.width != token_w:
        raise ValueError(
            f"layout {(layout.height, layout.width)} does not match latent "
            f"{(latent_height, latent_width)}"
        )
    sqrt_area = math.sqrt(latent_height * latent_width)
    height_axis = spatial_axis(latent_height, 2, sqrt_area)
    width_axis = spatial_axis(latent_width, 2, sqrt_area)
    times = video_time(layout.time, origin)
    rows = []
    for rect in layout.rects:
        if rect.et != 1:
            raise ValueError("this H3 grid keeps a temporal extent of 1")
        rows.append((
            times[rect.t],
            sample_axis(height_axis, rect.u + (rect.eh - 1) / 2.0),
            sample_axis(width_axis, rect.v + (rect.ew - 1) / 2.0),
        ))
    return torch.stack([torch.stack(row) for row in rows], dim=0)


def _frame_rows(latent_height: int, latent_width: int) -> torch.Tensor:
    sqrt_area = math.sqrt(latent_height * latent_width)
    height = spatial_axis(latent_height, 2, sqrt_area)
    width = spatial_axis(latent_width, 2, sqrt_area)
    hh, ww = torch.meshgrid(height, width, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def _span_sum(num_frames: int) -> float:
    return float(sum(FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(num_frames)))


def packed_positions(
    layout: LotLayout,
    latent_height: int,
    latent_width: int,
    text_len: int,
    num_audio_latents: int,
    keyframes: list[int] | None = None,
    refs: list[tuple] | None = None,
) -> tuple[torch.Tensor, int]:
    """``(S, 3)`` positions and the index where target video begins.

    Order matches H3's packing (and musubi-tuner's ``build_position_grid``): text, keyframes, refs, audio, target video. Audio
    does not advance the video clock. With no keyframes and no refs both start
    at ``text_len``. A reference image advances the cursor by 1. A video-kind
    reference ``(h, w, t)`` with ``t > 1`` advances it by the video span sum.
    Keyframe rows are inserted before the refs, but their time is the cursor
    after those refs, plus ``5/3 * frame_index``.
    """
    if text_len < 0 or num_audio_latents < 0:
        raise ValueError("text and audio lengths must be non-negative")
    cursor = float(text_len)
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    if text_len:
        text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    target_frame = _frame_rows(latent_height, latent_width)
    ref_rows = []
    for ref in refs or ():
        rh, rw = int(ref[0]), int(ref[1])
        rt = int(ref[2]) if len(ref) > 2 else 1
        frame = _frame_rows(rh, rw)
        if rt > 1:
            times = video_time(rt, cursor)
            block = torch.empty(rt * frame.shape[0], 3, dtype=torch.float64)
            block[:, 0] = times.repeat_interleave(frame.shape[0])
            block[:, 1:] = frame.repeat(rt, 1)
            ref_rows.append(block)
            cursor += _span_sum(rt)
            continue
        block = torch.empty(frame.shape[0], 3, dtype=torch.float64)
        block[:, 0] = cursor
        block[:, 1:] = frame
        ref_rows.append(block)
        cursor += 1.0
    parts = [text]
    for index in keyframes or ():
        block = torch.empty(target_frame.shape[0], 3, dtype=torch.float64)
        block[:, 0] = cursor + FRAME_RESCALE * float(index)
        block[:, 1:] = target_frame
        parts.append(block)
    parts.extend(ref_rows)
    if num_audio_latents:
        sqrt_area = math.sqrt(latent_height * latent_width)
        width_axis = spatial_axis(latent_width, 2, sqrt_area)
        audio = torch.zeros(num_audio_latents * AUDIO_CHANNELS, 3, dtype=torch.float64)
        audio[:, 0] = (cursor + torch.arange(num_audio_latents, dtype=torch.float64)).repeat(AUDIO_CHANNELS)
        audio[:, 2] = torch.cat([
            torch.full((num_audio_latents,), float(width_axis[0]), dtype=torch.float64),
            torch.full((num_audio_latents,), float(width_axis[-1]), dtype=torch.float64),
        ])
        parts.append(audio)
    video_start = sum(part.shape[0] for part in parts)
    parts.append(video_positions(layout, latent_height, latent_width, cursor))
    return torch.cat(parts, dim=0), video_start
