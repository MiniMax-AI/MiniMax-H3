#!/usr/bin/env python3
"""One Level-of-Token forward on the local GPU.

This does not load an H3 checkpoint and does not call the Separable Causal
Diffusion stack. It checks that the adapter and the H3 position grid run
on this machine.

    python3 scripts/lot/gpu_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter
from flow import clean_from_velocity, lot_clean_loss, sample_noisy
from h3_positions import packed_positions
from layout import TokenRect, layout_from_rects


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)

    adapter = LotVisualAdapter(16, 64, [(1, 1, 1), (1, 2, 2), (1, 2, 1)]).to(device)
    weight_in = torch.randn(64, 16, device=device)
    weight_out = torch.randn(16, 64, device=device)
    adapter.init_from_pretrained(weight_in, weight_out)

    tokens = torch.randn(1, 2, 4, 6, 16, device=device)
    layout = layout_from_rects(2, 4, 6, [
        TokenRect(t, u, v, 1, 2, 2)
        for t in range(2)
        for u in (0, 2)
        for v in (0, 2)
    ] + [
        TokenRect(t, u, 4, 1, 2, 1)
        for t in range(2)
        for u in (0, 2)
    ] + [
        TokenRect(t, u, 5, 1, 2, 1)
        for t in range(2)
        for u in (0, 2)
    ])
    y0 = adapter.bank.scale_clean(tokens, layout)
    y_t, _noise = sample_noisy(y0, 0.35)

    def backbone(hidden, centers, sigma):
        if hidden.device.type != "cuda":
            raise RuntimeError("backbone left the GPU")
        if centers.shape != (layout.count, 3):
            raise RuntimeError(f"centers {tuple(centers.shape)}")
        return hidden + centers.to(hidden.dtype).mean()

    velocity = adapter(y_t, 0.35, layout, backbone)
    loss = lot_clean_loss(clean_from_velocity(y_t, velocity, 0.35), y0, 0.35)
    loss.backward()
    positions, video_start = packed_positions(layout, 8, 12, text_len=4, num_audio_latents=2)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    name = torch.cuda.get_device_name(device)
    print(
        f"gpu {name} peak_mb {peak_mb:.1f} "
        f"tokens {layout.count}/{layout.dense_count} "
        f"video_start {video_start} positions {tuple(positions.shape)} "
        f"loss {float(loss.detach()):.6f} finite {bool(torch.isfinite(velocity).all())}"
    )
    if not torch.isfinite(velocity).all():
        raise SystemExit("velocity has non-finite values")


if __name__ == "__main__":
    main()
