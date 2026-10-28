#!/usr/bin/env python3
"""Closed 1x1 parity: an all-1x1 LoT tail against the dense H3 forward on one small latent.

Loads the pruned int8 checkpoint with block streaming. Writes no file.
Refuses while another LoT GPU script is alive. Does not call empty_cache.

    python3 scripts/lot/parity_h3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from h3 import make_h3_adapter  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import dense_layout  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT



def rel_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (left.float() - right.float()).pow(2).mean().sqrt()
    scale = right.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return float(delta / scale)


def main() -> None:
    refuse_if_busy("parity_h3.py")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    from h3_musubi import load_dit, predict

    model = load_dit(CHECKPOINT, swap=32)
    height = width = 16
    video = torch.randn(1, 24, 1, height, width, device="cuda", dtype=torch.float32)
    text = torch.randn(1, 4, model.config.text_dim, device="cuda", dtype=torch.bfloat16)

    def run(splice=None):
        audio = torch.Generator("cuda").manual_seed(0)
        with torch.inference_mode():
            return predict(model, video, 0.6, text, lot=splice, audio_generator=audio)

    off_a = run()
    off_b = run()
    floor = rel_rms(off_a, off_b)

    adapter = make_h3_adapter()
    adapter.init_from_pretrained(
        model.video_patch_proj.weight.detach().float().cpu(),
        model.final_layer.video_out.weight.detach().float().cpu(),
        model.video_patch_proj.bias.detach().float().cpu(),
        model.final_layer.video_out.bias.detach().float().cpu(),
    )
    adapter.cuda()
    layout = dense_layout(1, height // 2, width // 2)
    on = run(LotSplice(adapter, layout))
    gap = rel_rms(on, off_a)
    max_abs = float((on.float() - off_a.float()).abs().max())
    limit = max(1e-4, 2.0 * floor)
    ok = int(gap <= limit and max_abs <= 1e-2 and floor <= 1e-3)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        "LOT_H3 kind=parity task=t2va "
        f"dense_tokens={layout.dense_count} lot_tokens={layout.count} "
        f"rel_rms={gap:.6g} rel_rms_floor={floor:.6g} max_abs={max_abs:.6g} "
        f"peak_mb={peak_mb:.1f} dtype=bf16 ok={ok}"
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
