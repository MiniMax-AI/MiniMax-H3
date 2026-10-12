#!/usr/bin/env python3
"""Time one H3 VAE clip decode at the LoT timing shape. Writes nothing.

    python3 scripts/lot/time_decode_h3.py [--latent-t 37] [--height 48] [--width 84]

LoT shortens the DiT sequence only; the VAE decodes the full latent either way.
This measures that fixed cost on the same latent shape the DiT timing used
(768×1344, ``latent_t=37`` -> 124 pixel frames). Content does not change the
cost, so the latent is random. fp16 decoder, as ComfyUI loads it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from gpu_guard import refuse_if_busy  # noqa: E402

VAE = LOT_H3_VAE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--latent-t", type=int, default=37)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=84)
    args = parser.parse_args()

    refuse_if_busy("time_decode_h3.py")
    from h3_musubi import load_vae, pixel_frames as pixel_frames_for_latent

    decoder = load_vae(VAE)
    latent = torch.randn(1, 24, args.latent_t, args.height, args.width, device="cuda", dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        pixels = decoder.decode(latent)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        f"LOT_H3 kind=vae_decode latent=({args.latent_t},{args.height},{args.width}) "
        f"pixels={tuple(pixels.shape)} frames={pixel_frames_for_latent(args.latent_t)} "
        f"decode_ms={seconds * 1000:.1f} peak_mb={peak_mb:.1f} dtype=fp16 ok=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
