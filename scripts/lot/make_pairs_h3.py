#!/usr/bin/env python3
"""Build phase-4 Procrustes pairs from decoded H3 stills with H3's own VAE encoder.

    python3 scripts/lot/make_pairs_h3.py --src DIR [--frames t00] [--limit N] [--out DIR]

For every still ``x`` (768×1152 by default) and every non-unit H3 extent
``(1, eh, ew)``:

- fine:      ``encode(x)``, patchified to the 2×2 token grid, gathered in
             ``eh × ew`` blocks with ``gather_extent`` -> ``dense (n, 96·eh·ew)``
- reference: ``encode(resize(x, H/eh, W/ew))``, patchified -> ``(n, 96)``

One reference token covers the same pixels as one block of fine tokens, so the
rows align. Rectangles resize each axis separately. The sides must divide by
``32 · 4`` so every extent tiles exactly.

Stills only (T' = 1). The basis is fit on first-frame latents, not on the
4-frame groups of later latent frames. When ``SRC/<clip>/z.pt`` exists, the
fine latent of ``comfy_t00.png`` is compared with ``z[:, :, 0]`` and the
relative RMS is logged; it checks that the encoder and normalization match the
DiT's latent, not that the round trip is lossless.

Writes only into ``--out`` (default ``scripts/lot/runs/pairs_<name>``, gitignored),
and refuses an existing non-empty directory. Does not call empty_cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from flow import gather_extent  # noqa: E402
from h3 import H3_EXTENTS, H3_TOKEN_DIM, patchify  # noqa: E402
from layout import TokenRect  # noqa: E402

VAE = LOT_H3_VAE
DEFAULT_SRC = Path("/home/johndpope/Documents/GitHub/h3-atlas/logs/nikki_scrya_ref2va_768x1152")
UNIT = (1, 1, 1)
PIXELS_PER_TOKEN = 32


def block_rects(token_h: int, token_w: int, extent: tuple[int, int, int]) -> list[TokenRect]:
    """Row-major tiling of one frame by ``extent``, the reference token order."""
    _et, eh, ew = extent
    return [
        TokenRect(0, u, v, 1, eh, ew)
        for u in range(0, token_h, eh)
        for v in range(0, token_w, ew)
    ]


def load_still(path: Path) -> torch.Tensor:
    """``(1, 3, H, W)`` in ``[-1, 1]``."""
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def still_pairs(encode, still: torch.Tensor, extents) -> tuple[torch.Tensor, dict]:
    """Fine latent ``(1, 24, 1, h, w)`` and ``{extent: (dense, reference)}`` for one still."""
    _b, _c, height, width = still.shape
    fine = encode(still)
    tokens = patchify(fine)
    token_h, token_w = tokens.shape[2], tokens.shape[3]
    out = {}
    for extent in extents:
        _et, eh, ew = extent
        small = F.interpolate(still, size=(height // eh, width // ew), mode="bilinear",
                              antialias=True, align_corners=False)
        coarse = patchify(encode(small))
        if coarse.shape[2:4] != (token_h // eh, token_w // ew):
            raise RuntimeError(f"{extent}: coarse grid {tuple(coarse.shape[2:4])} does not tile the fine one")
        dense = gather_extent(tokens, block_rects(token_h, token_w, extent))[0]
        out[extent] = (dense.float().cpu(), coarse.reshape(-1, H3_TOKEN_DIM).float().cpu())
    return fine, out


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp(min=1e-8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="clip folders holding comfy_t*.png")
    parser.add_argument("--frames", default="t00", help="comma list of comfy_<frame>.png to use")
    parser.add_argument("--limit", type=int, default=0, help="first N clips only (0 = all)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--vae", type=Path, default=VAE)
    args = parser.parse_args()

    from gpu_guard import refuse_if_busy

    refuse_if_busy("make_pairs_h3.py")
    if not args.src.is_dir():
        raise SystemExit(f"source {args.src} does not exist")
    out = args.out or Path(__file__).resolve().parent / "runs" / f"pairs_{args.src.name}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; pick another --out")
    frames = [frame for frame in args.frames.split(",") if frame]
    clips = sorted(path for path in args.src.iterdir() if path.is_dir())
    if args.limit:
        clips = clips[: args.limit]
    stills = [(clip, clip / f"comfy_{frame}.png") for clip in clips for frame in frames
              if (clip / f"comfy_{frame}.png").is_file()]
    if not stills:
        raise SystemExit(f"no comfy_{{{args.frames}}}.png under {args.src}")

    from h3_musubi import load_vae

    vae = load_vae(args.vae, decoder=False, dtype=torch.float32)

    def encode(image: torch.Tensor) -> torch.Tensor:
        """``(1, 3, H, W)`` in [-1, 1] -> normalized latent ``(1, 24, 1, H/16, W/16)``."""
        with torch.inference_mode():
            return vae.encode(image.to("cuda", torch.float32))

    extents = [extent for extent in H3_EXTENTS if extent != UNIT]
    rows: dict[tuple, list] = {extent: [] for extent in extents}
    z_gaps = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (clip, path) in enumerate(stills):
            still = load_still(path)
            height, width = still.shape[-2:]
            if height % (PIXELS_PER_TOKEN * 4) or width % (PIXELS_PER_TOKEN * 4):
                raise SystemExit(f"{path}: {width}x{height} does not divide by 128")
            fine, pairs = still_pairs(encode, still, extents)
            for extent, pair in pairs.items():
                rows[extent].append(pair)
            z_path = clip / "z.pt"
            if path.name == "comfy_t00.png" and z_path.is_file():
                z = torch.load(z_path, map_location="cpu", weights_only=True)
                if tuple(z.shape[-2:]) == tuple(fine.shape[-2:]):
                    z_gaps.append(rel_rms(fine[:, :, 0].cpu(), z[:, :, 0]))
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(stills):
                gap = f" z_rel_rms_median={float(np.median(z_gaps)):.4f}" if z_gaps else ""
                print(f"LOT_H3 kind=pairs_progress stills={index + 1}/{len(stills)} "
                      f"seconds={time.perf_counter() - started:.1f}{gap}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    meta = {"src": str(args.src), "frames": frames, "stills": len(stills), "vae": str(args.vae),
            "resize": "bilinear antialias", "extents": {}}
    for extent in extents:
        dense = torch.cat([pair[0] for pair in rows[extent]])
        reference = torch.cat([pair[1] for pair in rows[extent]])
        key = "x".join(str(side) for side in extent)
        torch.save({"dense": dense, "reference": reference}, out / f"{key}.pt")
        meta["extents"][key] = int(dense.shape[0])
        print(f"LOT_H3 kind=pairs extent={key} rows={dense.shape[0]} dense_dim={dense.shape[1]}", flush=True)
    if z_gaps:
        meta["z_rel_rms"] = {"median": float(np.median(z_gaps)), "max": float(np.max(z_gaps)), "n": len(z_gaps)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"LOT_H3 kind=pairs_done out={out} stills={len(stills)} peak_mb={peak_mb:.1f} "
          f"seconds={time.perf_counter() - started:.1f} ok=1", flush=True)


if __name__ == "__main__":
    main()
