#!/usr/bin/env python3
"""Same-seed stills: dense H3, LoT with the mean lift, LoT with a Procrustes fit.

    python3 scripts/lot/render_h3.py --pairs DIR [--prompts 3] [--steps 20] [--out DIR]

Text conditioning is a cached H3 text embedding (``*_minimaxh3_te.safetensors``
from ``build_cache_h3.py``), so the Qwen3-VL encoder is never loaded. Every column
starts from the same noise. LoT columns use the 768×1152 band layout
(``clip_layout(1, 36, 24)``: 1×1, 2×2, 4×2 bands). The fitted column samples in
y-space and maps the clean latent back with ``to_x`` before decoding. A frozen
base was never trained on fitted scales, so that column is a smoke, not a
quality claim.

Samples with ``h3_musubi.sample`` (Euler, shift 12) and decodes with the H3 video VAE after sampling and writes PNGs plus a grid
into ``--out`` (default ``scripts/lot/runs/render``, gitignored).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from gpu_guard import refuse_if_busy  # noqa: E402
from h3 import clip_layout, make_h3_adapter  # noqa: E402
from layout import TokenRect, layout_from_rects  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from procrustes_h3 import CHECKPOINT, fit_bank, load_pairs, pretrained_maps  # noqa: E402

TE_CACHE = LOT_H3_STILLS_CACHE
CAPTIONS = LOT_H3_STILLS_CAPTIONS
VAE = LOT_H3_VAE
WIDTH, HEIGHT = 768, 1152
# A still fits fully resident: 20.2 GB int8 weights, 20.8 GB peak on the 24 GB card.
# Streaming 32 blocks made every forward ~13.5 s instead of ~0.69 s.
SWAP = 0


def load_prompts(count: int, paths: list[Path] | None = None) -> list[tuple[str, torch.Tensor]]:
    """``count`` cached embeddings with their caption text: ``paths``, else the first by name."""
    from safetensors import safe_open

    prompts = []
    for path in (paths or sorted(TE_CACHE.glob("*_minimaxh3_te.safetensors")))[:count]:
        with safe_open(str(path), "pt") as handle:
            hidden = handle.get_tensor("hidden_states")
            mask = handle.get_tensor("attention_mask")
        stem = path.name.replace("_minimaxh3_te.safetensors", "")
        caption = CAPTIONS / f"{stem}.txt"
        text = caption.read_text().strip() if caption.is_file() else stem
        prompts.append((text, hidden[mask].unsqueeze(0)))
    return prompts


def make_layout(name: str):
    token_h, token_w = HEIGHT // 32, WIDTH // 32
    if name == "bands":
        return clip_layout(1, token_h, token_w)
    side = 2 if name == "uniform2" else 4
    rects = [TokenRect(0, u, v, 1, side, side)
             for u in range(0, token_h, side) for v in range(0, token_w, side)]
    return layout_from_rects(1, token_h, token_w, rects)


def to_image(pixels: torch.Tensor, frame: int = 0) -> Image.Image:
    """One frame of a decoded ``(1, 3, F, H, W)`` video in ``[-1, 1]``."""
    rgb = (pixels[0, :, frame].permute(1, 2, 0).float().cpu().numpy() + 1.0) * 0.5
    array = (rgb * 255.0).round().clip(0, 255)
    return Image.fromarray(array.astype(np.uint8))


def grid(rows: list[list[Image.Image]], labels: list[str], scale: float = 0.33) -> Image.Image:
    cell_w, cell_h = int(WIDTH * scale), int(HEIGHT * scale)
    header = 28
    sheet = Image.new("RGB", (cell_w * len(labels), header + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        draw.text((col * cell_w + 6, 8), label, fill="black")
    for row, images in enumerate(rows):
        for col, image in enumerate(images):
            sheet.paste(image.resize((cell_w, cell_h), Image.LANCZOS), (col * cell_w, header + row * cell_h))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, default=None,
                        help="Procrustes pair dir; needed only for the lot_fit variant")
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "render")
    parser.add_argument("--layout", choices=("bands", "uniform2", "uniform4"), default="bands",
                        help="bands: 1x1/2x2/4x2 thirds; uniformN: every token NxN")
    parser.add_argument("--variants", default="dense,lot_mean,lot_fit")
    parser.add_argument("--swap", type=int, default=SWAP, help="blocks streamed from CPU (0 = all resident)")
    parser.add_argument("--trained", type=Path, default=None,
                        help="train_h3 output dir (adapter.pt, lora.safetensors): enables the dense_lora and "
                             "lot_trained variants and renders the trainer's held-out prompts")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank train_h3 used")
    parser.add_argument("--holdout", type=int, default=24, help="the --holdout train_h3 used for that cache")
    parser.add_argument("--pick", default=None, help="comma list of held-out indices to render, e.g. 7,3")
    parser.add_argument("--prompt-cache", type=Path, default=None,
                        help="with --trained: take held-out prompts from this train_h3 cache (default cache_iso3d)")
    args = parser.parse_args()

    refuse_if_busy("render_h3.py")
    from h3_musubi import load_dit, load_vae, make_lora, sample, set_lora

    # Build only what the requested variants use: each fp32 adapter is ~200 MB, and with
    # the int8 DiT fully resident there is no room for spares.
    wanted = {name for name in args.variants.split(",") if name}
    layout = make_layout(args.layout)
    splices = {}
    reports = []
    if "lot_mean" in wanted:
        mean = make_h3_adapter()
        mean.init_from_pretrained(*pretrained_maps(CHECKPOINT))
        splices["lot_mean"] = LotSplice(mean.cuda(), layout)
    if "lot_fit" in wanted:
        fitted = make_h3_adapter()
        fitted.init_from_pretrained(*pretrained_maps(CHECKPOINT))
        if args.pairs is None:
            raise SystemExit("the lot_fit variant needs --pairs")
        reports = fit_bank(fitted, load_pairs(args.pairs))
        splices["lot_fit"] = LotSplice(fitted.cuda(), layout, y_space=True)

    model = load_dit(CHECKPOINT, swap=args.swap)
    network = None
    if args.trained is not None:
        trained = make_h3_adapter()
        from train_h3 import load_adapter_state

        trained.load_state_dict(load_adapter_state(args.trained))
        if "lot_trained" in wanted:
            splices["lot_trained"] = LotSplice(trained.cuda(), layout, y_space=True)
        network = make_lora(model, args.rank)
        info = network.load_weights(str(args.trained / "lora.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"LoRA load missed {len(info.missing_keys)} keys")
        network.to(device="cuda", dtype=torch.bfloat16).eval()

    held = None
    if args.trained is not None:
        from train_h3 import CACHE, list_items

        held = [te for _latent, te in list_items(args.prompt_cache or CACHE, args.holdout)[1]]
        if args.pick:
            held = [held[int(index)] for index in args.pick.split(",")]
    prompts = load_prompts(args.prompts, held)
    variants = [name for name in args.variants.split(",") if name]
    latents: dict[tuple[int, str], torch.Tensor] = {}
    timings: dict[str, list[float]] = {name: [] for name in variants}
    seq = {"dense": layout.dense_count, "lot": layout.count}
    for index, (_text, embeds) in enumerate(prompts):
        embeds = embeds.to("cuda", torch.bfloat16)
        for name in variants:
            if network is not None:
                set_lora(network, name in ("dense_lora", "lot_trained"))
            start = time.perf_counter()
            with torch.inference_mode():
                latent = sample(model, embeds, width=WIDTH, height=HEIGHT, steps=args.steps,
                                seed=args.seed + index, lot=splices.get(name))
            torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - start)
            if name in ("lot_fit", "lot_trained"):
                latent = splices[name].to_x(latent.float())
            latents[(index, name)] = latent.float().cpu()
            print(f"LOT_H3 kind=render prompt={index} variant={name} seconds={timings[name][-1]:.1f} "
                  f"finite={int(bool(torch.isfinite(latent).all()))}", flush=True)
    del model, splices, network
    gc.collect()
    torch.cuda.synchronize()
    decoder = load_vae(VAE)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(prompts)):
        row = []
        for name in variants:
            with torch.inference_mode():
                image = to_image(decoder.decode(latents[(index, name)].to("cuda", next(decoder.parameters()).dtype)))
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
        rows.append(row)
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    labels = [f"{name}  {medians[name]:.0f}s/img" for name in variants]
    grid(rows, labels).save(args.out / "grid.png")
    meta = {
        "layout": args.layout, "steps": args.steps, "seed": args.seed, "canvas": f"{WIDTH}x{HEIGHT}", "layout_tokens": layout.count,
        "dense_tokens": layout.dense_count, "seq": seq,
        "seconds_median": medians, "prompts": [text for text, _ in prompts],
        "fit": [{**r, "extent": list(r["extent"])} for r in reports],
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"LOT_H3 kind=render_done out={args.out} seq={seq} "
          + " ".join(f"{k}_s={v:.1f}" for k, v in medians.items()) + " ok=1", flush=True)


if __name__ == "__main__":
    main()
