#!/usr/bin/env python3
"""Build a train_h3 cache: latents + captions, then Qwen3-VL text.

    python3 scripts/lot/build_cache_h3.py media --src DIR --out DIR [--frames 22] [--size 640 384]
    python3 scripts/lot/build_cache_h3.py latents [--out DIR]   # h3-atlas z.pt latents (no VAE)
    python3 scripts/lot/build_cache_h3.py text    [--out DIR]   # GPU: Qwen3-VL captions -> _te

``media`` encodes your own files with the H3 video VAE: every ``.mp4`` and every
``.png``/``.jpg``/``.webp`` in ``--src`` that has a same-stem ``.txt`` caption.
Clips keep their first ``--frames`` frames (17n+5: 22 -> 7 latent frames) and
should be 24 fps; stills become one latent frame. Each picture is scaled to cover
``--size W H`` and centre-cropped (default: centre-cropped to multiples of 32).
Writes ``<stem>_<W>x<H>_minimaxh3.safetensors`` (key ``latent_...``) and merges
the captions into ``captions.json``; then run ``text``.

``latents`` writes ``<stem>_0512x0768_minimaxh3.safetensors`` with key
``latent_7x48x32`` (the first 7 latent frames of each ``z.pt``: 22 pixel frames,
the start of a clip, so the causal first frame stays a keyframe) and
``captions.json``. No VAE: these are H3's own latents.

``text`` loads the Qwen3-VL-32B text encoder once, encodes every caption, writes
``<stem>_minimaxh3_te.safetensors`` (``hidden_states``, ``attention_mask``) and
exits. Run it alone on the GPU. It never runs beside a DiT.

Sources:
- ``set_t_talk`` items whose ``meta.json`` id is ``nikki``: talking-head clips with
  full H3 prompts.
- ``set_a_pose_ext``: pose transitions. ``meta.json`` has no prompt, so the caption
  is built from ``from_card`` -> ``to_card``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

ATLAS = Path("/home/johndpope/Documents/GitHub/h3-atlas/data")
TEXT_ENCODER = LOT_H3_TEXT_ENCODER
DEFAULT_OUT = Path(__file__).resolve().parent / "runs" / "cache_nikki"
LATENT_T = 7


def card_words(card: str) -> str:
    """``forward_lean_kneel_afa9fa9c`` -> ``forward lean kneel``: drop hash-like tokens."""
    words = [w for w in card.split("_") if not re.fullmatch(r"[0-9a-f]{6,}", w)]
    return " ".join(words)


def pose_caption(meta: dict) -> str:
    start = card_words(meta["from_card"])
    end = card_words(meta["to_card"])
    def article(words: str) -> str:
        return "an" if words[:1] in "aeiou" else "a"

    return (f"Full-body shot of the woman in a plain studio, moving smoothly from {article(start)} {start} "
            f"pose into {article(end)} {end} pose. Static locked-off camera, single continuous shot, "
            f"sharp focus.")


def sources() -> list[tuple[str, Path, str]]:
    """``(stem, z_path, caption)`` for every usable item."""
    items = []
    for meta_path in sorted((ATLAS / "set_t_talk").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("id") == "nikki" and (meta_path.parent / "z.pt").is_file():
            items.append((f"talk_{meta_path.parent.name}", meta_path.parent / "z.pt", meta["prompt"]))
    for meta_path in sorted((ATLAS / "set_a_pose_ext").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if (meta_path.parent / "z.pt").is_file():
            items.append((f"pose_{meta_path.parent.name}", meta_path.parent / "z.pt", pose_caption(meta)))
    return items


def build_latents(out: Path) -> None:
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    captions = {}
    shapes = {}
    for stem, z_path, caption in sources():
        z = torch.load(z_path, map_location="cpu", weights_only=True)
        if z.ndim != 5 or z.shape[:2] != (1, 24) or z.shape[2] < LATENT_T:
            print(f"skip {stem}: z {tuple(z.shape)}", flush=True)
            continue
        latent = z[0, :, :LATENT_T].float().contiguous()             # (24, 7, H, W)
        _c, t, h, w = latent.shape
        key = f"latent_{t}x{h}x{w}"
        save_file({key: latent}, str(out / f"{stem}_{w * 16:04d}x{h * 16:04d}_minimaxh3.safetensors"))
        captions[stem] = caption
        shapes[key] = shapes.get(key, 0) + 1
    (out / "captions.json").write_text(json.dumps(captions, indent=1) + "\n")
    print(f"LOT_H3 kind=cache_latents out={out} items={len(captions)} shapes={shapes}", flush=True)


VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def read_media(path: Path, frames: int) -> tuple[torch.Tensor, float | None]:
    """``(F, H, W, 3)`` uint8 frames and the clip's fps (``None`` for a still)."""
    import numpy as np
    from PIL import Image

    if path.suffix.lower() in IMAGE_SUFFIXES:
        return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy())[None], None
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        decoded = []
        for frame in container.decode(stream):
            decoded.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            if len(decoded) == frames:
                break
    if len(decoded) < frames:
        raise ValueError(f"{path.name}: {len(decoded)} frames, need {frames}")
    return torch.stack(decoded), fps


def fit_canvas(pixels: torch.Tensor, size: tuple[int, int] | None) -> torch.Tensor:
    """``(F, H, W, 3)`` uint8 -> ``(1, 3, F, H', W')`` in [-1, 1], sides multiples of 32."""
    import torch.nn.functional as F

    video = pixels.permute(3, 0, 1, 2).float().div(127.5).sub(1.0)       # (3, F, H, W)
    height, width = video.shape[-2:]
    if size is not None:
        target_w, target_h = size
        scale = max(target_w / width, target_h / height)
        resized = (max(target_h, round(height * scale)), max(target_w, round(width * scale)))
        video = F.interpolate(video.transpose(0, 1), size=resized, mode="bicubic",
                              antialias=True, align_corners=False).transpose(0, 1).clamp(-1, 1)
        height, width = resized
    else:
        target_w, target_h = width // 32 * 32, height // 32 * 32
    if target_w % 32 or target_h % 32:
        raise ValueError(f"--size {target_w}x{target_h}: both sides must be multiples of 32")
    top, left = (height - target_h) // 2, (width - target_w) // 2
    return video[:, :, top:top + target_h, left:left + target_w].unsqueeze(0)


def build_media(src: Path, out: Path, frames: int, size: tuple[int, int] | None) -> None:
    from safetensors.torch import save_file

    from gpu_guard import refuse_if_busy
    from h3_musubi import latent_frames, load_vae, pixel_frames

    if pixel_frames(latent_frames(frames)) != frames:
        raise SystemExit(f"--frames {frames} is not on the 17n+5 grid (5, 22, 39, 56, ...)")
    refuse_if_busy("build_cache_h3.py")
    out.mkdir(parents=True, exist_ok=True)
    captions_path = out / "captions.json"
    captions = json.loads(captions_path.read_text()) if captions_path.is_file() else {}
    vae = load_vae(LOT_H3_VAE, decoder=False, dtype=torch.float32)
    done = 0
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in VIDEO_SUFFIXES + IMAGE_SUFFIXES:
            continue
        caption = path.with_suffix(".txt")
        if not caption.is_file():
            print(f"skip {path.name}: no {caption.name}", flush=True)
            continue
        pixels, fps = read_media(path, frames)
        if fps is not None and abs(fps - 24.0) > 0.01:
            print(f"warn {path.name}: {fps:.3f} fps, H3 trains at 24", flush=True)
        video = fit_canvas(pixels, size)
        with torch.inference_mode():
            latent = vae.encode(video.to("cuda", torch.float32))[0].float().cpu()   # (24, T, h, w)
        if latent.shape[1] == 1:
            latent = latent[:, 0]
        key = "latent_" + "x".join(str(n) for n in latent.shape[1:])
        stem = path.stem
        save_file({key: latent.contiguous()},
                  str(out / f"{stem}_{video.shape[-1]:04d}x{video.shape[-2]:04d}_minimaxh3.safetensors"))
        captions[stem] = caption.read_text().strip()
        done += 1
        print(f"LOT_H3 kind=cache_media item={stem} pixels={tuple(video.shape[2:])} latent={key}", flush=True)
    captions_path.write_text(json.dumps(captions, indent=1) + "\n")
    print(f"LOT_H3 kind=cache_media_done out={out} items={done} captions={len(captions)}", flush=True)


def build_text(out: Path) -> None:
    from safetensors.torch import save_file

    from gpu_guard import refuse_if_busy
    import h3_musubi  # noqa: F401  (puts musubi-tuner on sys.path)
    from musubi_tuner.minimax_h3.text_encoder import (
        H3Presentation,
        encode_h3_presentation,
        load_h3_processor,
        load_h3_text_encoder,
    )

    refuse_if_busy("build_cache_h3.py")
    captions = json.loads((out / "captions.json").read_text())
    todo = {stem: text for stem, text in captions.items()
            if not (out / f"{stem}_minimaxh3_te.safetensors").is_file()}
    print(f"LOT_H3 kind=cache_text todo={len(todo)} of {len(captions)}", flush=True)
    if not todo:
        return
    processor = load_h3_processor()
    encoder = load_h3_text_encoder(str(TEXT_ENCODER), device="cuda")
    for index, (stem, text) in enumerate(todo.items()):
        with torch.inference_mode():
            hidden, _tags = encode_h3_presentation(processor, encoder, H3Presentation(text=text, processor_text=text))
        hidden = hidden.reshape(-1, hidden.shape[-1]).to(torch.bfloat16).cpu()
        save_file({"hidden_states": hidden, "attention_mask": torch.ones(hidden.shape[0], dtype=torch.bool)},
                  str(out / f"{stem}_minimaxh3_te.safetensors"))
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(todo):
            print(f"LOT_H3 kind=cache_text done={index + 1}/{len(todo)} rows={hidden.shape[0]} "
                  f"dim={hidden.shape[1]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("media", "latents", "text"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--src", type=Path, default=None, help="media: folder of clips/stills with .txt captions")
    parser.add_argument("--frames", type=int, default=22, help="media: frames kept per clip (17n+5)")
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="media: cover-resize and centre-crop to W x H (multiples of 32)")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.phase == "media":
        if args.src is None or not args.src.is_dir():
            raise SystemExit("media needs --src DIR")
        build_media(args.src, args.out, args.frames, tuple(args.size) if args.size else None)
    elif args.phase == "latents":
        build_latents(args.out)
    else:
        build_text(args.out)


if __name__ == "__main__":
    main()
