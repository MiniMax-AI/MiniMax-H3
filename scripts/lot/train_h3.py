#!/usr/bin/env python3
"""Phase 5: fine-tune H3 for Level-of-Token layouts. Stills first.

    LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs DIR [--steps 2000] [--swap 4]

What trains: the LoT adapter (per-extent ``in_proj`` / ``out_proj``, shape MLP)
and a rank-16 LoRA on ``attn.qkv_proj``, ``attn.out_proj``, ``mlp.fc1``,
``mlp.fc2`` in every block. AdaLN, the token refiner, the int8 base, and the
extent bank (``A``, ``s`` from ``--pairs``) stay frozen.

One step: sample a layout, move the clean latent to y-space (``to_y``), draw
sigma (``h3_musubi.sample_sigma``), noise, run the DiT with the LoT tail, and
take ``lot_h3_clean_loss`` (eq. 17 in H3's ``x0 - eps`` convention: velocity
MSE in y-space). About 20% of steps use the dense 1×1 layout, which is the
ordinary H3 LoRA loss and anchors the base.

Sigma: ``--dense-shift`` for the dense anchor steps (default 12, H3's own
density: median 0.92, ~3% of steps below 0.3) and ``--shift`` for LoT steps.
The first 2,000-step run used 12 for both and the coarse regions came out
soft: detail is decided at low sigma, which shift 12 almost never trains.
A float is the uniform-u shift map, ``sigmoid`` is logit-normal, ``lognorm:S``
is logit-normal under shift S.

Model: musubi-tuner's H3 transformer through ``h3_musubi`` (pruned int8
ConvRot base, H2D-only block swap, activation checkpointing, its kohya-style
LoRA network). The LoRA keys are ``lora_unet_blocks_N_...``, so musubi and
ComfyUI load the result directly.

Data: one or more H3 cache directories from ``build_cache_h3.py`` (``--cache``, repeatable) of
``*_minimaxh3.safetensors`` latents with ``*_minimaxh3_te.safetensors`` text,
so Qwen is never loaded here. Stills are ``(24, H, W)``; clips are
``(24, T, H, W)`` (cache 384×640, 22 frames for ``latent_t = 7``). Each latent
is randomly cropped to a token grid divisible by 4. ``--init DIR`` warm-starts
the adapter and LoRA from an earlier run's ``adapter.pt`` / ``lora.safetensors``
with a fresh optimizer.

Distillation (``--distill W``): on LoT steps, the frozen base (LoRA off, dense
layout) predicts the clean latent from the same ``x_t`` (same noise, same
sigma), and the student's clean estimate, mapped back with ``to_x``, is pulled
toward it with the same ``1 / max(sigma, 0.05)^2`` weight. The target is still
a one-step posterior mean, not a sample; what it adds is that LoT learns to
reproduce *dense H3's* prediction at every sigma instead of a noisy data
target, and the base teacher carries no style drift. ``--data-weight`` scales
the eq. 17 data term (0 = distillation only). Each LoT step costs one extra
no-grad dense forward. Evals also report ``gap_*``: the same weighted distance
between LoT and the base teacher, which is the number distillation drives.

Writes only under ``--out`` (default ``scripts/lot/runs/train_h3``, gitignored):
``log.jsonl``, and at ``--save-every`` and the end ``adapter.pt`` +
``lora.safetensors``. Refuses to start unless ``LOT_H3_TRAIN=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA init

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from flow import lot_h3_clean_loss, sample_noisy  # noqa: E402
from h3 import H3_EXTENTS, make_h3_adapter, patchify  # noqa: E402
from h3_musubi import predict, sample_sigma, set_lora  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import LotLayout, TokenRect, dense_layout, layout_from_rects  # noqa: E402

CACHE = LOT_H3_STILLS_CACHE
SUPER = 4  # layout super-cell, in tokens: every extent tiles a 4×4 cell
FINE = [(1, 1, 1)]
MID = [(1, 1, 2), (1, 2, 1), (1, 2, 2)]
COARSE = [(1, 2, 4), (1, 4, 2), (1, 4, 4), (1, 1, 4), (1, 4, 1)]


# ---------------------------------------------------------------------------------- data

def list_items(cache: Path, holdout: int) -> tuple[list, list]:
    """``(latent_path, te_path)`` pairs, split deterministically into train / held out."""
    if not cache.is_dir():
        raise SystemExit(f"cache directory {cache} does not exist")
    items = []
    for latent in sorted(cache.glob("*_minimaxh3.safetensors")):
        stem = re.sub(r"_\d{4}x\d{4}_minimaxh3\.safetensors$", "", latent.name)
        te = cache / f"{stem}_minimaxh3_te.safetensors"
        if te.is_file():
            items.append((latent, te))
    if len(items) <= holdout:
        raise SystemExit(f"{cache}: {len(items)} items with text, need more than {holdout}")
    rng = random.Random(0)
    rng.shuffle(items)
    return items[holdout:], items[:holdout]


MAX_LATENT_HW: tuple[int, int] | None = None   # set from --max-latent-hw in main()


def load_item(item, rng: random.Random, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean latent ``(1, 24, T, H, W)`` cropped to a 4-token multiple, and text ``(1, L, 5120)``.

    A still is ``T = 1``; a cached clip keeps its latent frames. ``--max-latent-hw``
    caps the spatial crop (a random window), for clips too large to train whole.
    """
    from safetensors import safe_open

    latent_path, te_path = item
    with safe_open(str(latent_path), "pt") as handle:
        key = next(k for k in handle.keys() if k.startswith("latent"))
        latent = handle.get_tensor(key).float()
    with safe_open(str(te_path), "pt") as handle:
        hidden = handle.get_tensor("hidden_states")
        mask = handle.get_tensor("attention_mask")
    if latent.ndim == 3:
        latent = latent.unsqueeze(1)                      # (24, 1, H, W)
    step = 2 * SUPER                                       # latent pixels per super-cell
    height, width = latent.shape[-2] // step * step, latent.shape[-1] // step * step
    if MAX_LATENT_HW is not None:
        height = min(height, MAX_LATENT_HW[0] // step * step)
        width = min(width, MAX_LATENT_HW[1] // step * step)
    top = rng.randint(0, latent.shape[-2] - height)
    left = rng.randint(0, latent.shape[-1] - width)
    latent = latent[..., top:top + height, left:left + width].unsqueeze(0)
    return latent.to(device), hidden[mask].unsqueeze(0).to(device, torch.bfloat16)


def load_adapter_state(run_dir: Path) -> dict:
    """``adapter.safetensors`` (as published) or ``adapter.pt`` (as train_h3 writes it)."""
    published = run_dir / "adapter.safetensors"
    if published.is_file():
        from safetensors.torch import load_file

        return load_file(str(published))
    return torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=True)


# -------------------------------------------------------------------------------- layout

def token_detail(latent: torch.Tensor) -> torch.Tensor:
    """``(T, H, W)`` detail per token: distance from its 4×4-cell mean, on the clean latent."""
    tokens = patchify(latent.float())[0]                   # (T, H, W, 96)
    grid = tokens.permute(0, 3, 1, 2)                       # (T, 96, H, W)
    coarse = F.avg_pool2d(grid, SUPER)
    coarse = F.interpolate(coarse, scale_factor=SUPER, mode="nearest")
    return (grid - coarse).norm(dim=1)


DEFAULT_MIX = {"uniform": 0.10, "mosaic": 0.15, "box": 0.20, "focus": 0.20, "texture": 0.15}


def parse_mix(text: str) -> dict[str, float]:
    """``"box:0.3,focus:0.2"`` -> weights for the non-dense layout families."""
    mix = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, weight = part.partition(":")
        if name not in DEFAULT_MIX:
            raise ValueError(f"unknown layout family {name!r}; choose from {sorted(DEFAULT_MIX)}")
        mix[name] = float(weight)
    return mix


def _box_desired(time: int, height: int, width: int, rng: random.Random) -> torch.Tensor:
    """1-2 primary boxes drifting across frames, an optional secondary box; the paper's mask/bbox family."""
    from layout_sources import desired_from_levels

    background = rng.choice((2, 4))
    fine = torch.zeros(time, height, width, dtype=torch.bool)
    mid = torch.zeros_like(fine)

    def box(target: torch.Tensor, size_lo: float, size_hi: float) -> None:
        bh, bw = rng.uniform(size_lo, size_hi), rng.uniform(size_lo, size_hi)
        y, x = rng.uniform(0, 1 - bh), rng.uniform(0, 1 - bw)
        dy, dx = rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)
        for t in range(time):
            f = t / max(time - 1, 1)
            yy = min(max(y + dy * f, 0.0), 1 - bh)
            xx = min(max(x + dx * f, 0.0), 1 - bw)
            u0, v0 = int(yy * height), int(xx * width)
            target[t, u0:max(int((yy + bh) * height), u0 + 1), v0:max(int((xx + bw) * width), v0 + 1)] = True

    for _ in range(rng.choice((1, 1, 2))):
        box(fine, 0.2, 0.55)
    if rng.random() < 0.5:
        box(mid, 0.3, 0.8)
    return desired_from_levels(fine, mid, background)


def _focus_desired(time: int, height: int, width: int, rng: random.Random) -> torch.Tensor:
    """An in-focus ellipse (1×1), a 2×2 ring, coarse outside; the depth-of-field family."""
    from layout_sources import desired_from_levels

    cy, cx = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    ry, rx = rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4)
    ring = rng.uniform(1.4, 2.2)
    dy, dx = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
    uu = (torch.arange(height).float() + 0.5) / height
    vv = (torch.arange(width).float() + 0.5) / width
    radius = []
    for t in range(time):
        f = t / max(time - 1, 1)
        r = (((uu[:, None] - cy - dy * f) / ry) ** 2 + ((vv[None, :] - cx - dx * f) / rx) ** 2).sqrt()
        radius.append(r)
    radius = torch.stack(radius)
    return desired_from_levels(radius <= 1.0, radius <= ring, rng.choice((2, 4)))


def sample_layout(latent: torch.Tensor, rng: random.Random, dense_p: float = 0.2,
                  mix: dict[str, float] | None = None, force: str | None = None) -> tuple[LotLayout, str]:
    """A layout for one example and its kind.

    ``dense``: all 1×1. ``uniform``: one extent everywhere. ``mosaic``: each 4×4
    super-cell takes one extent, finer where the clean latent has more detail.
    Content families from the LoT supplementary, all changing per latent frame:
    ``box`` (drifting primary/secondary boxes, the mask/bbox source), ``focus``
    (an in-focus ellipse, the depth-of-field source) and ``texture`` (per-frame
    quantiles of the clean latent's detail, the VRS source). ``mix`` weights the
    non-dense families (default ``DEFAULT_MIX``). ``force="eval_mosaic"`` is the
    exact held-out layout used by every eval so far (kept for comparable curves).
    """
    _b, _c, time, lat_h, lat_w = latent.shape
    height, width = lat_h // 2, lat_w // 2
    if force == "eval_mosaic":
        return _legacy_mosaic(latent, rng, 0.0)
    if force is None:
        if rng.random() < dense_p:
            return dense_layout(time, height, width), "dense"
        weights = mix or DEFAULT_MIX
        names = [n for n, w in weights.items() if w > 0]
        force = rng.choices(names, weights=[weights[n] for n in names])[0]
    if force == "dense":
        return dense_layout(time, height, width), "dense"
    if force == "uniform":
        extent = rng.choice(MID + COARSE)
        return _tile(time, height, width, lambda _t, _u, _v: extent), "uniform"
    if force == "mosaic":
        return _legacy_mosaic(latent, rng, 0.0, uniform_p=0.0)
    from layout_sources import layout_from_desired_any, score_layout

    if force == "box":
        return layout_from_desired_any(_box_desired(time, height, width, rng)), "box"
    if force == "focus":
        return layout_from_desired_any(_focus_desired(time, height, width, rng)), "focus"
    if force == "texture":
        fine_q = rng.uniform(0.5, 0.85)
        layout = score_layout(token_detail(latent).cpu(), fine_q=fine_q,
                              mid_q=max(0.0, fine_q - rng.uniform(0.15, 0.35)), background=rng.choice((2, 4)))
        return layout, "texture"
    raise ValueError(f"unknown layout family {force!r}")


def _legacy_mosaic(latent: torch.Tensor, rng: random.Random, dense_p: float,
                   uniform_p: float = 0.15) -> tuple[LotLayout, str]:
    """The original sampler (runs 1-4): dense / uniform / detail mosaic of 4×4 super-cells."""
    _b, _c, time, lat_h, lat_w = latent.shape
    height, width = lat_h // 2, lat_w // 2
    roll = rng.random()
    if roll < dense_p:
        return dense_layout(time, height, width), "dense"
    if roll < dense_p + uniform_p:
        extent = rng.choice(MID + COARSE)
        return _tile(time, height, width, lambda _t, _u, _v: extent), "uniform"
    detail = token_detail(latent).cpu()
    cells = F.avg_pool2d(detail.unsqueeze(1), SUPER).squeeze(1)    # (T, H/4, W/4)
    flat = cells.flatten()
    low_q, high_q = sorted((rng.uniform(0.2, 0.6), rng.uniform(0.5, 0.9)))
    low, high = float(flat.quantile(low_q)), float(flat.quantile(high_q))

    def pick(t: int, u: int, v: int):
        value = float(cells[t, u // SUPER, v // SUPER])
        if value >= high:
            return FINE[0]
        if value >= low:
            return rng.choice(MID)
        return rng.choice(COARSE)

    return _tile(time, height, width, pick), "mosaic"


def _tile(time: int, height: int, width: int, choose) -> LotLayout:
    rects = []
    for t in range(time):
        for u0 in range(0, height, SUPER):
            for v0 in range(0, width, SUPER):
                _et, eh, ew = choose(t, u0, v0)
                for u in range(u0, u0 + SUPER, eh):
                    for v in range(v0, v0 + SUPER, ew):
                        rects.append(TokenRect(t, u, v, 1, eh, ew))
    return layout_from_rects(time, height, width, rects)


def extent_losses(out, y_t, y0, sigma, layout: LotLayout) -> dict[str, float]:
    """Held-out squared clean error per extent, on the token lattice."""
    sig = float(sigma)
    err = (y_t.float() + sig * out.float() - y0.float()).square() / max(sig, 0.05) ** 2
    tokens = patchify(err)[0].mean(dim=-1)                  # (T, H, W)
    sums: dict[str, list] = {}
    for rect in layout.rects:
        block = tokens[rect.t, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew]
        sums.setdefault("x".join(map(str, rect.extent)), []).append(float(block.mean()))
    return {key: sum(values) / len(values) for key, values in sums.items()}


# --------------------------------------------------------------------------------- train

def parse_shift(value: str):
    """``12`` -> 12.0; ``sigmoid`` and ``lognorm:S`` pass through to ``sample_sigma``."""
    if value == "sigmoid" or value.startswith("lognorm:"):
        return value
    return float(value)


def check(adapter, dit, network, groups, items, rng, device, args) -> None:
    """Forward + backward once per layout kind. No optimizer step, nothing written."""
    params = [p for g in groups for p in g["params"]]
    for kind_wanted in ("dense", "box", "focus", "texture", "mosaic", "uniform"):
        x0, text = load_item(rng.choice(items), rng, device)
        layout, kind = sample_layout(x0, rng, force=kind_wanted)
        splice = LotSplice(adapter, layout, y_space=True)
        y0 = splice.to_y(x0)
        sigma = sample_sigma(args.dense_shift if kind == "dense" else args.shift, device)
        y_t, _ = sample_noisy(y0, sigma)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = predict(dit, y_t, 1.0 - sigma, text, lot=splice)
        loss = lot_h3_clean_loss(out, y_t, y0, sigma)
        if args.distill > 0 and kind != "dense":
            # Teacher: same x_t, LoRA off, dense, no grad. Pulled toward with the same weight.
            x_t = (1.0 - sigma).reshape(1, 1, 1, 1, 1) * x0 + sigma.reshape(1, 1, 1, 1, 1) * (y_t - (1.0 - sigma).reshape(1, 1, 1, 1, 1) * y0) / sigma.reshape(1, 1, 1, 1, 1)
            set_lora(network, False)
            with torch.no_grad():
                teacher_out = predict(dit, x_t, 1.0 - sigma, text)
            set_lora(network, True)
            target = x_t + sigma.reshape(1, 1, 1, 1, 1) * teacher_out.float()
            student = splice.to_x(y_t.float() + sigma.reshape(1, 1, 1, 1, 1) * out.float())
            loss = loss + args.distill * (student - target).square().mean() / float(sigma.clamp(min=0.05)) ** 2
        loss.backward()
        torch.cuda.synchronize()
        lora_grad = sum(float(p.grad.abs().sum()) for p in groups[0]["params"] if p.grad is not None)
        adapter_grad = sum(float(p.grad.abs().sum()) for p in groups[-1]["params"] if p.grad is not None)
        for p in params:
            p.grad = None
        print(f"LOT_H3 kind=train_check layout={kind} tokens={layout.count}/{layout.dense_count} "
              f"loss={float(loss.detach()):.4f} sigma={float(sigma):.3f} lora_grad={lora_grad:.3g} "
              f"adapter_grad={adapter_grad:.3g} seconds={time.perf_counter() - start:.2f} "
              f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok={int(math.isfinite(float(loss.detach())))}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True, help="Procrustes pair dir for the frozen bank")
    parser.add_argument("--cache", type=Path, action="append", default=None,
                        help=f"H3 cache dir; repeat to mix stills and clips (default {CACHE})")
    parser.add_argument("--shift", type=parse_shift, default=12.0,
                        help="sigma density for LoT steps: float shift, 'sigmoid', or 'lognorm:S'")
    parser.add_argument("--dense-shift", type=parse_shift, default=12.0,
                        help="sigma density for the dense anchor steps (12 = H3's own)")
    parser.add_argument("--distill", type=float, default=0.0,
                        help="weight of the frozen-dense-teacher term on LoT steps (0 = off)")
    parser.add_argument("--data-weight", type=float, default=1.0,
                        help="weight of the eq. 17 data term (0 = distillation only)")
    parser.add_argument("--max-latent-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="random spatial crop cap in latent pixels (multiples of 8), e.g. 40 32")
    parser.add_argument("--init", type=Path, default=None,
                        help="warm-start adapter.pt + lora.safetensors from an earlier run dir")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--swap", type=int, default=4, help="blocks streamed from CPU; >0 turns on checkpointing")
    parser.add_argument("--dense-p", type=float, default=0.2)
    parser.add_argument("--layout-mix", type=parse_mix, default=None,
                        help="weights of non-dense layout families, e.g. box:0.3,focus:0.3,texture:0.2,mosaic:0.1,"
                             "uniform:0.1 (default: DEFAULT_MIX; 'mosaic' alone reproduces runs 1-4)")
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=0, help="0 = save only at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs" / "train_h3")
    parser.add_argument("--check", action="store_true",
                        help="one forward + backward per layout kind, no optimizer step, writes nothing; "
                             "allowed without LOT_H3_TRAIN")
    args = parser.parse_args()

    global MAX_LATENT_HW
    MAX_LATENT_HW = tuple(args.max_latent_hw) if args.max_latent_hw else None
    if not args.check and os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("phase 5 is gated: set LOT_H3_TRAIN=1 to train (or --check for a dry pass)")
    from gpu_guard import refuse_if_busy

    refuse_if_busy("train_h3.py")
    from h3_musubi import load_dit, make_lora
    from procrustes_h3 import CHECKPOINT, fit_bank, load_pairs, pretrained_maps

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    caches = args.cache or [CACHE]
    train_items, held_items = [], []
    for cache in caches:
        train_part, held_part = list_items(cache, args.holdout)
        train_items += train_part
        held_items += held_part

    adapter = make_h3_adapter()
    adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))
    fit_bank(adapter, load_pairs(args.pairs))
    adapter.to(device)
    adapter.bank.requires_grad_(False)

    dit = load_dit(CHECKPOINT, device=device, swap=args.swap, train=True)
    network = make_lora(dit, args.rank)
    if args.init is not None:
        state = load_adapter_state(args.init)
        adapter.load_state_dict(state)
        info = network.load_weights(str(args.init / "lora.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"--init LoRA is missing {len(info.missing_keys)} keys")
        print(f"LOT_H3 kind=train_init from={args.init}", flush=True)
    network.requires_grad_(True)
    network.to(device=device, dtype=torch.bfloat16)

    lora_groups, _ = network.prepare_optimizer_params(args.lr)
    adapter_params = [p for name, p in adapter.named_parameters() if not name.startswith("bank.")]
    groups = lora_groups + [{"params": adapter_params, "lr": args.adapter_lr}]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
        opt_name = "adamw8bit"
    except ImportError:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        opt_name = "adamw"
    trainable = sum(p.numel() for g in groups for p in g["params"])
    print(f"LOT_H3 kind=train_start items={len(train_items)} holdout={len(held_items)} "
          f"lora_modules={len(network.unet_loras)} trainable={trainable} optimizer={opt_name} "
          f"swap={args.swap} shift={args.shift} dense_shift={args.dense_shift} caches={len(caches)} "
          f"distill={args.distill} data_weight={args.data_weight} "
          f"allocated_mb={torch.cuda.memory_allocated() / 2**20:.0f}", flush=True)

    if args.check:
        check(adapter, dit, network, groups, train_items, rng, device, args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def teacher_clean(x0, noise, sigma, text) -> torch.Tensor:
        """Frozen base (LoRA off, dense) clean estimate from the same x_t; H3's head is x0 - eps."""
        x_t = (1.0 - sigma).reshape(1, 1, 1, 1, 1) * x0 + sigma.reshape(1, 1, 1, 1, 1) * noise
        set_lora(network, False)
        try:
            with torch.no_grad():
                out = predict(dit, x_t, 1.0 - sigma, text)
        finally:
            set_lora(network, True)
        return (x_t + sigma.reshape(1, 1, 1, 1, 1) * out.float()).detach()

    def teacher_gap(splice, out, y_t, sigma, target) -> torch.Tensor:
        student = splice.to_x(y_t.float() + sigma.reshape(1, 1, 1, 1, 1) * out.float())
        weight = 1.0 / float(sigma.clamp(min=0.05)) ** 2
        return (student - target).square().mean() * weight

    def step_loss(item, item_rng, *, layout=None, sigma=None, noise_seed=None, with_teacher=False):
        x0, text = load_item(item, item_rng, device)
        if layout is None:
            layout, kind = sample_layout(x0, item_rng, args.dense_p, args.layout_mix)
        else:
            kind = "fixed"
        splice = LotSplice(adapter, layout, y_space=True)
        y0 = splice.to_y(x0)
        if sigma is None:
            sigma = sample_sigma(args.dense_shift if kind == "dense" else args.shift, device)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32).reshape(1)
        gen = None if noise_seed is None else torch.Generator(device=device).manual_seed(noise_seed)
        noise = torch.randn(y0.shape, device=device, generator=gen)
        y_t, _ = sample_noisy(y0, sigma, noise)
        out = predict(dit, y_t, 1.0 - sigma, text, lot=splice)
        data = lot_h3_clean_loss(out, y_t, y0, sigma)
        gap = None
        # Training draws its own layout (kind != "fixed"); evals pass one and opt in explicitly.
        if with_teacher or (args.distill > 0 and kind not in ("dense", "fixed")):
            gap = teacher_gap(splice, out, y_t, sigma, teacher_clean(x0, noise, sigma, text))
        loss = args.data_weight * data + (args.distill * gap if gap is not None and kind != "dense"
                                          and not with_teacher else 0.0)
        return loss, out, y_t, y0, sigma, layout, kind, data, gap

    def evaluate(step: int) -> None:
        network.eval()
        results = {"dense": [], "uniform2": [], "mosaic": [], "focus": [],
                   "gap_uniform2": [], "gap_mosaic": [], "gap_focus": []}
        per_extent: dict[str, list] = {}
        with torch.no_grad():
            for index, item in enumerate(held_items):
                for kind in ("dense", "uniform2", "mosaic", "focus"):
                    item_rng = random.Random(1000 + index)
                    x0, _text = load_item(item, random.Random(1000 + index), device)
                    _b, _c, tt, hh, ww = x0.shape
                    if kind == "dense":
                        layout = dense_layout(tt, hh // 2, ww // 2)
                    elif kind == "uniform2":
                        layout = _tile(tt, hh // 2, ww // 2, lambda *_: (1, 2, 2))
                    else:
                        layout, _ = sample_layout(x0, random.Random(2000 + index), force="eval_mosaic")
                    if kind == "focus":
                        layout, _ = sample_layout(x0, random.Random(3000 + index), force="focus")
                    sigma = (0.3, 0.6, 0.9)[index % 3]
                    _loss, out, y_t, y0, sig, lay, _, data, gap = step_loss(
                        item, item_rng, layout=layout, sigma=sigma, noise_seed=index,
                        with_teacher=(kind != "dense"))
                    results[kind].append(float(data))
                    if gap is not None:
                        results[f"gap_{kind}"].append(float(gap))
                    if kind == "mosaic":
                        for key, value in extent_losses(out, y_t, y0, sig, lay).items():
                            per_extent.setdefault(key, []).append(value)
        network.train()
        record = {"step": step, "eval": {k: sum(v) / len(v) for k, v in results.items()},
                  "eval_extent": {k: sum(v) / len(v) for k, v in sorted(per_extent.items())}}
        log.write(json.dumps(record) + "\n")
        log.flush()
        line = " ".join(f"{k}={v:.4f}" for k, v in record["eval"].items())
        print(f"LOT_H3 kind=train_eval step={step} {line}", flush=True)

    def save(tag: str) -> None:
        torch.save({k: v for k, v in adapter.state_dict().items()}, args.out / f"adapter{tag}.pt")
        network.save_weights(str(args.out / f"lora{tag}.safetensors"), torch.bfloat16,
                             {"lot_h3": "phase5", "rank": str(args.rank)})
        print(f"LOT_H3 kind=train_saved tag={tag or 'final'} out={args.out}", flush=True)

    network.train()
    evaluate(0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        item = rng.choice(train_items)
        loss, _out, _y_t, _y0, sigma, layout, kind, data, gap = step_loss(item, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        if not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"step {step}: loss {float(loss)}")
        optimizer.step()
        # Free the grads now, not at the next step: evaluation and saving run between
        # steps, and ~350 MB of live grads on top of the Adam state OOMed the step-250 eval.
        optimizer.zero_grad(set_to_none=True)
        record = {"step": step, "loss": float(loss.detach()), "data": float(data.detach()),
                  "gap": None if gap is None else float(gap.detach()), "sigma": float(sigma), "kind": kind,
                  "tokens": layout.count, "dense": layout.dense_count, "grad_norm": float(grad_norm),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if step % 25 == 0:
            log.flush()
            gap_text = "" if record["gap"] is None else f" gap={record['gap']:.4f}"
            print(f"LOT_H3 kind=train step={step} loss={record['loss']:.4f}{gap_text} sigma={record['sigma']:.3f} "
                  f"layout={kind} tokens={layout.count}/{layout.dense_count} "
                  f"s_per_step={(time.perf_counter() - started) / step:.2f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    save("")
    print(f"LOT_H3 kind=train_done steps={args.steps} seconds={time.perf_counter() - started:.0f} ok=1", flush=True)


if __name__ == "__main__":
    main()
