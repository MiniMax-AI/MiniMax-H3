#!/usr/bin/env python3
"""MPEG-Diffusion's three additions on H3 LoT (``mixed_res``): fit, train, render.

    python3 scripts/lot/mixed_res_h3.py fit-kv  --run RUN --out DIR [--items 24]   # ridge init of the dual-K/V up-maps
    python3 scripts/lot/mixed_res_h3.py region  --run RUN --out DIR [--items 200]  # self-selected region head
    python3 scripts/lot/mixed_res_h3.py mixer   --run RUN --out DIR [--items 200]  # seam mixer on the final latent
    python3 scripts/lot/mixed_res_h3.py render  --run RUN --out DIR [--region DIR/region_head.pt]
                                        [--dual DIR/up_maps.pt] [--mixer DIR/mixer.pt]

``RUN`` is a ``train_h3`` output dir (``adapter.pt`` or ``adapter.safetensors``, ``lora.safetensors``).
Paper: Menn et al., MPEG-Diffusion: Mixed-Resolution Video Generation, arXiv:2611.12069.

fit-kv: for each item and sigma, one LoT forward on a coarse layout and one on the dense layout
(same ``y_t``, LoRA on) capture every block's pre-RoPE video keys and values. Each coarse token
is paired with the dense keys/values at its cells; ``X^T X`` and ``X^T (cells - X)`` accumulate
per block, extent and K/V, and ``UpMaps.fit_moments_`` solves the ridge problem (their init).

region: at the switch point of the sampler (``--switch`` of ``--steps``), a coarse-layout
forward captures block ``--block``'s video states; the dense model's clean estimate at the same
``x_t`` labels the tokens whose coarse estimate is furthest from it (``region_labels``); the
head is trained with BCE on the per-token label share. Pairs are computed once, then the head
trains for ``--epochs`` on them (it is ~5k parameters).

mixer: content layouts (``box``/``focus``/``texture``) at low sigma; target is the dense
model's clean estimate, input the LoT one; latent L1 outside the 1x1 region (``mixer_loss``).

render: dense, plain LoT on ``--coarse``, and the adaptive sampler: ``--switch`` steps on the
coarse layout, then the region head picks a per-latent-frame layout for the rest (with
``--dual`` and ``--mixer`` when given). One grid plus ``meta.json`` (timings, token counts).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import refuse_if_busy  # noqa: E402
from h3 import H3_EXTENTS, make_h3_adapter  # noqa: E402
from h3_splice import LotSplice  # noqa: E402
from layout import dense_layout  # noqa: E402
from mixed_res import (DualKV, RegionHead, SeamMixer, UpMaps, mixer_loss, region_labels,  # noqa: E402
                       region_layout, region_mask, scores_to_cells, token_targets)
from procrustes_h3 import CHECKPOINT  # noqa: E402

HEAD_DIM = 128
MID_BLOCK = 25


def load_run(run: Path, swap: int, device="cuda"):
    """DiT + LoRA (on) + trained adapter from a ``train_h3`` dir."""
    from safetensors import safe_open

    from h3_musubi import load_dit, make_lora
    from train_h3 import load_adapter_state

    adapter = make_h3_adapter()
    adapter.load_state_dict(load_adapter_state(run))
    adapter.to(device).requires_grad_(False)
    with safe_open(str(run / "lora.safetensors"), "pt") as handle:
        rank = int((handle.metadata() or {}).get("rank", 16))
    dit = load_dit(CHECKPOINT, device=device, swap=swap)
    network = make_lora(dit, rank)
    info = network.load_weights(str(run / "lora.safetensors"))
    if info.missing_keys:
        raise RuntimeError(f"LoRA load missed {len(info.missing_keys)} keys")
    network.to(device=device, dtype=torch.bfloat16).eval()
    return dit, network, adapter


def coarse_layout(time_: int, height: int, width: int, side: int):
    from train_h3 import _tile

    return _tile(time_, height, width, lambda *_: (1, side, side))


def items_for(args):
    from train_h3 import CACHE, list_items

    train, held = [], []
    for cache in args.cache or [CACHE]:
        part_train, part_held = list_items(cache, args.holdout)
        train += part_train
        held += part_held
    rng = random.Random(args.seed)
    rng.shuffle(train)
    return train[:args.items], held


def noisy(splice, x0: torch.Tensor, sigma: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(y_t, x_t)`` with one shared ``eps``, as ``train_h3`` pairs the LoT student and the dense
    teacher: ``y_t = (1 - sigma) to_y(x0) + sigma eps`` and ``x_t = (1 - sigma) x0 + sigma eps``."""
    gen = torch.Generator(x0.device).manual_seed(seed)
    eps = torch.randn(x0.shape, device=x0.device, generator=gen)
    return (1.0 - sigma) * splice.to_y(x0) + sigma * eps, (1.0 - sigma) * x0 + sigma * eps


def clean_y(splice, dit, y_t, sigma, text, **kw) -> torch.Tensor:
    """The LoT clean estimate in the splice's y-space."""
    from h3_musubi import predict

    out = predict(dit, y_t, 1.0 - sigma, text, lot=splice, audio_generator=torch.Generator(y_t.device).manual_seed(7), **kw)
    return y_t.float() + sigma * out.float()


def clean(splice, dit, y_t, sigma, text, **kw) -> torch.Tensor:
    """The LoT clean estimate mapped back to the pretrained latent."""
    return splice.to_x(clean_y(splice, dit, y_t, sigma, text, **kw))


# --------------------------------------------------------------------------- fit-kv


def cell_groups(layout, device="cuda"):
    """Per larger extent: its token rows ``(N,)`` and the dense cell indices they cover ``(N, n)``."""
    width, plane = layout.width, layout.height * layout.width
    groups = []
    for extent, indices in layout.groups():
        if extent == (1, 1, 1):
            continue
        cells = [[r.t * plane + (r.u + du) * width + r.v + dv for du in range(r.eh) for dv in range(r.ew)]
                 for r in (layout.rects[i] for i in indices)]
        groups.append((extent, torch.tensor(indices, device=device), torch.tensor(cells, device=device)))
    return groups


@torch.no_grad()
def fit_kv(args) -> None:
    from train_h3 import load_item

    dit, _network, adapter = load_run(args.run, args.swap)
    maps = UpMaps(len(dit.blocks), HEAD_DIM, H3_EXTENTS, rank=args.dual_rank).cuda()
    moments: dict[str, list] = {}
    rows_seen = 0
    for index, item in enumerate(items_for(args)[0]):
        x0, text = load_item(item, random.Random(index), "cuda")
        _b, _c, tt, hh, ww = x0.shape
        dense = LotSplice(adapter, dense_layout(tt, hh // 2, ww // 2), y_space=True)
        coarse = LotSplice(adapter, coarse_layout(tt, hh // 2, ww // 2, (2, 4)[index % 2]), y_space=True)
        for sigma in args.sigmas:
            kv_dense, kv_coarse = {}, {}
            clean(dense, dit, noisy(dense, x0, sigma, index)[0], sigma, text, capture={"kv": kv_dense})
            clean(coarse, dit, noisy(coarse, x0, sigma, index)[0], sigma, text, capture={"kv": kv_coarse})
            groups = cell_groups(coarse.layout)
            for layer in kv_coarse:
                for which, slot in (("k", 0), ("v", 1)):
                    source = kv_coarse[layer][slot][0].float()                    # (L, H, d)
                    target = kv_dense[layer][slot][0].float()                     # (dense, H, d)
                    for extent, rows, cells in groups:
                        x = source[rows].reshape(-1, HEAD_DIM)                     # (N*H, d)
                        y = target[cells].transpose(1, 2).reshape(-1, cells.shape[1], HEAD_DIM)  # (N*H, n, d)
                        delta = (y - x.unsqueeze(1)).reshape(x.shape[0], -1)
                        key = f"{layer}|{extent}|{which}"
                        if key not in moments:
                            moments[key] = [torch.zeros(HEAD_DIM, HEAD_DIM, device="cuda"),
                                            torch.zeros(HEAD_DIM, delta.shape[1], device="cuda"), 0]
                        moments[key][0] += x.T @ x
                        moments[key][1] += x.T @ delta
                        moments[key][2] += x.shape[0]
            rows_seen += 1
        print(f"LOT_H3 kind=fit_kv item={index} pairs={rows_seen} keys={len(moments)}", flush=True)
    for key, (xtx, xty, count) in moments.items():
        layer, extent, which = key.split("|")
        maps.fit_moments_(int(layer), tuple(int(n) for n in extent.strip("()").split(",")), which,
                          xtx, xty, count, args.ridge)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": maps.state_dict(), "rank": args.dual_rank, "layers": len(dit.blocks),
                "run": str(args.run), "sigmas": args.sigmas}, args.out / "up_maps.pt")
    print(f"LOT_H3 kind=fit_kv_done out={args.out / 'up_maps.pt'} fitted={len(moments)} ok=1", flush=True)


def load_maps(path: Path, device="cuda") -> UpMaps:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    maps = UpMaps(saved["layers"], HEAD_DIM, H3_EXTENTS, rank=saved["rank"])
    maps.load_state_dict(saved["state"])
    return maps.to(device).eval()


# --------------------------------------------------------------------------- region


def switch_sigma(args) -> float:
    from h3_musubi import sigma_schedule

    return float(sigma_schedule(args.steps)[args.switch])


@torch.no_grad()
def dense_clean(dit, network, x_t, sigma, text):
    """The dense model's clean estimate (LoRA off: the base model)."""
    from h3_musubi import predict, set_lora

    set_lora(network, False)
    try:
        out = predict(dit, x_t, 1.0 - sigma, text, audio_generator=torch.Generator(x_t.device).manual_seed(7))
    finally:
        set_lora(network, True)
    return x_t.float() + sigma * out.float()


def train_region(args) -> None:
    from train_h3 import load_item

    dit, network, adapter = load_run(args.run, args.swap)
    sigma = switch_sigma(args)
    pairs = []
    with torch.no_grad():
        for index, item in enumerate(items_for(args)[0]):
            x0, text = load_item(item, random.Random(index), "cuda")
            _b, _c, tt, hh, ww = x0.shape
            splice = LotSplice(adapter, coarse_layout(tt, hh // 2, ww // 2, args.coarse), y_space=True)
            y_t, x_t = noisy(splice, x0, sigma, index)
            capture = {"block": args.block}
            x0_coarse = clean(splice, dit, y_t, sigma, text, capture=capture)
            labels = region_labels(x0_coarse, dense_clean(dit, network, x_t, sigma, text), args.fraction)
            pairs.append((capture["states"].cpu(), token_targets(labels.cpu(), splice.layout)))
            print(f"LOT_H3 kind=region_pair item={index} tokens={splice.layout.count}", flush=True)
    del dit, network
    gc.collect()
    torch.cuda.empty_cache()
    head = RegionHead(pairs[0][0].shape[-1]).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    held = pairs[: max(1, len(pairs) // 10)]
    train = pairs[len(held):] or pairs
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        total = 0.0
        for states, target in train:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(states.cuda()), target.cuda())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
        with torch.no_grad():
            held_loss = sum(float(torch.nn.functional.binary_cross_entropy_with_logits(head(s.cuda()), t.cuda()))
                            for s, t in held) / len(held)
        print(f"LOT_H3 kind=region_epoch epoch={epoch} train_bce={total / len(train):.4f} held_bce={held_loss:.4f}",
              flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "hidden": pairs[0][0].shape[-1], "block": args.block,
                "coarse": args.coarse, "switch": args.switch, "steps": args.steps, "fraction": args.fraction,
                "run": str(args.run)}, args.out / "region_head.pt")
    print(f"LOT_H3 kind=region_done out={args.out / 'region_head.pt'} ok=1", flush=True)


# ---------------------------------------------------------------------------- mixer


def train_mixer(args) -> None:
    from train_h3 import load_item, sample_layout

    dit, network, adapter = load_run(args.run, args.swap)
    dual = DualKV(load_maps(args.dual)) if args.dual else None
    pairs = []
    with torch.no_grad():
        for index, item in enumerate(items_for(args)[0]):
            x0, text = load_item(item, random.Random(index), "cuda")
            layout, kind = sample_layout(x0, random.Random(index), force=("box", "focus", "texture")[index % 3])
            splice = LotSplice(adapter, layout, y_space=True)
            sigma = random.Random(index).uniform(0.02, 0.3)
            y_t, x_t = noisy(splice, x0, sigma, index)
            student = clean(splice, dit, y_t, sigma, text, dual=dual)
            pairs.append((student.cpu(), dense_clean(dit, network, x_t, sigma, text).cpu(), layout))
            print(f"LOT_H3 kind=mixer_pair item={index} kind={kind} sigma={sigma:.3f}", flush=True)
    del dit, network
    gc.collect()
    torch.cuda.empty_cache()
    mixer = SeamMixer(width=args.mixer_width).cuda()
    optimizer = torch.optim.Adam(mixer.parameters(), lr=args.lr)
    held = pairs[: max(1, len(pairs) // 10)]
    train = pairs[len(held):] or pairs

    def held_loss(identity: bool) -> float:
        with torch.no_grad():
            return sum(float(mixer_loss(s.cuda() if identity else mixer(s.cuda(), lay), t.cuda(), lay))
                       for s, t, lay in held) / len(held)

    print(f"LOT_H3 kind=mixer_start identity_l1={held_loss(True):.5f}", flush=True)
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        total = 0.0
        for student, target, layout in train:
            loss = mixer_loss(mixer(student.cuda(), layout), target.cuda(), layout)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss)
        print(f"LOT_H3 kind=mixer_epoch epoch={epoch} train_l1={total / len(train):.5f} "
              f"held_l1={held_loss(False):.5f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": mixer.state_dict(), "width": args.mixer_width, "run": str(args.run)}, args.out / "mixer.pt")
    print(f"LOT_H3 kind=mixer_done out={args.out / 'mixer.pt'} ok=1", flush=True)


# --------------------------------------------------------------------------- render


@torch.no_grad()
def adaptive_sample(dit, adapter, text, *, height, width, steps, seed, coarse, switch, head=None, block=MID_BLOCK,
                    alpha=0.5, floor=None, dual=None, mixer=None, background=4):
    """H3's Euler update in the current layout's y-space (as ``h3_musubi.sample`` with a splice).

    Steps ``< switch`` run on the coarse layout; at the switch the region head picks the layout
    (per latent frame) for the rest. The update is written as ``y' = (1 - s') y0 + s' eps`` with
    ``eps = (y - (1 - s) y0) / s``, which is the Euler step; at the switch ``y0`` is re-expressed in
    the new layout's y-space (``to_y(to_x(y0))``) and ``eps`` is kept. Returns ``(x0, layout, tokens)``."""
    from h3_musubi import sigma_schedule

    gen = torch.Generator("cuda").manual_seed(seed)
    y = torch.randn(1, 24, 1, height // 16, width // 16, device="cuda", generator=gen)
    layout = coarse_layout(1, height // 32, width // 32, coarse)
    sigmas = sigma_schedule(steps)
    tokens = []
    for index in range(steps):
        sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
        splice = LotSplice(adapter, layout, y_space=True)
        capture = {"block": block} if head is not None and index == switch - 1 else None
        y0 = clean_y(splice, dit, y, sigma, text, capture=capture, dual=dual if index >= switch else None)
        tokens.append(layout.count)
        eps = (y.float() - (1.0 - sigma) * y0) / sigma
        if capture is not None:
            field = scores_to_cells(head(capture["states"]).float().cpu(), layout)
            layout = region_layout(region_mask(field, alpha=alpha, floor=floor), background=background)
            y0 = LotSplice(adapter, layout, y_space=True).to_y(splice.to_x(y0))
        y = (1.0 - sigma_next) * y0 + sigma_next * eps
    x = LotSplice(adapter, layout, y_space=True).to_x(y)
    if mixer is not None:
        x = mixer(x.float(), layout)
    return x, layout, tokens


def render(args) -> None:
    from h3_musubi import sample, set_lora
    from render_h3 import load_prompts
    from train_h3 import CACHE, list_items

    dit, network, adapter = load_run(args.run, args.swap)
    head = mixer = dual = None
    if args.region:
        saved = torch.load(args.region, map_location="cpu", weights_only=False)
        head = RegionHead(saved["hidden"]).cuda().eval()
        head.load_state_dict(saved["state"])
        args.block = saved["block"]
    if args.dual:
        dual = DualKV(load_maps(args.dual))
    if args.mixer:
        saved = torch.load(args.mixer, map_location="cpu", weights_only=False)
        mixer = SeamMixer(width=saved["width"]).cuda().eval()
        mixer.load_state_dict(saved["state"])
    held = [te for _latent, te in list_items(args.prompt_cache or CACHE, args.holdout)[1]]
    prompts = load_prompts(args.prompts, held)
    columns = ["dense", f"lot{args.coarse}x{args.coarse}"] + (["adaptive"] if head is not None else [])
    latents, meta = {}, {"timings": {c: [] for c in columns}, "tokens": {c: [] for c in columns}}
    for index, (_text, embeds) in enumerate(prompts):
        embeds = embeds.to("cuda", torch.bfloat16)
        for name in columns:
            start = time.perf_counter()
            if name == "dense":
                set_lora(network, False)
                latent = sample(dit, embeds, height=args.height, width=args.width, steps=args.steps,
                                seed=args.seed + index)
                set_lora(network, True)
                count = [(args.height // 32) * (args.width // 32)] * args.steps
            else:
                latent, layout, count = adaptive_sample(
                    dit, adapter, embeds, height=args.height, width=args.width, steps=args.steps,
                    seed=args.seed + index, coarse=args.coarse,
                    switch=args.switch if name == "adaptive" else args.steps + 1,
                    head=head if name == "adaptive" else None, block=args.block, alpha=args.alpha,
                    floor=args.floor, dual=dual if name == "adaptive" else None,
                    mixer=mixer if name == "adaptive" else None)
                fine = sum(r.eh * r.ew for r in layout.rects if r.extent == (1, 1, 1)) / layout.dense_count
                meta.setdefault("fine_share", {}).setdefault(name, []).append(round(fine, 4))
            torch.cuda.synchronize()
            meta["timings"][name].append(time.perf_counter() - start)
            meta["tokens"][name].append(sum(count) / len(count))
            latents[(index, name)] = latent.float().cpu()
            print(f"LOT_H3 kind=mixed_render prompt={index} variant={name} "
                  f"seconds={meta['timings'][name][-1]:.1f} mean_tokens={meta['tokens'][name][-1]:.0f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({f"p{i}_{n}": v for (i, n), v in latents.items()}, args.out / "latents.pt")
    meta.update({"columns": columns, "prompts": [p[0] for p in prompts], "run": str(args.run), "steps": args.steps,
                 "switch": args.switch, "coarse": args.coarse, "canvas": f"{args.width}x{args.height}",
                 "region": str(args.region), "dual": str(args.dual), "mixer": str(args.mixer)})
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"LOT_H3 kind=mixed_render_done out={args.out} ok=1", flush=True)
    # The DiT does not give its VRAM back inside this process (block-swap state), so replace the
    # process with the decode phase instead of decoding here.
    os.execv(sys.executable, [sys.executable, __file__, "decode", "--run", str(args.run), "--out", str(args.out)])


def decode(args) -> None:
    """VAE-decode ``latents.pt`` from ``render`` into PNGs and ``grid.png``."""
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE
    from render_h3 import grid, to_image

    meta = json.loads((args.out / "meta.json").read_text())
    latents = torch.load(args.out / "latents.pt", map_location="cpu")
    decoder = load_vae(LOT_H3_VAE)
    rows = []
    for index in range(len(meta["prompts"])):
        row = []
        for name in meta["columns"]:
            with torch.no_grad():
                pixels = decoder.decode(latents[f"p{index}_{name}"].to("cuda", next(decoder.parameters()).dtype))
            image = to_image(pixels)
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
        rows.append(row)
    grid(rows, meta["columns"]).save(args.out / "grid.png")
    print(f"LOT_H3 kind=mixed_decode_done out={args.out} ok=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("fit-kv", "region", "mixer", "render", "decode"))
    parser.add_argument("--run", type=Path, required=True, help="train_h3 output dir (adapter + lora)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", default=None)
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20, help="sampler steps (the switch is a step index)")
    parser.add_argument("--switch", type=int, default=6, help="steps on the coarse layout before the region head")
    parser.add_argument("--coarse", type=int, choices=(2, 4), default=2, help="coarse layout token side")
    parser.add_argument("--block", type=int, default=MID_BLOCK, help="block whose video states feed the head")
    parser.add_argument("--fraction", type=float, default=0.25, help="labelled share of tokens per frame (paper: 25%%)")
    parser.add_argument("--alpha", type=float, default=0.5, help="threshold m + alpha (p - m)")
    parser.add_argument("--floor", type=float, default=None, help="99th-percentile floor below which all stays coarse")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dual-rank", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    parser.add_argument("--mixer-width", type=int, default=64)
    parser.add_argument("--region", type=Path, default=None)
    parser.add_argument("--dual", type=Path, default=None)
    parser.add_argument("--mixer", type=Path, default=None)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--prompt-cache", type=Path, default=None)
    parser.add_argument("--size", type=int, nargs=2, default=(1344, 768), metavar=("W", "H"))
    args = parser.parse_args()
    args.width, args.height = args.size
    if args.width % 32 or args.height % 32:
        raise SystemExit("--size sides must be multiples of 32")
    if args.phase == "decode":                  # the child of render: its parent holds the guard
        decode(args)
        return
    refuse_if_busy("mixed_res_h3.py")
    {"fit-kv": fit_kv, "region": train_region, "mixer": train_mixer, "render": render}[args.phase](args)


if __name__ == "__main__":
    main()
