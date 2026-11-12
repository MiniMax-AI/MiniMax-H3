#!/usr/bin/env python3
"""MPEG-Diffusion's own arrangement on H3: a native LR track plus HR tokens on a selected region.

    python3 lot/two_track_h3.py render --out DIR [--steps 20 --switch 18 --hr-steps 6] [--maps DIR/up_maps_tt.pt]
                                       [--region DIR/region_head_tt.pt | --box 0.25 0.2 0.75 0.9]
    python3 lot/two_track_h3.py fit-kv --out DIR [--items 24]     # ridge fit of f_up, base model at LR and HR
    python3 lot/two_track_h3.py region --out DIR [--items 200]    # linear region head on the LR track's states

Paper: Menn et al., MPEG-Diffusion: Mixed-Resolution Video Generation, arXiv:2611.12069.

``mixed_res_h3.py`` puts the paper's pieces on LoT's merged tokens, so its coarse region is only as
good as the LoT adapter. Here the LR track is what the paper uses: the unmodified base model at
half resolution (no adapter, no LoRA), which is good from the start. H3's rotary positions are
normalised by ``sqrt(area)``, so the LR and HR grids already share one coordinate frame.

Sampler (their Sec. 4, "Inference"):

1. ``--switch`` steps denoise the whole frame at LR (``W/2 x H/2``).
2. At the switch a region ``M`` is chosen on the LR token grid (``--region`` head, or ``--box``).
3. The clean LR estimate is upscaled, re-noised with fresh noise, and the remaining steps run one
   packed sequence ``[text, audio, LR video (whole grid), HR video (M only)]`` with dual-resolution
   attention (``mixed_res.two_track_attention``): LR queries see only LR keys, so the LR track stays
   the native model; HR queries see their own keys inside ``M`` and ``f_up`` of the LR keys outside.
4. The LR latent is upscaled and the HR latent is pasted over ``M``; the VAE decodes that.

The upscaler ``U`` is the VAE itself (decode, bicubic x2, encode): it needs no training and returns
a latent the decoder already agrees with. ``--feather`` blends the paste edge in latent space.
Not trained here (the paper fine-tunes them): the DiT with the maps, the resolution embedding,
``f_down`` (unused: the LR track is complete), the learned upscaler and the mixer.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import refuse_if_busy  # noqa: E402
from mixed_res import (RegionHead, TwoTrackPlan, UpMaps, patched_attention, region_mask,  # noqa: E402
                       two_track_attention)
from procrustes_h3 import CHECKPOINT  # noqa: E402

HEAD_DIM = 128
MID_BLOCK = 25
EXTENT = (1, 2, 2)


# ------------------------------------------------------------------------- forward


def two_track_forward(dit, lr: torch.Tensor, hr: torch.Tensor | None, plan: TwoTrackPlan | None, t: float,
                      text: torch.Tensor, *, maps: UpMaps | None = None, audio_generator=None,
                      capture: dict | None = None, res_embed: "ResolutionEmbedding | None" = None):
    """Velocities ``(v_lr, v_hr)`` of the LR latent and of the HR latent on the plan's region.

    ``lr`` is ``(1, 24, T, h, w)``, ``hr`` is ``(1, 24, T, 2h, 2w)``; ``v_hr`` is zero outside the
    region. With ``plan=None`` (or an empty region) this is the base model on ``lr`` alone.
    ``capture={"block": i}`` returns that block's LR video states in ``capture["states"]``;
    ``capture={"kv": {}}`` collects every block's pre-RoPE LR keys and values.
    ``res_embed`` adds the paper's learned LR / HR vectors to the video rows.

    With grad enabled (training) the attention patch is installed for good (``_training_patch``)
    so activation checkpointing recomputes the same attention in the backward pass.
    """
    from h3_musubi import _layout, _silence, _sync_swap, audio_sigma
    from musubi_tuner.minimax_h3.packing import (H3RowSegment, _unpack_video_rows, build_position_grid,
                                                 build_timestep_rows, pack_audio_rows, pack_video_rows)

    if text.ndim == 2:
        text = text.unsqueeze(0)
    _sync_swap(dit)
    device = dit.video_patch_proj.weight.device
    sigma_a = float(audio_sigma(1.0 - t))
    _b, _c, frames, lat_h, lat_w = lr.shape
    layout = _layout(text.shape[1], frames, lat_h, lat_w)
    segment = layout.target_video_segment
    if segment.stop != layout.row_count:
        raise RuntimeError("the target video must be the last segment of the packed sequence")
    lr_count = segment.row_count
    hr_count = plan.hr_count if plan is not None else 0
    if plan is not None and plan.lr_count != lr_count:
        raise ValueError(f"plan has {plan.lr_count} LR tokens, the latent has {lr_count}")

    rows = pack_video_rows(lr.to(device, torch.float32))
    positions = build_position_grid(layout)[0]
    hr_rotation = None
    if hr_count:
        hr_layout = _layout(text.shape[1], frames, 2 * lat_h, 2 * lat_w)
        hr_positions = build_position_grid(hr_layout)[0][-4 * lr_count:]
        rows = torch.cat([rows, pack_video_rows(hr.to(device, torch.float32))[:, plan.hr_index]], dim=1)
        positions = torch.cat([positions, hr_positions[plan.hr_index.cpu()]])
    video = dit.video_patch_proj(rows).to(dit.dtype)
    if res_embed is not None:
        video = res_embed(video, lr_count)
    audio = _silence(frames, sigma_a, device, audio_generator)
    audio_rows = dit.audio_patch_proj(pack_audio_rows(audio.to(device)).to(torch.float32)).to(dit.dtype)
    text_rows = dit.token_refiner(dit.condition_proj(text.to(device, dit.dtype)))
    hidden = torch.cat([text_rows, audio_rows, video], dim=1)

    segments = layout.segments[:-1] + (H3RowSegment(segment.role, segment.kind, segment.start,
                                                    segment.start + lr_count + hr_count),)
    packed = replace(layout, segments=segments, row_count=segment.start + lr_count + hr_count)
    rotation = dit._rotation_table(positions.unsqueeze(0).to(device), hidden.dtype)
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    timesteps = build_timestep_rows(packed, text_token_tags=tags, model_t_video=t, model_t_audio=1.0 - sigma_a)
    temb = dit._timestep_embeddings(timesteps.unique_timesteps, device)

    patch = contextlib.nullcontext()
    want_kv = capture is not None and "kv" in capture
    training = torch.is_grad_enabled()
    persistent = training or getattr(dit, "_two_track_context", None) is not None
    if hr_count or want_kv or persistent:
        if hr_count:
            hr_rotation = dit._rotation_table(hr_positions.unsqueeze(0).to(device), hidden.dtype)
        context = dict(plan=plan if plan is not None and hr_count else _empty_plan(frames, lat_h, lat_w, device),
                       maps=maps, video_start=segment.start, hr_rotation=hr_rotation,
                       capture=capture.get("kv") if want_kv else None)
        if persistent:
            _training_patch(dit).update(context)
        else:
            patch = patched_attention(dit.blocks, attention=two_track_attention, **context)
    checkpoint = training and dit.gradient_checkpointing
    with patch:
        for index, block in enumerate(dit.blocks):
            if dit.blocks_to_swap:
                dit.offloader.wait_for_block(index)
            if checkpoint:
                hidden = torch.utils.checkpoint.checkpoint(block, hidden, temb, timesteps.block_segments, rotation,
                                                           use_reentrant=False)
            else:
                hidden = block(hidden, temb, timesteps.block_segments, rotation)
            if dit.blocks_to_swap:
                dit.offloader.submit_move_blocks_forward(dit.blocks, index)
            if capture is not None and capture.get("block") == index:
                states = hidden[0, segment.start:segment.start + lr_count].float()
                capture["states"] = states if capture.get("grad") else states.detach()
    hidden = hidden.to(device)

    final = dit.final_layer
    shift, scale = final.adaln_proj(temb)
    index = timesteps.video_timestep_index
    states = final.norm(hidden[:, segment.start:]) * (1.0 + scale[index]) + shift[index]
    out = final.video_out(states.to(torch.float32))
    v_lr = _unpack_video_rows(out[:, :lr_count], layout.target_video).to(lr.dtype)
    if not hr_count:
        return v_lr, None
    full = out.new_zeros(1, 4 * lr_count, out.shape[-1])
    full[:, plan.hr_index] = out[:, lr_count:]
    return v_lr, _unpack_video_rows(full, hr_layout.target_video).to(hr.dtype)


def _empty_plan(frames: int, lat_h: int, lat_w: int, device) -> TwoTrackPlan:
    return TwoTrackPlan(torch.zeros(frames, lat_h // 2, lat_w // 2, dtype=torch.bool)).to(device)


def _training_patch(dit) -> dict:
    """Route every block's attention through ``two_track_attention`` for the life of the process and
    return the shared context dict. A context manager would be gone before the backward pass, and
    checkpointed blocks recompute attention there."""
    context = getattr(dit, "_two_track_context", None)
    if context is None:
        import types

        context = {}
        for layer, block in enumerate(dit.blocks):
            def forward(self, hidden, rotation=None, _layer=layer):
                return two_track_attention(self, hidden, rotation, layer=_layer, **context)

            block.attn.forward = types.MethodType(forward, block.attn)
        dit._two_track_context = context
    context.clear()
    return context


class ResolutionEmbedding(torch.nn.Module):
    """The paper's resolution embedding: one learned vector for LR rows and one for HR rows, zero at init."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.lr = torch.nn.Parameter(torch.zeros(hidden_size))
        self.hr = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, video: torch.Tensor, lr_count: int) -> torch.Tensor:
        add = torch.cat([self.lr.expand(lr_count, -1), self.hr.expand(video.shape[1] - lr_count, -1)])
        return video + add.to(video.dtype).unsqueeze(0)


# ------------------------------------------------------------------------ upscaler


class VaeUpscaler:
    """``U``: decode, bicubic x2, encode. Stills only (one latent frame)."""

    def __init__(self, vae):
        self.vae = vae
        self.dtype = next(vae.parameters()).dtype

    @torch.no_grad()
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        pixels = self.vae.decode(latent.to("cuda", self.dtype))[:, :, 0].float().clamp(-1, 1)
        pixels = F.interpolate(pixels, scale_factor=2, mode="bicubic", align_corners=False).clamp(-1, 1)
        return self.vae.encode(pixels.to(self.dtype)).float()

    @torch.no_grad()
    def down(self, latent: torch.Tensor) -> torch.Tensor:
        """``D``: decode, area x0.5, encode (the LR latent of a cached HR item)."""
        pixels = self.vae.decode(latent.to("cuda", self.dtype))[:, :, 0].float().clamp(-1, 1)
        pixels = F.interpolate(pixels, scale_factor=0.5, mode="area")
        return self.vae.encode(pixels.to(self.dtype)).float()


def mask_to_latent(mask: torch.Tensor, scale: int, feather: float = 0.0) -> torch.Tensor:
    """LR-token mask ``(T, h, w)`` -> ``(1, 1, T, h*scale, w*scale)`` float, optionally feathered inward."""
    up = mask.float().repeat_interleave(scale, 1).repeat_interleave(scale, 2)
    if feather > 0:
        # Erode by the feather radius, then blur: the blend stays inside the region the HR track denoised.
        radius = int(round(feather))
        eroded = -F.max_pool2d(-up[:, None], 2 * radius + 1, stride=1, padding=radius)
        kernel = torch.ones(1, 1, 2 * radius + 1, 2 * radius + 1) / (2 * radius + 1) ** 2
        up = F.conv2d(F.pad(eroded, (radius,) * 4, mode="replicate"), kernel)[:, 0]
    return up[None, None]


# ------------------------------------------------------------------------- sampler


def box_mask(box, frames: int, height: int, width: int) -> torch.Tensor:
    x0, y0, x1, y1 = box
    mask = torch.zeros(frames, height, width, dtype=torch.bool)
    mask[:, round(y0 * height):round(y1 * height), round(x0 * width):round(x1 * width)] = True
    return mask


@torch.no_grad()
def native_sample(dit, text, *, height, width, steps, seed, stop=None, capture_at=None, block=MID_BLOCK,
                  res_embed=None):
    """Base-model Euler sampling; returns ``(x, x0_last, states)`` after ``stop`` steps (default all)."""
    from h3_musubi import sigma_schedule

    gen = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(1, 24, 1, height // 16, width // 16, device="cuda", generator=gen)
    audio_gen = torch.Generator("cuda").manual_seed(seed + 1)
    sigmas = sigma_schedule(steps)
    x0 = states = None
    for index in range(steps if stop is None else stop):
        sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
        capture = {"block": block} if capture_at == index else None
        velocity, _ = two_track_forward(dit, x, None, None, 1.0 - sigma, text, audio_generator=audio_gen,
                                        capture=capture, res_embed=res_embed)
        if capture is not None:
            states = capture["states"]
        x0 = x + sigma * velocity.float()
        x = x0 + (sigma_next / sigma) * (x - x0)
    return x, x0, states


@torch.no_grad()
def two_track_sample(dit, text, upscale, *, height, width, steps, switch, seed, maps=None, head=None, box=None,
                     block=MID_BLOCK, alpha=0.5, floor=None, feather=0.0, on_mask=None, hr_steps=None,
                     res_embed=None, decoder=None):
    """The four-stage sampler of the module docstring. Returns ``(latent, info)`` at ``width x height``.

    ``decoder`` is ``(U, G)`` from ``train_tt.py decoder``: ``U`` replaces the VAE upscale (at the
    switch and at the end) and ``G`` merges the tracks (their eqs. 5-6) instead of the feathered paste."""
    from h3_musubi import sigma_schedule

    lat_h, lat_w = height // 32, width // 32            # LR latent; also the HR token grid
    tok_h, tok_w = lat_h // 2, lat_w // 2               # LR token grid
    sigmas = sigma_schedule(steps)
    lr, x0_lr, states = native_sample(dit, text, height=height // 2, width=width // 2, steps=steps, seed=seed,
                                      stop=switch, capture_at=switch - 1 if head is not None else None, block=block,
                                      res_embed=res_embed)
    up = upscale if decoder is None else decoder[0]
    if head is not None:
        scores = head(states).float().cpu().reshape(1, tok_h, tok_w)
        mask = region_mask(scores, alpha=alpha, floor=floor)
    else:
        mask = box_mask(box or (0.25, 0.25, 0.75, 0.75), 1, tok_h, tok_w)
    if on_mask is not None:
        on_mask(mask)
    tokens = [tok_h * tok_w] * switch
    if not mask.any() or switch >= steps:
        return up(lr), {"mask": mask, "tokens": tokens + [tok_h * tok_w] * (steps - switch)}
    plan = TwoTrackPlan(mask).to("cuda")
    gen = torch.Generator("cuda").manual_seed(seed + 2)
    audio_gen = torch.Generator("cuda").manual_seed(seed + 3)
    sigma = float(sigmas[switch])
    hr = (1.0 - sigma) * up(x0_lr) + sigma * torch.randn(1, 24, 1, 2 * lat_h, 2 * lat_w, device="cuda",
                                                              generator=gen)
    # After the switch both tracks share the schedule: the model's own, or ``hr_steps`` steps
    # linear in sigma (H3's shift 12 leaves few low-sigma steps, where the HR detail is made).
    tail = sigmas[switch:] if not hr_steps else torch.linspace(sigma, 0.0, hr_steps + 1, dtype=torch.float64)
    for index in range(len(tail) - 1):
        sigma, sigma_next = float(tail[index]), float(tail[index + 1])
        v_lr, v_hr = two_track_forward(dit, lr, hr, plan, 1.0 - sigma, text, maps=maps, audio_generator=audio_gen,
                                       res_embed=res_embed)
        ratio = sigma_next / sigma
        lr = lr + (1.0 - ratio) * sigma * v_lr.float()
        hr = hr + (1.0 - ratio) * sigma * v_hr.float()
        tokens.append(plan.lr_count + plan.hr_count)
    if decoder is not None:
        upscaler, mixer = decoder
        inside = mask_to_latent(mask, 2).to("cuda")                     # LR latent grid
        filled = (1.0 - inside) * lr + inside * upscale.down(hr)        # their eq. 5: fill M with D(z_h)
        latent = mixer(hr, upscaler(filled), mask_to_latent(mask, 4).to("cuda"))
        return latent, {"mask": mask, "tokens": tokens}
    weight = mask_to_latent(mask, 4, feather).to("cuda")
    latent = weight * hr + (1.0 - weight) * upscale(lr)
    return latent, {"mask": mask, "tokens": tokens}


# -------------------------------------------------------------------------- phases


def load_base(swap: int):
    from h3_musubi import load_dit

    return load_dit(CHECKPOINT, device="cuda", swap=swap)


def load_maps(path: Path) -> UpMaps:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    maps = UpMaps(saved["layers"], HEAD_DIM, [EXTENT], rank=saved["rank"])
    maps.load_state_dict(saved["state"])
    return maps.to("cuda").eval()


def load_trained(dit, train_dir: Path, tag: str = ""):
    """LoRA (on), up-maps and resolution embedding from a ``train_tt.py lora`` run. Returns ``(network, maps, res_embed)``."""
    from safetensors import safe_open

    from h3_musubi import make_lora

    with safe_open(str(train_dir / f"lora{tag}.safetensors"), "pt") as handle:
        rank = int((handle.metadata() or {}).get("rank", 16))
    network = make_lora(dit, rank)
    info = network.load_weights(str(train_dir / f"lora{tag}.safetensors"))
    if info.missing_keys:
        raise RuntimeError(f"LoRA load missed {len(info.missing_keys)} keys")
    network.to(device="cuda", dtype=torch.bfloat16).eval()
    saved = torch.load(train_dir / f"tt_state{tag}.pt", map_location="cpu", weights_only=False)
    maps = UpMaps(saved["layers"], HEAD_DIM, [EXTENT], rank=saved["rank"])
    maps.load_state_dict(saved["maps"])
    res_embed = ResolutionEmbedding(saved["hidden"])
    res_embed.load_state_dict(saved["res_embed"])
    return network, maps.to("cuda").eval(), res_embed.to("cuda").eval()


def load_decoder(path: Path):
    from mixed_res import LatentUpscaler, TwoTrackMixer

    saved = torch.load(path, map_location="cpu", weights_only=False)
    upscaler, mixer = LatentUpscaler(width=saved["up_width"]), TwoTrackMixer(width=saved["mix_width"])
    upscaler.load_state_dict(saved["upscaler"])
    mixer.load_state_dict(saved["mixer"])
    return upscaler.cuda().eval(), mixer.cuda().eval()


def load_upscaler() -> VaeUpscaler:
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    return VaeUpscaler(load_vae(LOT_H3_VAE))


def held_prompts(args):
    from render_h3 import load_prompts
    from train_h3 import CACHE, list_items

    held = [te for _latent, te in list_items(args.cache or CACHE, args.holdout)[1]]
    return load_prompts(args.prompts, held)


@torch.no_grad()
def render(args) -> None:
    from PIL import Image, ImageDraw
    from h3_musubi import sample
    from render_h3 import to_image

    from h3_musubi import set_lora

    dit = load_base(args.swap)
    upscale = load_upscaler()
    maps = load_maps(args.maps) if args.maps else None
    network = res_embed = None
    if args.train:
        network, maps, res_embed = load_trained(dit, args.train, args.tag)
    decoder = load_decoder(args.decoder) if args.decoder else None
    head = None
    if args.region:
        saved = torch.load(args.region, map_location="cpu", weights_only=False)
        head = RegionHead(saved["hidden"]).cuda().eval()
        head.load_state_dict(saved["state"])
        args.block = saved["block"]
    width, height = args.size
    columns = [c for c in ("hr", "lr", "two_track") if c in args.variants]
    meta = {"timings": {c: [] for c in columns}, "tokens": {c: [] for c in columns}, "hr_share": []}
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (caption, embeds) in enumerate(held_prompts(args)):
        embeds = embeds.to("cuda", torch.bfloat16)
        row = []
        for name in columns:
            mask = None
            torch.cuda.synchronize()
            start = time.perf_counter()
            if network is not None:
                set_lora(network, name == "two_track")      # the baselines are the base model
            if name == "hr":
                latent = sample(dit, embeds, height=height, width=width, steps=args.steps, seed=args.seed + index)
                count = float((height // 32) * (width // 32))
            elif name == "lr":
                latent, _x0, _s = native_sample(dit, embeds, height=height // 2, width=width // 2, steps=args.steps,
                                                seed=args.seed + index)
                latent = upscale(latent)
                count = float((height // 64) * (width // 64))
            else:
                latent, info = two_track_sample(
                    dit, embeds, upscale, height=height, width=width, steps=args.steps, switch=args.switch,
                    seed=args.seed + index, maps=maps, head=head, box=args.box, block=args.block,
                    alpha=args.alpha, floor=args.floor, feather=args.feather, hr_steps=args.hr_steps,
                    res_embed=res_embed, decoder=decoder)
                mask = info["mask"]
                count = sum(info["tokens"]) / len(info["tokens"])
                meta["hr_share"].append(round(float(mask.float().mean()), 4))
            torch.cuda.synchronize()
            meta["timings"][name].append(round(time.perf_counter() - start, 2))
            meta["tokens"][name].append(round(count, 1))
            pixels = upscale.vae.decode(latent.to("cuda", upscale.dtype))
            image = to_image(pixels)
            if mask is not None and args.outline:
                image = outline(image, mask)
            image.save(args.out / f"p{index}_{name}.png")
            row.append(image)
            print(f"LOT_H3 kind=two_track_render prompt={index} variant={name} "
                  f"seconds={meta['timings'][name][-1]} mean_tokens={count:.0f}", flush=True)
        rows.append(row)
    scale = 0.5
    cell = (int(width * scale), int(height * scale))
    sheet = Image.new("RGB", (cell[0] * len(columns), 28 + cell[1] * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(columns):
        draw.text((col * cell[0] + 6, 8), label, fill="black")
    for r, images in enumerate(rows):
        for col, image in enumerate(images):
            sheet.paste(image.resize(cell, Image.LANCZOS), (col * cell[0], 28 + r * cell[1]))
    sheet.save(args.out / "grid.png")
    meta.update({"columns": columns, "steps": args.steps, "switch": args.switch, "canvas": f"{width}x{height}",
                 "maps": str(args.maps), "train": str(args.train), "decoder": str(args.decoder), "region": str(args.region), "box": args.box, "feather": args.feather})
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"LOT_H3 kind=two_track_render_done out={args.out} ok=1", flush=True)


def outline(image, mask: torch.Tensor):
    """Draw the region's boundary (one LR token is 64 px)."""
    from PIL import ImageDraw

    image = image.copy()
    draw = ImageDraw.Draw(image)
    grid = mask[0]
    cell = image.height // grid.shape[0]
    for u in range(grid.shape[0]):
        for v in range(grid.shape[1]):
            if not grid[u, v]:
                continue
            x, y = v * cell, u * cell
            if u == 0 or not grid[u - 1, v]:
                draw.line([(x, y), (x + cell, y)], fill="white", width=2)
            if u == grid.shape[0] - 1 or not grid[u + 1, v]:
                draw.line([(x, y + cell), (x + cell, y + cell)], fill="white", width=2)
            if v == 0 or not grid[u, v - 1]:
                draw.line([(x, y), (x, y + cell)], fill="white", width=2)
            if v == grid.shape[1] - 1 or not grid[u, v + 1]:
                draw.line([(x + cell, y), (x + cell, y + cell)], fill="white", width=2)
    return image


def train_items(args):
    from train_h3 import CACHE, list_items

    train, _held = list_items(args.cache or CACHE, args.holdout)
    random.Random(args.seed).shuffle(train)
    return train[:args.items]


def crop_even(x0: torch.Tensor) -> torch.Tensor:
    """Crop a cached latent so its LR version has an even latent size (HR latent sides multiple of 4)."""
    height, width = x0.shape[-2] // 4 * 4, x0.shape[-1] // 4 * 4
    return x0[..., :height, :width]


@torch.no_grad()
def fit_kv(args) -> None:
    """Ridge fit of ``f_up`` (their Sec. 3.1): the base model's keys/values at LR and at HR on the
    same images and noise level, each LR token paired with its four HR tokens."""
    from train_h3 import load_item

    dit = load_base(args.swap)
    layers = len(dit.blocks)
    maps = UpMaps(layers, HEAD_DIM, [EXTENT], rank=args.rank).cuda()
    moments: dict = {}
    if args.pairs:
        from build_pairs_tt import list_pairs, load_pair

        sources = [("pair", entry) for entry in list_pairs(args.pairs, 8)[0][:args.items]]
    else:
        upscale = load_upscaler()
        sources = [("cache", item) for item in train_items(args)]
    for index, (kind, item) in enumerate(sources):
        if kind == "pair":
            pair = load_pair(item)
            hr0, lr0, text = pair["hr"], pair["lr"], pair["text"]
        else:
            x0, text = load_item(item, random.Random(index), "cuda")
            hr0 = crop_even(x0)
            lr0 = upscale.down(hr0)
        plan = TwoTrackPlan(torch.zeros(1, lr0.shape[-2] // 2, lr0.shape[-1] // 2, dtype=torch.bool))
        cells = torch.argsort(plan.cell_order).reshape(plan.lr_count, 4).cuda()      # LR token -> its HR rows
        for sigma in args.sigmas:
            gen = torch.Generator("cuda").manual_seed(index)
            kv_lr, kv_hr = {}, {}
            x_lr = (1.0 - sigma) * lr0 + sigma * torch.randn(lr0.shape, device="cuda", generator=gen)
            x_hr = (1.0 - sigma) * hr0 + sigma * torch.randn(hr0.shape, device="cuda", generator=gen)
            two_track_forward(dit, x_lr, None, None, 1.0 - sigma, text, capture={"kv": kv_lr})
            two_track_forward(dit, x_hr, None, None, 1.0 - sigma, text, capture={"kv": kv_hr})
            for layer in kv_lr:
                for which, slot in (("k", 0), ("v", 1)):
                    x = kv_lr[layer][slot][0].float().reshape(-1, HEAD_DIM)                       # (N*H, d)
                    y = kv_hr[layer][slot][0].float()[cells].transpose(1, 2).reshape(-1, 4, HEAD_DIM)
                    delta = (y - x.unsqueeze(1)).reshape(x.shape[0], -1)
                    entry = moments.setdefault((layer, which), [0.0, 0.0, 0, 0.0, 0.0])
                    entry[0] = entry[0] + x.T @ x
                    entry[1] = entry[1] + x.T @ delta
                    entry[2] += x.shape[0]
                    entry[3] += float(delta.pow(2).sum())                    # copy-init residual
                    entry[4] += float(y.pow(2).sum())
        print(f"LOT_H3 kind=tt_fit_kv item={index}", flush=True)
    report = {"k": [], "v": []}
    for (layer, which), (xtx, xty, count, copy_err, energy) in moments.items():
        maps.fit_moments_(layer, EXTENT, which, xtx, xty, count, args.ridge)
        name = f"l{layer}_2x2_{which}"
        weight = maps.down[name].float() @ maps.up[name].float()
        # |Y - XW|^2 = |Y|^2 - 2 tr(W^T X^T Y) + tr(W^T X^T X W), on the training moments.
        fit_err = copy_err - 2.0 * float((weight * xty).sum()) + float((weight * (xtx @ weight)).sum())
        report[which].append((layer, (copy_err / energy) ** 0.5, (max(fit_err, 0.0) / energy) ** 0.5))
    for which in ("k", "v"):
        rows = sorted(report[which])
        copy = sum(r[1] for r in rows) / len(rows)
        fit = sum(r[2] for r in rows) / len(rows)
        print(f"LOT_H3 kind=tt_fit_kv_residual which={which} copy_rel={copy:.3f} fit_rel={fit:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": maps.state_dict(), "rank": args.rank, "layers": layers, "sigmas": args.sigmas,
                "residual": report}, args.out / "up_maps_tt.pt")
    print(f"LOT_H3 kind=tt_fit_kv_done out={args.out / 'up_maps_tt.pt'} ok=1", flush=True)


def train_region(args) -> None:
    """Linear head on the LR track's block states at the switch. Label (their self-supervised one, in
    latent space): the ``--fraction`` of LR tokens where the upscaled LR clean estimate is furthest
    from the true HR latent."""
    from h3_musubi import sigma_schedule
    from train_h3 import load_item

    dit = load_base(args.swap)
    upscale = load_upscaler()
    sigma = float(sigma_schedule(args.steps)[args.switch - 1])
    pairs = []
    with torch.no_grad():
        for index, item in enumerate(train_items(args)):
            x0, text = load_item(item, random.Random(index), "cuda")
            hr0 = crop_even(x0)
            lr0 = upscale.down(hr0)
            gen = torch.Generator("cuda").manual_seed(index)
            x_lr = (1.0 - sigma) * lr0 + sigma * torch.randn(lr0.shape, device="cuda", generator=gen)
            capture = {"block": args.block}
            velocity, _ = two_track_forward(dit, x_lr, None, None, 1.0 - sigma, text, capture=capture)
            error = (upscale(x_lr + sigma * velocity.float()) - hr0).abs().mean(dim=1)       # (1, 4h, 4w) latent px
            error = F.avg_pool2d(error, 4).flatten()                                          # LR tokens
            label = (error >= torch.quantile(error, 1.0 - args.fraction)).float()
            pairs.append((capture["states"].cpu(), label.cpu()))
            print(f"LOT_H3 kind=tt_region_pair item={index} tokens={label.numel()}", flush=True)
    del dit
    torch.cuda.empty_cache()
    head = RegionHead(pairs[0][0].shape[-1]).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    held = pairs[: max(1, len(pairs) // 10)]
    train = pairs[len(held):] or pairs
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        total = 0.0
        for states, target in train:
            loss = F.binary_cross_entropy_with_logits(head(states.cuda()), target.cuda())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        with torch.no_grad():
            held_bce = sum(float(F.binary_cross_entropy_with_logits(head(s.cuda()), t.cuda())) for s, t in held)
        print(f"LOT_H3 kind=tt_region_epoch epoch={epoch} train_bce={total / len(train):.4f} "
              f"held_bce={held_bce / len(held):.4f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "hidden": pairs[0][0].shape[-1], "block": args.block,
                "switch": args.switch, "steps": args.steps, "fraction": args.fraction},
               args.out / "region_head_tt.pt")
    print(f"LOT_H3 kind=tt_region_done out={args.out / 'region_head_tt.pt'} ok=1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("render", "fit-kv", "region"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None, help="still cache with text (prompts and fit data)")
    parser.add_argument("--holdout", type=int, default=24)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", default="auto", help="DiT blocks streamed from CPU; auto (by free VRAM): 0 at 64 GB+, 8 at 28+, 16 at 21+, else 24")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--switch", type=int, default=18, help="LR-only steps before the HR region starts")
    parser.add_argument("--hr-steps", type=int, default=6,
                        help="steps after the switch, linear in sigma (0: the rest of the schedule)")
    parser.add_argument("--size", type=int, nargs=2, default=(1344, 768), metavar=("W", "H"))
    parser.add_argument("--variants", default="hr,lr,two_track")
    parser.add_argument("--maps", type=Path, default=None, help="up_maps_tt.pt from fit-kv (default: copy)")
    parser.add_argument("--train", type=Path, default=None,
                        help="train_tt.py lora output dir: LoRA + fine-tuned maps + resolution embedding (overrides --maps)")
    parser.add_argument("--tag", default="", help="checkpoint tag in --train, e.g. _1000")
    parser.add_argument("--decoder", type=Path, default=None, help="decoder.pt from train_tt.py decoder (U and G)")
    parser.add_argument("--pairs", type=Path, action="append", default=None,
                        help="fit-kv: build_pairs_tt.py dirs (HR + LR latents precomputed; repeatable)")
    parser.add_argument("--region", type=Path, default=None, help="region_head_tt.pt (default: --box)")
    parser.add_argument("--box", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument("--feather", type=float, default=4.0, help="paste-edge blend radius, latent pixels")
    parser.add_argument("--outline", action="store_true", help="draw the region boundary on the two_track image")
    parser.add_argument("--block", type=int, default=MID_BLOCK)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=HEAD_DIM, help="rank of the f_up delta (head dim = full)")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 0.95],
                        help="fit-kv: noise levels; cover the sigmas the HR track runs at after the switch")
    args = parser.parse_args()
    from gpu_guard import auto_swap

    args.swap = auto_swap(args.swap)
    if args.size[0] % 64 or args.size[1] % 64:
        raise SystemExit("--size sides must be multiples of 64 (the LR latent needs whole 2x2 patches)")
    refuse_if_busy("two_track_h3")
    {"render": render, "fit-kv": fit_kv, "region": train_region}[args.phase](args)


if __name__ == "__main__":
    main()
