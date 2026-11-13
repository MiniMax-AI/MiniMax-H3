#!/usr/bin/env python3
"""Train the two-track (MPEG-Diffusion) pieces on H3 from ``build_pairs_tt.py`` pairs.

    LOT_H3_TRAIN=1 python3 lot/train_tt.py lora    --pairs P --out R --maps F/up_maps_tt.pt [--steps 2000]
    LOT_H3_TRAIN=1 python3 lot/train_tt.py region  --pairs P --out R [--train R]   # region head on face/subject masks
    LOT_H3_TRAIN=1 python3 lot/train_tt.py decoder --pairs P --out R               # upscaler U + mixer G

Paper: Dennis Menn, Tianpei Gu, Guande He, Aryan V S, Yanke Song, Nan Liu, Desai Xie, Xun Huang, Chenfeng Xu and Diana Marculescu, MPEG-Diffusion: Mixed-Resolution Video Generation, arXiv:2611.12069.

``lora`` (their Sec. 4 "Training", eq. 2): a rank-``--rank`` LoRA on every block's qkv/out/fc1/fc2,
the ``f_up`` maps (start from ``two_track_h3.py fit-kv``) and the resolution embedding (one vector
for LR rows, one for HR rows, zero at init) train together. Each step takes one pair, encoded at
both resolutions and noised to the same sigma (independent noise per resolution); the HR region
``M`` is the pair's face mask (``--mask``), else its subject mask, else a random box. One packed
forward runs both tracks with dual-resolution attention, and the loss is H3's velocity MSE on the
LR latent outside ``M`` plus on the HR latent inside ``M``, each averaged over its own region. The
paper fully fine-tunes the DiT; a LoRA is the affordable version on one card. What this buys at
sampling time: the HR track can start earlier (``--switch``), while structure is still open, so it
can redraw faces instead of only adding texture to the LR track's.

``region`` (their Sec. 4.3 with the object label): the linear head ``s = w^T LN(h)`` on block
``--block``'s LR states at the switch sigma, trained with BCE against the face / subject mask (the
paper: LLM + Grounded SAM; here YuNet faces and YOLO people from ``build_pairs_tt.py``). With
``--train R`` the states come from the fine-tuned model, as at sampling time. Reports the held-out
mIoU of the thresholded mask, the paper's metric (0.59-0.67 on Wan).

``decoder`` (their Sec. 4.2 and H, eqs. 5, 6, 11): a learned latent upscaler ``U`` (bicubic plus
a zero-initialised residual) and the mixer ``G``, trained on real pairs: HR latent inside ``M``,
LR latent outside, target the full HR latent. Loss: latent L1 outside ``M`` (up-weighted on a ring
at the boundary), ``U``'s own latent L1, and, every step, pixel L1 + ``--lpips`` LPIPS on a
decoded crop centred on the boundary (the VAE decoder is frozen; gradients pass through it). No
DiT is loaded.

Writes only under ``--out``: ``log.jsonl`` and checkpoints (``lora{tag}.safetensors`` +
``tt_state{tag}.pt``, ``region_head_tt.pt``, ``decoder{tag}.pt``). ``two_track_h3.py render``
loads them with ``--train``, ``--region`` and ``--decoder``. Refuses to start unless
``LOT_H3_TRAIN=1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pairs_tt import list_pairs, load_pair  # noqa: E402
from gpu_guard import refuse_if_busy  # noqa: E402
from mixed_res import LatentUpscaler, RegionHead, TwoTrackMixer, TwoTrackPlan, UpMaps, region_mask  # noqa: E402
from two_track_h3 import (EXTENT, HEAD_DIM, MID_BLOCK, ResolutionEmbedding, mask_to_latent,  # noqa: E402
                          two_track_forward)


# --------------------------------------------------------------------------- masks


def random_box(height: int, width: int, rng: random.Random, share: float = 0.25) -> torch.Tensor:
    """``(1, h, w)`` bool box covering about ``share`` of the LR token grid, random aspect and place."""
    area = share * height * width
    aspect = math.exp(rng.uniform(-0.7, 0.7))
    box_h = max(1, min(height, round(math.sqrt(area / aspect))))
    box_w = max(1, min(width, round(area / box_h)))
    top, left = rng.randint(0, height - box_h), rng.randint(0, width - box_w)
    mask = torch.zeros(1, height, width, dtype=torch.bool)
    mask[0, top:top + box_h, left:left + box_w] = True
    return mask


def pick_mask(pair: dict, kind: str, rng: random.Random, box_p: float) -> tuple[torch.Tensor, str]:
    """The HR region for a training step: ``kind`` (face / subject / union), falling back to a box."""
    face, subject = pair["face"], pair["subject"]
    if rng.random() >= box_p:
        options = {"face": [face, subject], "subject": [subject, face], "union": [face | subject]}[kind]
        for mask in options:
            if mask.any() and not mask.all():
                return mask, kind
    return random_box(face.shape[1], face.shape[2], rng), "box"


def label_mask(pair: dict, kind: str) -> torch.Tensor:
    return {"face": pair["face"], "subject": pair["subject"], "union": pair["face"] | pair["subject"]}[kind]


# ---------------------------------------------------------------------------- lora


def eq2_loss(v_lr, v_hr, pair, eps_lr, eps_hr, mask, hr_weight: float):
    """Their eq. 2 in H3's ``x0 - eps`` convention: LR outside ``M`` plus HR inside ``M``."""
    inside_lr = mask_to_latent(mask, 2).to(v_lr.device)
    inside_hr = mask_to_latent(mask, 4).to(v_lr.device)
    outside = 1.0 - inside_lr
    channels = v_lr.shape[1]
    lr_loss = ((v_lr.float() - (pair["lr"] - eps_lr)) ** 2 * outside).sum() / (outside.sum() * channels).clamp(min=1)
    hr_loss = ((v_hr.float() - (pair["hr"] - eps_hr)) ** 2 * inside_hr).sum() / (inside_hr.sum() * channels).clamp(min=1)
    return lr_loss + hr_weight * hr_loss, lr_loss, hr_loss


def step_forward(dit, pair, mask, sigma: float, maps, res_embed, generator):
    eps_lr = torch.randn(pair["lr"].shape, device="cuda", generator=generator)
    eps_hr = torch.randn(pair["hr"].shape, device="cuda", generator=generator)
    x_lr = (1.0 - sigma) * pair["lr"] + sigma * eps_lr
    x_hr = (1.0 - sigma) * pair["hr"] + sigma * eps_hr
    plan = TwoTrackPlan(mask).to("cuda")
    v_lr, v_hr = two_track_forward(dit, x_lr, x_hr, plan, 1.0 - sigma, pair["text"], maps=maps,
                                   audio_generator=generator, res_embed=res_embed)
    return v_lr, v_hr, eps_lr, eps_hr


def train_lora(args) -> None:
    from h3_musubi import load_dit, make_lora, sample_sigma
    from procrustes_h3 import CHECKPOINT
    from two_track_h3 import load_maps

    train, held = list_pairs(args.pairs, args.holdout)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dit = load_dit(CHECKPOINT, device="cuda", swap=args.swap, train=True)
    network = make_lora(dit, args.rank)
    hidden_size = dit.video_patch_proj.out_features
    res_embed = ResolutionEmbedding(hidden_size).cuda()
    if args.init is not None:
        info = network.load_weights(str(args.init / "lora.safetensors"))
        if info.missing_keys:
            raise RuntimeError(f"--init LoRA is missing {len(info.missing_keys)} keys")
        saved = torch.load(args.init / "tt_state.pt", map_location="cpu", weights_only=False)
        maps = UpMaps(saved["layers"], HEAD_DIM, [EXTENT], rank=saved["rank"])
        maps.load_state_dict(saved["maps"])
        res_embed.load_state_dict(saved["res_embed"])
        maps = maps.cuda()
    elif args.maps is not None:
        maps = load_maps(args.maps)
    else:
        print("LOT_H3 kind=tt_train_warn maps=copy (run two_track_h3.py fit-kv first; copies give texture noise)",
              flush=True)
        maps = UpMaps(len(dit.blocks), HEAD_DIM, [EXTENT], rank=HEAD_DIM).cuda()
    maps.train().requires_grad_(True)
    network.requires_grad_(True)
    network.to(device="cuda", dtype=torch.bfloat16)
    lora_groups, _ = network.prepare_optimizer_params(args.lr)
    groups = lora_groups + [{"params": list(maps.parameters()), "lr": args.maps_lr},
                            {"params": list(res_embed.parameters()), "lr": args.embed_lr}]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
    except ImportError:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    params = [p for g in groups for p in g["params"]]
    print(f"LOT_H3 kind=tt_train_start pairs={len(train)} held={len(held)} trainable={sum(p.numel() for p in params)} "
          f"rank={args.rank} swap={args.swap} shift={args.shift} mask={args.mask}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def save(tag: str) -> None:
        network.save_weights(str(args.out / f"lora{tag}.safetensors"), torch.bfloat16,
                             {"lot_h3": "two_track", "rank": str(args.rank)})
        torch.save({"maps": maps.state_dict(), "rank": maps.rank, "layers": len(dit.blocks), "hidden": hidden_size,
                    "res_embed": res_embed.state_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
                   args.out / f"tt_state{tag}.pt")
        print(f"LOT_H3 kind=tt_train_saved tag={tag or 'final'} out={args.out}", flush=True)

    def evaluate(step: int) -> None:
        totals = {"lr": [], "hr": []}
        with torch.no_grad():
            for number, entry in enumerate(held):
                pair = load_pair(entry)
                mask, _kind = pick_mask(pair, args.mask, random.Random(number), 0.0)
                for sigma in (0.3, 0.6, 0.9):
                    gen = torch.Generator("cuda").manual_seed(1000 + number)
                    v_lr, v_hr, e_lr, e_hr = step_forward(dit, pair, mask, sigma, maps, res_embed, gen)
                    _loss, lr_loss, hr_loss = eq2_loss(v_lr, v_hr, pair, e_lr, e_hr, mask, 1.0)
                    totals["lr"].append(float(lr_loss))
                    totals["hr"].append(float(hr_loss))
        record = {"step": step, "eval_lr": sum(totals["lr"]) / len(totals["lr"]),
                  "eval_hr": sum(totals["hr"]) / len(totals["hr"])}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(f"LOT_H3 kind=tt_train_eval step={step} lr={record['eval_lr']:.4f} hr={record['eval_hr']:.4f}", flush=True)

    network.train()
    if not args.check:
        evaluate(0)
    started = time.perf_counter()
    generator = torch.Generator("cuda").manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        total = 0.0
        for _micro in range(args.accum):
            pair = load_pair(rng.choice(train))
            mask, kind = pick_mask(pair, args.mask, rng, args.box_p)
            sigma = float(sample_sigma(args.shift, "cpu", count=1)[0])
            v_lr, v_hr, e_lr, e_hr = step_forward(dit, pair, mask, sigma, maps, res_embed, generator)
            loss, lr_loss, hr_loss = eq2_loss(v_lr, v_hr, pair, e_lr, e_hr, mask, args.hr_weight)
            (loss / args.accum).backward()
            total += float(loss.detach()) / args.accum
        if not math.isfinite(total):
            raise RuntimeError(f"step {step}: loss {total}")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {"step": step, "loss": total, "lr_loss": float(lr_loss.detach()), "hr_loss": float(hr_loss.detach()),
                  "sigma": sigma, "mask": kind, "hr_tokens": int(mask.sum()) * 4, "grad_norm": float(grad_norm),
                  "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if args.check:
            print(f"LOT_H3 kind=tt_train_check loss={total:.4f} peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} "
                  f"s={record['seconds']} ok=1", flush=True)
            return
        if step % 25 == 0:
            log.flush()
            print(f"LOT_H3 kind=tt_train step={step} loss={total:.4f} lr={record['lr_loss']:.4f} "
                  f"hr={record['hr_loss']:.4f} sigma={sigma:.3f} mask={kind} "
                  f"s_per_step={(time.perf_counter() - started) / step:.2f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save(f"_{step}")
    save("")
    print(f"LOT_H3 kind=tt_train_done steps={args.steps} ok=1", flush=True)


# -------------------------------------------------------------------------- region


def train_region(args) -> None:
    from h3_musubi import sigma_schedule
    from two_track_h3 import load_base, load_trained

    train, held = list_pairs(args.pairs, args.holdout)
    cached = args.features is not None and args.features.is_file()
    dit = None if cached else load_base(args.swap)
    res_embed = None
    if args.train is not None and not cached:
        _network, _maps, res_embed = load_trained(dit, args.train)
    # One head for several switch steps: each pair is seen at one of them (cycled), so the same layer
    # reads the layout early (step 6 of 20) or late (step 14).
    switches = args.switches or [args.switch]
    schedule = sigma_schedule(args.steps_schedule)

    def augment(pair, rng):
        """Random off-centre crop and horizontal flip of the LR latent and its masks, so the subject's
        position stops predicting the label (the first head learned "the subject is in the middle")."""
        if rng.random() >= args.crop_aug:
            return pair
        lr, face, subject = pair["lr"], pair["face"], pair["subject"]
        tok_h, tok_w = face.shape[-2:]
        crop_h = max(4, round(tok_h * rng.uniform(0.7, 1.0)))
        crop_w = max(4, round(tok_w * rng.uniform(0.5, 0.85)))
        top, left = rng.randint(0, tok_h - crop_h), rng.randint(0, tok_w - crop_w)
        rows, cols = slice(top, top + crop_h), slice(left, left + crop_w)
        lr = lr[..., 2 * top:2 * (top + crop_h), 2 * left:2 * (left + crop_w)]
        face, subject = face[:, rows, cols], subject[:, rows, cols]
        if rng.random() < 0.5:
            lr, face, subject = lr.flip(-1), face.flip(-1), subject.flip(-1)
        return {**pair, "lr": lr.contiguous(), "face": face.contiguous(), "subject": subject.contiguous()}

    def collect(entries, offset, train_set=True):
        out = []
        rng = random.Random(args.seed + offset)
        with torch.no_grad():
            for number, entry in enumerate(entries):
                pair = load_pair(entry)
                if train_set:
                    pair = augment(pair, rng)
                sigma = float(schedule[switches[number % len(switches)] - 1])
                gen = torch.Generator("cuda").manual_seed(offset + number)
                x_lr = (1.0 - sigma) * pair["lr"] + sigma * torch.randn(pair["lr"].shape, device="cuda", generator=gen)
                capture = {"block": args.block}
                two_track_forward(dit, x_lr, None, None, 1.0 - sigma, pair["text"], capture=capture,
                                  res_embed=res_embed)
                label = label_mask(pair, args.label)
                out.append((capture["states"].bfloat16().cpu(), label.flatten().float(), label,   # bf16: activations reach ~2.6e5, past fp16
                            entry.get("subject_class") or "-", switches[number % len(switches)]))
                if number % 50 == 0:
                    print(f"LOT_H3 kind=tt_region_pair item={number}/{len(entries)} positive={int(label.sum())}",
                          flush=True)
        return out

    if cached:
        # Phase 1 already ran: reuse its features (refits take seconds, not a 30-minute collection).
        saved_features = torch.load(args.features, map_location="cpu", weights_only=False)
        pairs, held_pairs, switches = saved_features["train"], saved_features["held"], saved_features["switches"]
        print(f"LOT_H3 kind=tt_region_features_loaded path={args.features} train={len(pairs)} held={len(held_pairs)}",
              flush=True)
    else:
        pairs = collect(train[:args.items], 0)
        held_pairs = collect(held, 10_000, train_set=False)
        del dit
        torch.cuda.empty_cache()
        if args.features is not None:
            args.features.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"train": pairs, "held": held_pairs, "switches": switches, "block": args.block,
                        "label": args.label, "train_dir": str(args.train)}, args.features)
            print(f"LOT_H3 kind=tt_region_features_saved path={args.features}", flush=True)
    positive = sum(float(p[1].sum()) for p in pairs)
    count = sum(p[1].numel() for p in pairs)
    pos_weight = torch.tensor(min(20.0, (count - positive) / max(positive, 1.0)), device="cuda")
    head = RegionHead(pairs[0][0].shape[-1]).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (-1.0, None, -1)                                    # held-out mIoU, state, epoch
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(pairs)
        total = 0.0
        for states, target, _mask, _cls, _sw in pairs:
            loss = F.binary_cross_entropy_with_logits(head(states.cuda().float()), target.cuda(), pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        ious, groups = [], {}
        with torch.no_grad():
            for states, _target, mask, cls, sw in held_pairs:
                if not mask.any():
                    continue
                scores = head(states.cuda().float()).float().cpu().reshape(mask.shape)
                picked = region_mask(scores, alpha=args.alpha)
                iou = float((picked & mask).sum()) / max(1.0, float((picked | mask).sum()))
                ious.append(iou)
                for key in (f"class:{'person' if cls == 'person' else 'other'}", f"switch:{sw}"):
                    groups.setdefault(key, []).append(iou)
        miou = sum(ious) / max(1, len(ious))
        detail = " ".join(f"{k}={sum(v) / len(v):.3f}(n={len(v)})" for k, v in sorted(groups.items()))
        print(f"LOT_H3 kind=tt_region_epoch epoch={epoch} bce={total / len(pairs):.4f} held_miou={miou:.3f} {detail}",
              flush=True)
        if miou > best[0]:
            best = (miou, {k: v.detach().clone() for k, v in head.state_dict().items()}, epoch)
    # Keep the epoch with the best held-out mIoU (the training loss keeps falling after mIoU stops rising).
    miou = best[0]
    head.load_state_dict(best[1])
    print(f"LOT_H3 kind=tt_region_best epoch={best[2]} held_miou={miou:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": head.state_dict(), "best_epoch": best[2], "hidden": pairs[0][0].shape[-1], "block": args.block,
                "switch": switches[len(switches) // 2], "switches": switches, "steps": args.steps_schedule,
                "label": args.label,
                "train": str(args.train), "held_miou": miou}, args.out / "region_head_tt.pt")
    print(f"LOT_H3 kind=tt_region_done out={args.out / 'region_head_tt.pt'} held_miou={miou:.3f} ok=1", flush=True)


# ------------------------------------------------------------------------- decoder


def boundary_ring(mask_hr: torch.Tensor, radius: int) -> torch.Tensor:
    """``(1, 1, T, H, W)`` float: 1 within ``radius`` latent pixels of the region edge."""
    flat = mask_hr[:, :, 0]
    grown = F.max_pool2d(flat, 2 * radius + 1, stride=1, padding=radius)
    shrunk = -F.max_pool2d(-flat, 2 * radius + 1, stride=1, padding=radius)
    return (grown - shrunk).unsqueeze(2)


def train_decoder(args) -> None:
    from h3_musubi import load_vae
    from lot_paths import LOT_H3_VAE

    train, held = list_pairs(args.pairs, args.holdout)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    vae = load_vae(LOT_H3_VAE)
    vae.requires_grad_(False)
    vae_dtype = next(vae.parameters()).dtype
    upscaler = LatentUpscaler(width=args.up_width).cuda()
    mixer = TwoTrackMixer(width=args.mix_width).cuda()
    if args.init is not None:
        saved = torch.load(args.init, map_location="cpu", weights_only=False)
        upscaler.load_state_dict(saved["upscaler"])
        mixer.load_state_dict(saved["mixer"])
    perceptual = None
    if args.lpips > 0:
        import lpips

        perceptual = lpips.LPIPS(net="vgg").cuda().eval().requires_grad_(False)
    params = list(upscaler.parameters()) + list(mixer.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a")

    def losses(pair, mask, *, pixel: bool):
        hr, lr = pair["hr"], pair["lr"]
        mask_hr = mask_to_latent(mask, 4).cuda()
        lr_up = upscaler(lr)
        merged = mixer(hr, lr_up, mask_hr)
        ring = boundary_ring(mask_hr, args.ring)
        weight = (1.0 - mask_hr) * (1.0 + args.ring_weight * ring)
        latent = ((merged - hr).abs() * weight).sum() / (weight.sum() * hr.shape[1]).clamp(min=1)
        up = (lr_up - hr).abs().mean()
        out = {"latent": latent, "up": up}
        if pixel:
            # A crop centred on a boundary pixel: the VAE decoder sees consistent neighbours there or not.
            size = args.crop
            edge = torch.nonzero(ring[0, 0, 0] > 0)
            height, width = hr.shape[-2:]
            if len(edge):
                cy, cx = edge[rng.randrange(len(edge))].tolist()
            else:
                cy, cx = height // 2, width // 2
            top = min(max(0, cy - size // 2), max(0, height - size))
            left = min(max(0, cx - size // 2), max(0, width - size))
            window = (..., slice(top, top + size), slice(left, left + size))
            got = vae.decode(merged[window].to(vae_dtype)).float()
            with torch.no_grad():
                want = vae.decode(hr[window].to(vae_dtype)).float()
            out["pixel"] = (got - want).abs().mean()
            if perceptual is not None:
                out["lpips"] = perceptual(got[:, :, 0].clamp(-1, 1), want[:, :, 0].clamp(-1, 1)).mean()
        return out

    def total_of(parts):
        return (parts["latent"] + args.up_weight * parts["up"] + args.pixel_weight * parts.get("pixel", 0.0)
                + args.lpips * parts.get("lpips", 0.0))

    def evaluate(step):
        sums = {}
        with torch.no_grad():
            for number, entry in enumerate(held):
                pair = load_pair(entry)
                mask, _kind = pick_mask(pair, args.mask, random.Random(number), 0.0)
                for key, value in losses(pair, mask, pixel=False).items():
                    sums.setdefault(key, []).append(float(value))
        record = {"step": step, **{f"eval_{k}": sum(v) / len(v) for k, v in sums.items()}}
        log.write(json.dumps(record) + "\n")
        log.flush()
        print("LOT_H3 kind=tt_decoder_eval " + " ".join(f"{k}={v:.4f}" for k, v in record.items()), flush=True)

    evaluate(0)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        pair = load_pair(rng.choice(train))
        mask, kind = pick_mask(pair, args.mask, rng, args.box_p)
        parts = losses(pair, mask, pixel=args.pixel_weight > 0 or args.lpips > 0)
        loss = total_of(parts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        record = {"step": step, "loss": float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()},
                  "mask": kind, "seconds": round(time.perf_counter() - started, 1)}
        log.write(json.dumps(record) + "\n")
        if args.check:
            print(f"LOT_H3 kind=tt_decoder_check loss={record['loss']:.4f} "
                  f"peak_mb={torch.cuda.max_memory_allocated() / 2**20:.0f} ok=1", flush=True)
            return
        if step % 50 == 0:
            log.flush()
            print(f"LOT_H3 kind=tt_decoder step={step} " + " ".join(
                f"{k}={v:.4f}" for k, v in record.items() if isinstance(v, float) and k != "seconds"), flush=True)
        if args.eval_every and step % args.eval_every == 0:
            evaluate(step)
        if args.save_every and step % args.save_every == 0:
            save_decoder(args, upscaler, mixer, f"_{step}")
    save_decoder(args, upscaler, mixer, "")
    print(f"LOT_H3 kind=tt_decoder_done out={args.out / 'decoder.pt'} ok=1", flush=True)


def save_decoder(args, upscaler, mixer, tag: str) -> None:
    torch.save({"upscaler": upscaler.state_dict(), "mixer": mixer.state_dict(), "up_width": args.up_width,
                "mix_width": args.mix_width}, args.out / f"decoder{tag}.pt")


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("phase", choices=("lora", "region", "decoder"))
    parser.add_argument("--pairs", type=Path, action="append", required=True, help="build_pairs_tt.py dir (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap", default="auto", help="DiT blocks streamed from CPU; auto (by free VRAM): 0 at 64 GB+, 8 at 28+, 16 at 21+, else 24")
    parser.add_argument("--mask", choices=("face", "subject", "union"), default="face",
                        help="which pair mask is the HR region (falls back to the other, then to a box)")
    parser.add_argument("--box-p", type=float, default=0.2, help="share of steps with a random box region instead")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--check", action="store_true", help="one step, print peak memory, write no checkpoint")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--init", type=Path, default=None,
                        help="lora: resume from a train_tt lora dir; decoder: from a decoder.pt")
    # lora
    parser.add_argument("--maps", type=Path, default=None, help="lora: up_maps_tt.pt from two_track_h3.py fit-kv")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--maps-lr", type=float, default=1e-4)
    parser.add_argument("--embed-lr", type=float, default=1e-3)
    parser.add_argument("--shift", default="3.0",
                        help="training sigma density (h3_musubi.sample_sigma): a float shift, sigmoid, lognorm:S")
    parser.add_argument("--hr-weight", type=float, default=1.0)
    parser.add_argument("--accum", type=int, default=1)
    # region
    parser.add_argument("--train", type=Path, default=None, help="region: states from this train_tt lora dir")
    parser.add_argument("--label", choices=("face", "subject", "union"), default="union")
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--block", type=int, default=MID_BLOCK)
    parser.add_argument("--steps-schedule", type=int, default=20, help="region: sampler steps the switch refers to")
    parser.add_argument("--switch", type=int, default=12, help="region: the sampler's switch step")
    parser.add_argument("--features", type=Path, default=None,
                        help="region: phase-1 features file; written after collection, reused when it exists")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="region: Adam weight decay on the head")
    parser.add_argument("--crop-aug", type=float, default=0.0,
                        help="region: share of training pairs given a random off-centre crop + flip (0.7 recommended)")
    parser.add_argument("--switches", type=int, nargs="+", default=None,
                        help="region: train one head for several switch steps (cycled over the pairs)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    # decoder
    parser.add_argument("--up-width", type=int, default=128)
    parser.add_argument("--mix-width", type=int, default=64)
    parser.add_argument("--up-weight", type=float, default=1.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    parser.add_argument("--lpips", type=float, default=0.5, help="LPIPS weight (0 = off; downloads VGG once)")
    parser.add_argument("--crop", default="auto",
                        help="decoder: decoded crop side in latent pixels (x16 px). 16 peaks at 12 GB and 24 runs out "
                             "on 32 GB; auto: 32 on 64 GB+, else 16")
    parser.add_argument("--ring", type=int, default=2, help="decoder: boundary ring radius, latent pixels")
    parser.add_argument("--ring-weight", type=float, default=4.0)
    args = parser.parse_args()
    from gpu_guard import auto_swap

    args.swap = auto_swap(args.swap)
    if args.crop == "auto":
        from gpu_guard import card_gb

        args.crop = 32 if card_gb() >= 64 else 16
    args.crop = int(args.crop)
    if args.phase == "region" and args.lr == 1e-4:
        args.lr = 1e-3
    if args.phase == "decoder" and args.lr == 1e-4:
        args.lr = 2e-4
    if args.phase == "decoder" and args.steps == 2000:
        args.steps = 4000
    try:
        args.shift = float(args.shift)
    except ValueError:
        pass
    if os.environ.get("LOT_H3_TRAIN") != "1":
        raise SystemExit("set LOT_H3_TRAIN=1 to train (this script writes checkpoints under --out)")
    refuse_if_busy("train_tt.py")
    {"lora": train_lora, "region": train_region, "decoder": train_decoder}[args.phase](args)


if __name__ == "__main__":
    main()
