#!/usr/bin/env python3
"""Synthetic flow-matching for the Level-of-Token adapter.

One GPU, random latents, no H3 checkpoint, no Separable Causal Diffusion
imports. A small transformer is the backbone, so the extent heads and the
shape MLP are what get trained.

Stops at --steps or --minutes, whichever comes first. The last stdout line is:

    LOT_RESULT ok=1 steps=N start=F end=F nan=0 peak_mb=F out=PATH

    python3 scripts/lot/train_synth.py --steps 200 --out scripts/lot/runs/probe
    python3 scripts/lot/train_synth.py --minutes 480 --out scripts/lot/runs/day
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter
from flow import clean_from_velocity, lot_clean_loss, sample_noisy
from layout import TokenRect, layout_from_detail, layout_from_rects, prepare_detail


EXTENTS = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 2, 4), (1, 4, 2)]


class LotBackbone(nn.Module):
    """A few attention blocks over the LoT sequence. Not an H3 weight."""

    def __init__(self, hidden: int, layers: int, heads: int):
        super().__init__()
        self.pos = nn.Linear(3, hidden)
        self.sigma = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden),
                "attn": nn.MultiheadAttention(hidden, heads, batch_first=True),
                "norm2": nn.LayerNorm(hidden),
                "mlp": nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Linear(hidden * 4, hidden),
                ),
            }))

    def forward(self, hidden, centers, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(float(sigma), device=hidden.device, dtype=hidden.dtype)
        else:
            sigma = sigma.to(device=hidden.device, dtype=hidden.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(hidden.shape[0])
        pos = self.pos(centers.to(dtype=hidden.dtype))
        state = hidden + pos.unsqueeze(0) + self.sigma(sigma.reshape(-1, 1)).unsqueeze(1)
        for block in self.blocks:
            normed = block["norm1"](state)
            attended, _ = block["attn"](normed, normed, normed, need_weights=False)
            state = state + attended
            state = state + block["mlp"](block["norm2"](state))
        return state


def tiled(height: int, width: int, eh: int, ew: int):
    rects = [
        TokenRect(0, u, v, 1, eh, ew)
        for u in range(0, height, eh)
        for v in range(0, width, ew)
    ]
    return layout_from_rects(1, height, width, rects)


def layout_pool(height: int, width: int, count: int):
    pool = [
        tiled(height, width, 1, 1),
        tiled(height, width, 2, 2),
        tiled(height, width, 4, 4),
        tiled(height, width, 2, 4),
        tiled(height, width, 4, 2),
    ]
    generator = torch.Generator().manual_seed(0)
    while len(pool) < count:
        detail = torch.rand(height, width, generator=generator)
        corner = height // 4
        detail[-corner:, -corner:] = 1
        if len(pool) % 2 == 0:
            detail[:corner, :corner] = 0
        pool.append(layout_from_detail(prepare_detail(detail), {4: 0.35, 2: 0.7}, root=4))
    return pool


def make_clean(batch: int, height: int, width: int, token_dim: int, device: torch.device):
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    smooth = torch.exp(-(grid_y.square() + grid_x.square()))
    corner = ((grid_y > 0.35) & (grid_x > 0.35)).to(dtype=torch.float32)
    base = torch.randn(batch, 1, 1, 1, token_dim, device=device)
    ripple = torch.randn(batch, 1, 1, 1, token_dim, device=device)
    spike = torch.randn(batch, 1, 1, 1, token_dim, device=device)
    field = 0.4 * base + ripple * smooth[None, None, :, :, None] + 1.5 * spike * corner[None, None, :, :, None]
    return field


def result_line(ok: int, steps: int, start: float, end: float, nan: int, peak_mb: float, out: Path) -> str:
    return (
        f"LOT_RESULT ok={ok} steps={steps} start={start:.6f} end={end:.6f} "
        f"nan={nan} peak_mb={peak_mb:.1f} out={out}"
    )


@torch.no_grad()
def fixed_eval(adapter, backbone, sets) -> float:
    """Mean loss on frozen batches. Comparable across steps; the train loss is not."""
    adapter.eval()
    backbone.eval()
    total = 0.0
    for layout, y0, y_t, sigma in sets:
        velocity = adapter(y_t, sigma, layout, backbone)
        total += float(lot_clean_loss(clean_from_velocity(y_t, velocity, sigma), y0, sigma))
    adapter.train()
    backbone.train()
    return total / len(sets)


def main():
    parser = argparse.ArgumentParser(description="Synthetic LoT flow-matching on one GPU")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--minutes", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("scripts/lot/runs/probe"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Write one overwritten last.pt every N steps. 0 writes no checkpoint.",
    )
    args = parser.parse_args()
    if args.steps is None and args.minutes is None:
        args.steps = 200
    if args.hidden % args.heads != 0:
        raise SystemExit("hidden must be divisible by heads")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    pool = layout_pool(args.grid, args.grid, 16)
    adapter = LotVisualAdapter(args.token_dim, args.hidden, EXTENTS, shape_hidden=128).to(device)
    backbone = LotBackbone(args.hidden, args.layers, args.heads).to(device)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(backbone.parameters()),
        lr=args.lr,
    )

    deadline = None if args.minutes is None else time.perf_counter() + args.minutes * 60.0
    losses = []
    evals = []
    saw_nan = False
    step = 0
    started = time.perf_counter()
    eval_sets = []
    for layout in (pool[0], pool[2]):
        clean = make_clean(args.batch, args.grid, args.grid, args.token_dim, device)
        sigma = torch.full((args.batch,), 0.5, device=device)
        y0 = adapter.bank.scale_clean(clean, layout)
        noise = torch.randn_like(y0)
        y_t, _noise = sample_noisy(y0, sigma, noise)
        eval_sets.append((layout, y0, y_t, sigma))
    with log_path.open("a") as log_file:
        while True:
            if args.steps is not None and step >= args.steps:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            layout = pool[step % len(pool)]
            clean = make_clean(args.batch, args.grid, args.grid, args.token_dim, device)
            y0 = adapter.bank.scale_clean(clean, layout)
            sigma = 0.05 + 0.95 * torch.rand(args.batch, device=device)
            y_t, _noise = sample_noisy(y0, sigma)
            velocity = adapter(y_t, sigma, layout, backbone)
            loss = lot_clean_loss(clean_from_velocity(y_t, velocity, sigma), y0, sigma)
            if not torch.isfinite(loss):
                saw_nan = True
                record = {"step": step, "loss": None, "tokens": layout.count, "nan": True}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(f"step {step} non-finite loss", flush=True)
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(backbone.parameters()),
                1.0,
            )
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            if step % args.log_every == 0:
                held = fixed_eval(adapter, backbone, eval_sets)
                evals.append(held)
                record = {
                    "step": step,
                    "loss": value,
                    "eval": held,
                    "tokens": layout.count,
                    "dense": layout.dense_count,
                    "seconds": round(time.perf_counter() - started, 1),
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(
                    f"step {step} loss {value:.4f} eval {held:.4f} tokens {layout.count}/{layout.dense_count}",
                    flush=True,
                )
            if args.save_every and step > 0 and step % args.save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "loss": value,
                        "adapter": adapter.state_dict(),
                        "backbone": backbone.state_dict(),
                    },
                    out / "last.pt",
                )
            step += 1

    if args.save_every > 0 and losses and not saw_nan:
        torch.save(
            {
                "step": step,
                "loss": losses[-1],
                "adapter": adapter.state_dict(),
                "backbone": backbone.state_dict(),
            },
            out / "last.pt",
        )
    if not saw_nan:
        evals.append(fixed_eval(adapter, backbone, eval_sets))
    start = evals[0] if evals else float("nan")
    end = evals[-1] if evals else float("nan")
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    ok = int(step > 0 and not saw_nan and len(evals) >= 2 and end < start)
    print(result_line(ok, step, start, end, int(saw_nan), peak_mb, out), flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
