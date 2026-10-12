#!/usr/bin/env python3
"""Sample a Level-of-Token checkpoint.

The checkpoint is the synthetic adapter plus the small backbone from
``train_synth.py``. Integration follows the training path
``y_t = (1 - t) y0 + t eps``, so ``dy/dt = eps - y0``. The sampler starts at
noise (``t = 1``) and takes Euler steps down to ``t = 0``, then removes the
extent scale.

This does not load an H3 checkpoint and does not import Separable Causal
Diffusion.

    python3 scripts/lot/infer.py --ckpt scripts/lot/runs/day/last.pt --steps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter
from flow import euler_step
from train_synth import LotBackbone, tiled


def sigma_grid(steps: int) -> torch.Tensor:
    """Inclusive endpoints from 1 down to 0. The model is queried at t > 0."""
    if steps < 1:
        raise ValueError("steps must be positive")
    return torch.linspace(1.0, 0.0, steps + 1)


@torch.no_grad()
def integrate(predict, y: torch.Tensor, layout, times: torch.Tensor) -> torch.Tensor:
    """Euler integration of ``dy/dt = predict(y, t)``. ``times`` decreases."""
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a 1-D schedule with a start and an end")
    state = y
    for index in range(times.shape[0] - 1):
        t = float(times[index])
        t_next = float(times[index + 1])
        velocity = predict(state, t, layout)
        if velocity.shape != state.shape:
            raise ValueError(f"velocity shape {tuple(velocity.shape)} != state {tuple(state.shape)}")
        state = euler_step(state, velocity, t, t_next)
    return state


def _extent_from_key(key: str) -> tuple[int, int, int]:
    text = key.split(".", 1)[1].rsplit(".", 1)[0]
    et, eh, ew = text.split("x")
    return int(et), int(eh), int(ew)


def architecture_from_state(adapter_state, backbone_state) -> dict:
    """Recover the trainer shapes. Head count is not stored in the checkpoint."""
    hidden, token_dim = adapter_state["pretrained_in"].shape
    extents = []
    seen = set()
    for key in adapter_state:
        if key.startswith("in_proj.") and key.endswith(".weight"):
            extent = _extent_from_key(key)
            if extent not in seen:
                seen.add(extent)
                extents.append(extent)
    if not extents:
        raise ValueError("checkpoint has no extent heads")
    feat_dim = int(adapter_state["shape_mlp.0.weight"].shape[1])
    shape_hidden = int(adapter_state["shape_mlp.0.weight"].shape[0])
    layers = len({
        key.split(".")[1]
        for key in backbone_state
        if key.startswith("blocks.") and key.endswith("norm1.weight")
    })
    return {
        "token_dim": int(token_dim),
        "hidden": int(hidden),
        "extents": extents,
        "include_time": feat_dim == 5,
        "shape_hidden": shape_hidden,
        "layers": layers,
    }


def load_synth(path: Path, device: torch.device, heads: int = 4):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    spec = architecture_from_state(blob["adapter"], blob["backbone"])
    if spec["hidden"] % heads != 0:
        raise ValueError(f"hidden {spec['hidden']} is not divisible by heads {heads}")
    adapter = LotVisualAdapter(
        spec["token_dim"],
        spec["hidden"],
        spec["extents"],
        include_time=spec["include_time"],
        shape_hidden=spec["shape_hidden"],
    )
    backbone = LotBackbone(spec["hidden"], spec["layers"], heads)
    adapter.load_state_dict(blob["adapter"])
    backbone.load_state_dict(blob["backbone"])
    adapter.to(device).eval()
    backbone.to(device).eval()
    return adapter, backbone, spec, blob.get("step")


def choose_layout(name: str, grid: int):
    table = {
        "dense": (1, 1),
        "2": (2, 2),
        "4": (4, 4),
        "2x4": (2, 4),
        "4x2": (4, 2),
    }
    if name not in table:
        raise ValueError(f"layout {name} is not one of {sorted(table)}")
    eh, ew = table[name]
    if grid % eh or grid % ew:
        raise ValueError(f"grid {grid} is not divisible by extent {(eh, ew)}")
    return tiled(grid, grid, eh, ew)


@torch.no_grad()
def sample(
    adapter,
    backbone,
    layout,
    batch: int,
    steps: int,
    generator: torch.Generator | None = None,
    noise: torch.Tensor | None = None,
):
    device = next(adapter.parameters()).device
    if noise is None:
        noise = torch.randn(
            batch,
            layout.time,
            layout.height,
            layout.width,
            adapter.token_dim,
            device=device,
            generator=generator,
        )
    times = sigma_grid(steps).to(device)

    def predict(state, t, current):
        return adapter(state, t, current, backbone)

    latent = integrate(predict, noise, layout, times)
    return adapter.bank.unscale(latent, layout)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left = a.detach().float().reshape(-1)
    right = b.detach().float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0))


def save_layout_comparison(
    samples: dict[str, torch.Tensor],
    path: Path,
    train_step: int,
    seed: int,
    steps: int,
) -> list[tuple[str, str, float]]:
    """Channel 0 of each layout on one color scale. Scores are full-latent cosines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(samples)
    fields = [samples[name][0, 0, :, :, 0].detach().float().cpu() for name in names]
    stacked = torch.stack(fields)
    limit = float(stacked.abs().max().clamp(min=1e-6))
    reference = samples[names[0]]
    scores = [(names[0], name, cosine(reference, samples[name])) for name in names]

    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.8), dpi=140)
    image = None
    for ax, name, field, (_left, _right, score) in zip(axes, names, fields, scores):
        image = ax.imshow(field, cmap="magma", vmin=-limit, vmax=limit)
        ax.set_title(f"{name}\ncosine {score:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes.tolist(), fraction=0.02, pad=0.02)
    fig.suptitle(f"Same noise seed {seed}, {steps} Euler steps, checkpoint {train_step}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return scores


def run_layout_comparison(adapter, backbone, spec, train_step, args) -> None:
    """One noise tensor, five layouts. Prints cosine similarity against the dense layout."""
    device = next(adapter.parameters()).device
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(
        args.batch,
        1,
        args.grid,
        args.grid,
        adapter.token_dim,
        device=device,
        generator=generator,
    )
    names = ("dense", "2", "4", "2x4", "4x2")
    samples = {}
    for name in names:
        layout = choose_layout(name, args.grid)
        missing = sorted({rect.extent for rect in layout.rects} - set(map(tuple, spec["extents"])))
        if missing:
            raise SystemExit(f"layout {name} needs extents {missing}")
        latent = sample(adapter, backbone, layout, args.batch, args.steps, noise=noise)
        if not torch.isfinite(latent).all():
            raise SystemExit(f"layout {name} produced non-finite values")
        samples[f"{name} {layout.count}/{layout.dense_count}"] = latent.detach().cpu()
        print(
            f"layout {name} tokens {layout.count}/{layout.dense_count} "
            f"mean {float(latent.mean()):.4f} std {float(latent.std()):.4f}",
            flush=True,
        )
    preview = args.out.with_suffix(".png")
    scores = save_layout_comparison(samples, preview, int(train_step or 0), args.seed, args.steps)
    torch.save({"samples": samples, "scores": scores, "seed": args.seed, "train_step": train_step}, args.out)
    for left, right, score in scores:
        print(f"cosine {left} vs {right} {score:.4f}", flush=True)
    print(f"LOT_COMPARE seed={args.seed} preview={preview}", flush=True)


def save_preview(latent: torch.Tensor, path: Path) -> None:
    """Write channel 0 of the first sample. Values are scaled into 0..255."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field = latent[0, 0, :, :, 0].detach().float().cpu()
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=140)
    image = ax.imshow(field, cmap="magma")
    ax.set_title("LoT sample, channel 0")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sample a synthetic LoT checkpoint")
    parser.add_argument("--ckpt", type=Path, default=Path("scripts/lot/runs/day/last.pt"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--layout", default="4", choices=("dense", "2", "4", "2x4", "4x2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("scripts/lot/runs/day/sample.pt"))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not args.ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    device = torch.device("cuda")
    adapter, backbone, spec, step = load_synth(args.ckpt, device, heads=args.heads)
    if spec["token_dim"] != adapter.token_dim:
        raise SystemExit("token dim mismatch")
    if args.compare:
        run_layout_comparison(adapter, backbone, spec, step, args)
        return
    layout = choose_layout(args.layout, args.grid)
    missing = sorted({rect.extent for rect in layout.rects} - set(map(tuple, spec["extents"])))
    if missing:
        raise SystemExit(f"layout extents {missing} are not in the checkpoint")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    latent = sample(adapter, backbone, layout, args.batch, args.steps, generator)
    if not torch.isfinite(latent).all():
        raise SystemExit("sample has non-finite values")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latent": latent.detach().cpu(),
            "layout": args.layout,
            "grid": args.grid,
            "steps": args.steps,
            "seed": args.seed,
            "train_step": step,
        },
        args.out,
    )
    preview = args.out.with_suffix(".png")
    save_preview(latent, preview)
    print(
        f"LOT_INFER train_step={step} ode_steps={args.steps} layout={args.layout} "
        f"tokens={layout.count}/{layout.dense_count} "
        f"mean={float(latent.mean()):.4f} std={float(latent.std()):.4f} "
        f"out={args.out} preview={preview}",
        flush=True,
    )


if __name__ == "__main__":
    main()
