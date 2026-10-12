#!/usr/bin/env python3
"""Phase 4: Procrustes-fit the H3 extent bank, then optionally one frozen smoke.

    python3 scripts/lot/procrustes_h3.py --pairs DIR [--smoke] [--out FILE]

``DIR`` holds one ``{et}x{eh}x{ew}.pt`` per extent, each a dict with
``dense (N, 96*et*eh*ew)`` and ``reference (N, 96)`` in ``gather_extent`` row
order. The user supplies these. This script never builds pairs, and exits if
the directory is missing. There is no ``1x1x1.pt``: a unit extent is the
identity and must stay so.

The fit reads only ``video_patch_proj`` and ``final_layer.video_out`` from the
checkpoint, on CPU. ``--smoke`` loads the int8 DiT (32 blocks streamed) and runs one
frozen forward on the 384×640, ``latent_t=7`` canvas in y-space. It asserts
shape, finite values, and the packed length. It does not judge picture
quality: a frozen base sees fitted scales it was never trained on.

Nothing is written unless ``--out`` is given. Does not call empty_cache.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lot_paths import LOT_H3_CHECKPOINT, LOT_H3_STILLS_CACHE, LOT_H3_STILLS_CAPTIONS, LOT_H3_TEXT_ENCODER, LOT_H3_VAE  # noqa: E402,F401

from adapter import LotVisualAdapter, _key  # noqa: E402
from h3 import H3_EXTENTS, H3_TOKEN_DIM, clip_layout, make_h3_adapter  # noqa: E402

CHECKPOINT = LOT_H3_CHECKPOINT
UNIT = (1, 1, 1)
ORTHO_TOL = 1e-4

# Phase-5 canvas: 384×640 -> latent (24, 40), tokens 12×20, 7 latent frames.
SMOKE_T, SMOKE_H, SMOKE_W, SMOKE_TEXT = 7, 24, 40, 128
SMOKE_SIGMA = 0.4
SMOKE_SWAP = 32


def parse_extent(stem: str) -> tuple[int, int, int]:
    parts = stem.split("x")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"pair file {stem!r} is not named {{et}}x{{eh}}x{{ew}}")
    return tuple(int(part) for part in parts)


def load_pairs(directory: Path) -> dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Read and validate every pair file. Exits if the directory is missing or empty."""
    if not directory.is_dir():
        raise SystemExit(f"pair directory {directory} does not exist. Phase 4 does not invent pairs.")
    files = sorted(directory.glob("*.pt"))
    if not files:
        raise SystemExit(f"pair directory {directory} has no *.pt files.")
    pairs = {}
    for path in files:
        extent = parse_extent(path.stem)
        if extent == UNIT:
            raise ValueError("1x1x1.pt: a unit extent is the identity lift and is not fit")
        if extent not in H3_EXTENTS:
            raise ValueError(f"{path.name}: extent {extent} is not an H3 extent")
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(blob, dict) or "dense" not in blob or "reference" not in blob:
            raise ValueError(f"{path.name}: expected a dict with 'dense' and 'reference'")
        pairs[extent] = check_pair(extent, blob["dense"], blob["reference"], path.name)
    return pairs


def check_pair(extent, dense: torch.Tensor, reference: torch.Tensor, name: str = "pair"):
    et, eh, ew = extent
    dense_dim = H3_TOKEN_DIM * et * eh * ew
    if dense.ndim != 2 or dense.shape[1] != dense_dim:
        raise ValueError(f"{name}: dense {tuple(dense.shape)} != (N, {dense_dim})")
    if reference.ndim != 2 or reference.shape[1] != H3_TOKEN_DIM:
        raise ValueError(f"{name}: reference {tuple(reference.shape)} != (N, {H3_TOKEN_DIM})")
    if dense.shape[0] != reference.shape[0]:
        raise ValueError(f"{name}: {dense.shape[0]} dense rows vs {reference.shape[0]} reference rows")
    if dense.shape[0] < H3_TOKEN_DIM:
        raise ValueError(f"{name}: {dense.shape[0]} rows cannot fix a rank-{H3_TOKEN_DIM} basis")
    if not (torch.isfinite(dense).all() and torch.isfinite(reference).all()):
        raise ValueError(f"{name}: non-finite values")
    return dense.float(), reference.float()


def pretrained_maps(checkpoint: Path) -> tuple[torch.Tensor, ...]:
    """``(W_in, W_out, b_in, b_out)`` read lazily from the checkpoint, fp32, CPU."""
    from safetensors import safe_open

    with safe_open(str(checkpoint), "pt") as handle:
        return tuple(
            handle.get_tensor(key).float()
            for key in (
                "video_patch_proj.weight",
                "final_layer.video_out.weight",
                "video_patch_proj.bias",
                "final_layer.video_out.bias",
            )
        )


def fit_bank(adapter: LotVisualAdapter, pairs) -> list[dict]:
    """``fit_extent`` per pair, then the plan's three checks. Raises on any failure."""
    if int(adapter.pretrained_ready) != 1:
        raise RuntimeError("call init_from_pretrained first; fit_extent rebuilds heads from it")
    unit = _key(UNIT)
    before = {
        name: tensor.detach().clone()
        for name, tensor in (
            *adapter.in_proj[unit].state_dict().items(),
            *(("out_" + k, v) for k, v in adapter.out_proj[unit].state_dict().items()),
        )
    }
    reports = []
    for extent in sorted(pairs):
        dense, reference = pairs[extent]
        adapter.fit_extent(extent, dense, reference)
        basis = adapter.bank.basis(extent).double()
        eye = torch.eye(basis.shape[1], dtype=torch.float64)
        ortho_err = float((basis.T @ basis - eye).abs().max())
        scale = float(adapter.bank.scale(extent))
        if ortho_err > ORTHO_TOL:
            raise ValueError(f"{extent}: A^T A differs from I by {ortho_err:.3g}")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{extent}: scale {scale} is not finite and positive")
        reports.append({"extent": extent, "rows": dense.shape[0], "scale": scale, "ortho_err": ortho_err})
    after = {
        name: tensor
        for name, tensor in (
            *adapter.in_proj[unit].state_dict().items(),
            *(("out_" + k, v) for k, v in adapter.out_proj[unit].state_dict().items()),
        )
    }
    for name, tensor in before.items():
        if not torch.equal(tensor, after[name]):
            raise ValueError(f"1x1x1 head {name} changed during the fit")
    if float(adapter.bank.scale(UNIT)) != 1.0:
        raise ValueError("1x1x1 scale moved off 1")
    return reports


def smoke(adapter: LotVisualAdapter) -> None:
    """One frozen y-space forward on the mixed phase-5 canvas."""
    from flow import sample_noisy
    from gpu_guard import refuse_if_busy
    from h3_splice import LotSplice
    from h3_musubi import audio_frames, load_dit, predict

    refuse_if_busy("procrustes_h3.py")
    model = load_dit(CHECKPOINT, swap=SMOKE_SWAP)

    layout = clip_layout(SMOKE_T, SMOKE_H // 2, SMOKE_W // 2)
    adapter.cuda()
    splice = LotSplice(adapter, layout, y_space=True)
    # A random stand-in clean latent. It is a smoke input, not a pair.
    x0 = torch.randn(1, 24, SMOKE_T, SMOKE_H, SMOKE_W, device="cuda", dtype=torch.float32)
    y0 = splice.to_y(x0)
    y_t, _noise = sample_noisy(y0, SMOKE_SIGMA)
    text = torch.randn(1, SMOKE_TEXT, model.config.text_dim, device="cuda", dtype=torch.bfloat16)
    n_audio = audio_frames(SMOKE_T)
    seen = []
    hook = model.blocks[0].register_forward_hook(lambda _m, inputs, _o: seen.append(tuple(inputs[0].shape[1:])))
    try:
        with torch.inference_mode():
            out = predict(model, y_t, 1.0 - SMOKE_SIGMA, text, lot=splice)
    finally:
        hook.remove()
    expected_seq = SMOKE_TEXT + n_audio * 2 + layout.count
    # H3's head is x0 - eps, so one step to sigma 0 is y_t + sigma * out.
    y0_hat = y_t + SMOKE_SIGMA * out.float()
    x0_hat = splice.to_x(y0_hat)
    ok = (
        tuple(out.shape) == tuple(y_t.shape)
        and bool(torch.isfinite(out).all())
        and bool(torch.isfinite(x0_hat).all())
        and bool(seen)
        and seen[0][0] == expected_seq
    )
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        "LOT_H3 kind=smoke task=t2va "
        f"latent_t={SMOKE_T} canvas=384x640 lot_tokens={expected_seq} "
        f"seq={seen[0] if seen else None} out={tuple(out.shape)} "
        f"out_rms={float(out.float().pow(2).mean().sqrt()):.4g} "
        f"peak_mb={peak_mb:.1f} dtype=bf16 ok={int(ok)}",
        flush=True,
    )
    if not ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pairs", type=Path, required=True, help="directory of {et}x{eh}x{ew}.pt pair files")
    parser.add_argument("--smoke", action="store_true", help="also run one frozen forward on the GPU")
    parser.add_argument("--out", type=Path, default=None, help="save the fitted adapter state here")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing checkpoint {CHECKPOINT}")
    adapter = make_h3_adapter()
    adapter.init_from_pretrained(*pretrained_maps(CHECKPOINT))
    for report in fit_bank(adapter, pairs):
        et, eh, ew = report["extent"]
        print(
            f"LOT_H3 kind=procrustes extent={et}x{eh}x{ew} rows={report['rows']} "
            f"scale={report['scale']:.6g} ortho_err={report['ortho_err']:.3g} ok=1",
            flush=True,
        )
    if args.smoke:
        smoke(adapter)
    if args.out is not None:
        torch.save(adapter.state_dict(), args.out)
        print(f"LOT_H3 kind=saved path={args.out}", flush=True)


if __name__ == "__main__":
    main()
