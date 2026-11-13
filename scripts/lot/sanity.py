#!/usr/bin/env python3
"""CPU sanity checks for the LoT compute cut. No checkpoint, no GPU, no H3 weights.

The saving that counts is the attention sequence. A mixed layout of legal H3
extents must attend over fewer tokens than the dense lattice, and the velocity
written back must still cover every uniform cell.

    python3 scripts/lot/sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter  # noqa: E402
from h3 import H3_EXTENTS, make_h3_adapter  # noqa: E402
from layout import TokenRect, dense_layout, layout_from_rects  # noqa: E402
from train_synth import LotBackbone  # noqa: E402


def block_macs(seq: int, hidden: int, layers: int, batch: int = 1) -> int:
    """Multiply-adds inside ``LotBackbone`` attention blocks.

    Per layer, per the modules that class builds: fused QKV, attention output,
    the two sequence-by-sequence products, and the 4x MLP. Position and sigma
    linears are outside this count.
    """
    if min(seq, hidden, layers, batch) < 1:
        raise ValueError("sequence, hidden, layers, and batch must be positive")
    qkv_out = batch * seq * hidden * hidden * 4
    attention = batch * seq * seq * hidden * 2
    mlp = batch * seq * hidden * hidden * 8
    return layers * (qkv_out + attention + mlp)


def mixed_layout() -> "object":
    """Six rectangles on an 8x8 lattice. Sides are 4 and 2, so every extent is in ``H3_EXTENTS``."""
    rects = [
        TokenRect(0, 0, 0, 1, 4, 4),
        TokenRect(0, 0, 4, 1, 4, 4),
        TokenRect(0, 4, 0, 1, 2, 4),
        TokenRect(0, 4, 4, 1, 2, 4),
        TokenRect(0, 6, 0, 1, 2, 4),
        TokenRect(0, 6, 4, 1, 2, 4),
    ]
    return layout_from_rects(1, 8, 8, rects)


def check_short_sequence_full_canvas() -> dict:
    layout = mixed_layout()
    dense = dense_layout(1, 8, 8)
    if layout.count >= dense.count:
        raise AssertionError(f"mixed tokens {layout.count} are not fewer than dense {dense.count}")
    if layout.dense_count != 64 or dense.count != 64:
        raise AssertionError("canvas token count changed")
    covered = sum(rect.et * rect.eh * rect.ew for rect in layout.rects)
    if covered != layout.dense_count:
        raise AssertionError(f"rectangles cover {covered}, canvas is {layout.dense_count}")

    hidden = 32
    layers = 2
    batch = 2
    adapter = LotVisualAdapter(8, hidden, list(H3_EXTENTS), shape_hidden=16)
    backbone = LotBackbone(hidden, layers, 4)
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def record(states, centers, _sigma):
        seen.append((tuple(states.shape), tuple(centers.shape)))
        return backbone(states, centers, _sigma)

    tokens = torch.randn(batch, 1, 8, 8, 8)
    velocity = adapter(tokens, 0.4, layout, record)
    if velocity.shape != tokens.shape:
        raise AssertionError(f"velocity {tuple(velocity.shape)} is not the full canvas {tuple(tokens.shape)}")
    if seen != [((batch, layout.count, hidden), (layout.count, 3))]:
        raise AssertionError(f"backbone saw {seen}, expected one call of length {layout.count}")
    if not torch.isfinite(velocity).all():
        raise AssertionError("velocity is not finite")

    dense_seen: list[int] = []

    def record_dense(states, _centers, _sigma):
        dense_seen.append(states.shape[1])
        return states

    adapter(tokens[:1], 0.4, dense, record_dense)
    if dense_seen != [dense.count]:
        raise AssertionError(f"dense backbone length {dense_seen}")

    macs = block_macs(layout.count, hidden, layers, batch)
    dense_macs = block_macs(dense.count, hidden, layers, batch)
    if macs >= dense_macs / 2:
        raise AssertionError(f"block MACs {macs} are not under half of dense {dense_macs}")
    return {
        "tokens": layout.count,
        "dense": dense.count,
        "macs": macs,
        "dense_macs": dense_macs,
    }


def check_h3_bank_rejects_coarse_square() -> None:
    """An 8x8 square is an image-quadtree cell. It is not an H3 video extent."""
    adapter = make_h3_adapter(hidden_size=16)
    layout = layout_from_rects(1, 8, 8, [TokenRect(0, 0, 0, 1, 8, 8)])
    tokens = torch.zeros(1, 1, 8, 8, adapter.token_dim)
    try:
        adapter(tokens, 0.2, layout, lambda states, _centers, _sigma: states)
    except KeyError as exc:
        if "(1, 8, 8)" not in str(exc):
            raise AssertionError(f"unexpected rejection: {exc}") from exc
        return
    raise AssertionError("H3 bank accepted an 8x8 token")


def run_checks() -> dict:
    numbers = check_short_sequence_full_canvas()
    check_h3_bank_rejects_coarse_square()
    return numbers


def main() -> None:
    numbers = run_checks()
    print(
        "SANITY ok 2 "
        f"tokens {numbers['tokens']}/{numbers['dense']} "
        f"macs {numbers['macs']}/{numbers['dense_macs']} "
        "writes=0"
    )


if __name__ == "__main__":
    main()
