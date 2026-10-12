#!/usr/bin/env python3
"""CPU checks for the Level-of-Token adapter. No weights and no GPU.

    python3 scripts/lot/test_lot.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import LotVisualAdapter  # noqa: E402
from flow import (  # noqa: E402
    _asymflow,
    apply_extent_scales,
    asymmetric_target,
    clean_from_velocity,
    compress_patches,
    euler_step,
    fit_procrustes,
    gather_extent,
    lot_clean_loss,
    lot_h3_clean_loss,
    mean_basis,
    recover_dense_velocity,
    sample_noisy,
    scatter_extent,
)
from infer import integrate, sigma_grid  # noqa: E402
from h3_positions import packed_positions, sample_axis, spatial_axis, video_positions  # noqa: E402
from h3 import (  # noqa: E402
    H3_EXTENTS,
    clip_layout,
    H3_HIDDEN,
    H3_PATCH,
    H3_TOKEN_DIM,
    gate_frame_layout,
    make_h3_adapter,
    over_wan_frame_budget,
    patchify,
    unpatchify,
)
from h3_splice import LotSplice  # noqa: E402
from layout import (  # noqa: E402
    TokenRect,
    blur_radius,
    dense_layout,
    layout_from_blur,
    layout_from_detail,
    layout_from_rects,
    layout_from_regions,
    layout_from_vrs,
    shape_features,
)
from sanity import (  # noqa: E402
    block_macs,
    check_h3_bank_rejects_coarse_square,
    check_short_sequence_full_canvas,
)


def test_shape_and_centers():
    features = shape_features((1, 2, 4))
    assert torch.allclose(features, torch.tensor([1.0, 2.0, 3.0, -1.0]))
    timed = shape_features((2, 2, 2), include_time=True)
    assert timed.shape == (5,) and float(timed[-1]) == 1.0
    rect = TokenRect(3, 4, 8, 1, 2, 4)
    assert rect.center() == (3.0, 4.5, 9.5)
    layout = dense_layout(1, 1, 1)
    assert torch.equal(layout.centers(), torch.zeros(1, 3))
    assert layout.compression == 1.0


def test_partition_rules():
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    layout = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    assert layout.count == 7
    detail_t = torch.zeros(2, 4, 4)
    detail_t[1, 0, 0] = 1
    timed = layout_from_detail(detail_t, {4: 0.5, 2: 0.5}, root=4)
    assert [rect.eh for rect in timed.rects if rect.t == 0] == [4]
    assert sum(rect.t == 1 for rect in timed.rects) == 7
    assert layout.rects[0] == TokenRect(0, 0, 0, 1, 1, 1)
    assert any(rect.eh == 2 for rect in layout.rects)
    try:
        layout_from_detail(detail, {4: 0.8, 2: 0.2}, root=4)
    except ValueError:
        pass
    else:
        raise AssertionError("decreasing thresholds should fail")
    try:
        layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2), TokenRect(0, 0, 0, 1, 1, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping rects should fail")


def test_regions_prefer_finer_level():
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[:2, :2] = True
    layout = layout_from_regions(8, 8, background_level=0, regions=[(mask, 3)], root=8)
    covering = [rect for rect in layout.rects if rect.u == 0 and rect.v == 0]
    assert len(covering) == 1 and covering[0].extent == (1, 1, 1)
    assert any(rect.eh == 4 for rect in layout.rects)
    assert layout.dense_count == 64


def test_blur_and_vrs():
    depth = torch.full((8, 8), 10.0)
    depth[0, 0] = 1.0
    radius = blur_radius(depth, focal_depth=1.0, r_target=4.0)
    assert float(radius[0, 0]) < 1e-4
    assert float(radius[7, 7]) > 3.0
    layout = layout_from_blur(radius, {8: 3.0, 4: 1.0, 2: 0.2}, root=8)
    origin = next(rect for rect in layout.rects if rect.u == 0 and rect.v == 0)
    assert origin.eh == 1 and origin.ew == 1
    assert any(rect.eh == 4 for rect in layout.rects)

    flat = layout_from_vrs(torch.ones(8, 8), sensitivity=0.05, ambient=0.05, root=8)
    assert flat.count == 1 and flat.rects[0].extent == (1, 8, 8)
    spiked = torch.zeros(8, 8)
    spiked[0, 0] = 1
    busy = layout_from_vrs(spiked, sensitivity=0.01, ambient=0.01, root=8)
    assert busy.count > 1


def test_procrustes_and_mean_basis():
    basis = mean_basis(4, (1, 2, 2))
    assert basis.shape == (16, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-6)
    assert torch.allclose(mean_basis(5, (1, 1, 1)), torch.eye(5))

    torch.manual_seed(0)
    dense_dim, token_dim, rows = 12, 4, 64
    guess = torch.linalg.qr(torch.randn(dense_dim, token_dim, dtype=torch.float64), mode="reduced").Q
    reference = torch.randn(rows, token_dim, dtype=torch.float64)
    dense = 2.0 * reference @ guess.T
    fitted, scale = fit_procrustes(dense.float(), reference.float())
    assert torch.allclose(fitted.double(), guess, atol=1e-4)
    assert torch.allclose(scale, torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(fitted.T @ fitted, torch.eye(token_dim), atol=1e-4)


def test_velocity_recovery_matches_equation_9():
    try:
        _calibration, _mixin, path = _asymflow()
        assert path.name == "common.py" and "LakonLab" in str(path)
    except FileNotFoundError:
        pass   # no LakonLab checkout: recover_dense_velocity uses the plain-torch eq. 9 below
    basis = mean_basis(4, (1, 1, 3))
    y0 = torch.randn(2, 12)
    eps = torch.randn(2, 12)
    sigma = torch.tensor([0.2, 0.8])
    y_t = (1 - sigma)[:, None] * y0 + sigma[:, None] * eps
    predicted = asymmetric_target(y0, eps, basis)
    recovered = recover_dense_velocity(predicted, y_t, basis, sigma)
    assert torch.allclose(recovered, eps - y0, atol=1e-5)
    y0_hat = clean_from_velocity(y_t, recovered, sigma)
    assert float(lot_clean_loss(y0_hat, y0, sigma)) < 1e-8
    assert torch.allclose(euler_step(eps, eps - y0, 1.0, 0.0), y0, atol=1e-6)


def test_unit_layout_reproduces_pretrained_head():
    torch.manual_seed(1)
    batch, token_dim, hidden = 2, 4, 8
    adapter = LotVisualAdapter(token_dim, hidden, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(hidden, token_dim)
    bias_in = torch.randn(hidden)
    weight_out = torch.randn(token_dim, hidden)
    bias_out = torch.randn(token_dim)
    adapter.init_from_pretrained(weight_in, weight_out, bias_in, bias_out)
    tokens = torch.randn(batch, 1, 2, 2, token_dim)
    layout = dense_layout(1, 2, 2)
    velocity = adapter(tokens, 0.4, layout, lambda hidden_states, _coords, _sigma: hidden_states)
    flat = tokens.reshape(batch, 4, token_dim)
    manual = (flat @ weight_in.T + bias_in) @ weight_out.T + bias_out
    assert torch.allclose(velocity.reshape(batch, 4, token_dim), manual, atol=1e-5)

    coarse = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    embedded, centers = adapter.embed(tokens, coarse)
    packed = tokens.reshape(batch, 1, token_dim * 4)
    compressed = compress_patches(packed, adapter.bank.basis((1, 2, 2)))
    shared = compressed @ weight_in.T + bias_in
    assert torch.allclose(embedded, shared, atol=1e-5)
    assert torch.allclose(centers, torch.tensor([[0.0, 0.5, 0.5]]))


def test_extent_scale_and_backward():
    torch.manual_seed(2)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.bank.fit_((1, 2, 2), dense, reference)
    assert torch.allclose(adapter.bank.scale((1, 2, 2)), torch.tensor(2.0), atol=1e-4)
    clean = torch.randn(1, 1, 2, 2, 4)
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    scaled = adapter.bank.scale_clean(clean, layout)
    assert torch.allclose(adapter.bank.unscale(scaled, layout), clean, atol=1e-5)

    y0 = torch.randn(2, 1, 4, 4, 4)
    detail = torch.zeros(4, 4)
    detail[0, 0] = 1
    mixed = layout_from_detail(detail, {4: 0.5, 2: 0.5}, root=4)
    y_t, _noise = sample_noisy(y0, 0.5)
    velocity = adapter(y_t, 0.5, mixed, lambda states, _coords, _sigma: states)
    loss = lot_clean_loss(clean_from_velocity(y_t, velocity, 0.5), y0, 0.5)
    loss.backward()
    assert adapter.out_proj["1x1x1"].weight.grad.abs().sum() > 0
    assert adapter.out_proj["1x2x2"].weight.grad.abs().sum() > 0
    assert adapter.bank.basis((1, 2, 2)).grad is None


def test_h3_patch_geometry():
    assert H3_TOKEN_DIM == 96
    assert H3_PATCH == (1, 2, 2)
    assert H3_HIDDEN == 5376
    assert len(H3_EXTENTS) == 9
    assert over_wan_frame_budget(3073, 1)
    assert not over_wan_frame_budget(3072, 1)
    latent = torch.zeros(1, 2, 1, 2, 2)
    latent[0, 1, 0, 0, 1] = 7
    tokens = patchify(latent, patch=(1, 2, 2))
    assert tokens.shape[-1] == 8
    assert float(tokens[0, 0, 0, 0, 5]) == 7
    video = torch.randn(1, 24, 2, 4, 6)
    roundtrip = unpatchify(patchify(video))
    assert torch.equal(roundtrip, video)
    adapter = make_h3_adapter(hidden_size=32)
    assert adapter.token_dim == 96
    assert adapter.bank.basis((1, 4, 4)).shape == (16 * 96, 96)


def test_fit_extent_rebuilds_heads():
    torch.manual_seed(3)
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    weight_in = torch.randn(8, 4)
    weight_out = torch.randn(4, 8)
    bias_out = torch.randn(4)
    adapter.init_from_pretrained(weight_in, weight_out, None, bias_out)
    before = adapter.out_proj["1x2x2"].weight.detach().clone()
    guess = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    reference = torch.randn(32, 4)
    dense = 2.0 * reference @ guess.T
    adapter.fit_extent((1, 2, 2), dense, reference)
    expected = guess.float() @ weight_out
    assert torch.allclose(adapter.out_proj["1x2x2"].weight, expected, atol=1e-4)
    assert not torch.allclose(adapter.out_proj["1x2x2"].weight, before, atol=1e-4)
    assert torch.allclose(adapter.out_proj["1x1x1"].weight, weight_out, atol=1e-5)
    lifted = adapter.bank.basis((1, 2, 2)) @ bias_out
    assert torch.allclose(adapter.out_proj["1x2x2"].bias, lifted, atol=1e-4)


def test_h3_positions_match_base_grid():
    axis = torch.tensor([0.0, 10.0, 30.0], dtype=torch.float64)
    assert float(sample_axis(axis, 0)) == 0.0
    assert float(sample_axis(axis, 1.5)) == 20.0
    layout = dense_layout(2, 2, 3)
    positions = video_positions(layout, latent_height=4, latent_width=6, origin=5.0)
    area = math.sqrt(4 * 6)
    height = spatial_axis(4, 2, area)
    width = spatial_axis(6, 2, area)
    assert torch.allclose(positions[0, 1], height[0])
    assert torch.allclose(positions[1, 2], width[1])
    coarse = layout_from_rects(2, 2, 3, [
        TokenRect(0, 0, 0, 1, 2, 2),
        TokenRect(0, 0, 2, 1, 2, 1),
        TokenRect(1, 0, 0, 1, 2, 2),
        TokenRect(1, 0, 2, 1, 2, 1),
    ])
    coarse_pos = video_positions(coarse, 4, 6, origin=5.0)
    assert torch.allclose(coarse_pos[0, 1], 0.5 * (height[0] + height[1]))
    packed, video_start = packed_positions(layout, 4, 6, text_len=3, num_audio_latents=2)
    assert video_start == 3 + 4
    assert torch.allclose(packed[video_start:], video_positions(layout, 4, 6, origin=3.0))
    assert float(packed[0, 0]) == 0.0 and float(packed[2, 0]) == 2.0
    assert float(packed[3, 0]) == 3.0 and float(packed[4, 0]) == 4.0
    assert float(packed[5, 0]) == 3.0 and float(packed[5, 2]) != float(packed[3, 2])

    import sys as _sys
    from lot_paths import FIZGIG_SRC as fizgig
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.model import image_position_ids

    reference = image_position_ids(3, 4, 6, num_audio_latents=2, latent_t=2)
    assert torch.allclose(packed, reference)
    still = dense_layout(1, 2, 3)
    with_cond, _start = packed_positions(
        still, 4, 6, text_len=3, num_audio_latents=2, keyframes=[0], refs=[(4, 6), (4, 6, 2)],
    )
    cond_ref = image_position_ids(
        3, 4, 6, num_audio_latents=2, latent_t=1, keyframes=[0], refs=[(4, 6), (4, 6, 2)],
    )
    assert torch.allclose(with_cond, cond_ref)


def test_vectorized_gather_and_scale():
    torch.manual_seed(0)
    layout = layout_from_rects(1, 4, 4, [
        TokenRect(0, 0, 0, 1, 2, 2),
        TokenRect(0, 0, 2, 1, 2, 2),
        TokenRect(0, 2, 0, 1, 1, 2),
        TokenRect(0, 2, 2, 1, 1, 2),
        TokenRect(0, 3, 0, 1, 1, 4),
    ])
    tokens = torch.randn(2, 1, 4, 4, 3, requires_grad=True)
    gathered = []
    for _extent, indices in layout.groups():
        rects = [layout.rects[index] for index in indices]
        gathered.append(gather_extent(tokens, rects))
        slow = torch.stack([
            tokens[:, rect.t:rect.t + rect.et, rect.u:rect.u + rect.eh, rect.v:rect.v + rect.ew]
            .reshape(tokens.shape[0], -1)
            for rect in rects
        ], dim=1)
        assert torch.equal(gathered[-1], slow)
    canvas = tokens.new_zeros(tokens.shape)
    for (_extent, indices), values in zip(layout.groups(), gathered):
        scatter_extent(canvas, values, [layout.rects[index] for index in indices])
    assert torch.equal(canvas, tokens)
    scales = {extent: torch.tensor(2.0 + index) for index, (extent, _rects) in enumerate(layout.groups())}
    scaled = apply_extent_scales(tokens.detach(), layout, scales, invert=False)
    restored = apply_extent_scales(scaled, layout, scales, invert=True)
    assert torch.allclose(restored, tokens.detach())
    assert not torch.allclose(scaled, tokens.detach())


def test_gate_band_and_sigma():
    layout = gate_frame_layout()
    assert layout.count == 462
    assert layout.dense_count == 1008
    assert abs(layout.compression - 1008 / 462) < 1e-9
    assert {rect.extent for rect in layout.rects} == {(1, 1, 1), (1, 2, 2), (1, 4, 2)}
    coarse = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    adapter = LotVisualAdapter(4, 8, [(1, 1, 1), (1, 2, 2)])
    tokens = torch.randn(1, 1, 2, 2, 4)
    states = torch.randn(1, 1, 8)
    at_sigma = adapter.velocity_from_states(states, tokens, 0.4, coarse)
    swapped = adapter.velocity_from_states(states, tokens, 0.6, coarse)
    assert not torch.allclose(at_sigma, swapped)


def test_final_layer_sees_modulated_states():
    import sys as _sys
    from lot_paths import FIZGIG_SRC as fizgig
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.model import FinalLayer

    torch.manual_seed(1)
    hidden, token_dim, t_dim = 8, 4, 4
    layer = FinalLayer(hidden, t_dim, token_dim, 2, 1e-6)
    adapter = LotVisualAdapter(token_dim, hidden, [(1, 1, 1)])
    adapter.init_from_pretrained(torch.randn(hidden, token_dim), layer.video_out.weight.detach(), None, layer.video_out.bias.detach())
    states = torch.randn(3, hidden)
    t_emb = torch.randn(1, t_dim)
    splice = LotSplice(adapter, dense_layout(1, 1, 1))
    modulated = splice.modulate(layer, states, t_emb, 0)
    assert torch.allclose(adapter.out_proj["1x1x1"](modulated), layer.video_out(modulated))
    assert not torch.allclose(adapter.out_proj["1x1x1"](states), layer.video_out(modulated))


def test_splice_shortens_and_cached_refuses():
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    adapter = LotVisualAdapter(4, 8, [(1, 2, 2)])
    splice = LotSplice(adapter, layout)
    rows = torch.randn(4, 4)
    tokens = splice.video_tokens(rows, 1, 4, 4)
    assert splice.embed_rows(tokens).shape == (1, 8)
    dense_pos = torch.zeros(2 + 4, 3, dtype=torch.float64)
    dense_pos[-4:, 0] = 5
    replaced = splice.replace_video_positions(dense_pos, 1, 4, 4)
    assert replaced.shape == (3, 3)
    assert float(replaced[-1, 0]) == 5.0
    projected = splice.project_rows(torch.randn(1, 8), tokens, 0.4)
    assert projected.shape == (4, 4)

    import sys as _sys
    from lot_paths import FIZGIG_SRC as fizgig
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.model import MiniMaxH3DiT

    model = MiniMaxH3DiT.__new__(MiniMaxH3DiT)
    model._lot = splice
    try:
        MiniMaxH3DiT.forward_cached(model, torch.zeros(1, 24, 1, 4, 4), torch.tensor(0.5), torch.zeros(1, 2, 8))
    except RuntimeError as exc:
        assert "forward_cached" in str(exc)
    else:
        raise AssertionError("forward_cached accepted a LoT tail")
    model._tread = (0.5, 0, 1)
    try:
        MiniMaxH3DiT.forward(model, torch.zeros(1, 24, 1, 4, 4), torch.tensor(0.5), torch.zeros(1, 2, 8))
    except RuntimeError as exc:
        assert "TREAD" in str(exc)
    else:
        raise AssertionError("forward accepted LoT and TREAD together")


def test_sanity_short_sequence_writes_nothing():
    def refuse_save(*_args, **_kwargs):
        raise AssertionError("sanity check tried to write a checkpoint")

    original = torch.save
    torch.save = refuse_save
    try:
        numbers = check_short_sequence_full_canvas()
        check_h3_bank_rejects_coarse_square()
    finally:
        torch.save = original
    assert numbers["tokens"] == 6 and numbers["dense"] == 64
    assert numbers["macs"] == block_macs(6, 32, 2, 2)
    assert numbers["macs"] < numbers["dense_macs"] / 2


def test_euler_inference_recovers_clean():
    layout = dense_layout(1, 2, 2)
    clean = torch.randn(2, 1, 2, 2, 3)
    noise = torch.randn_like(clean)

    def predict(_state, _t, _layout):
        return noise - clean

    sampled = integrate(predict, noise, layout, sigma_grid(4))
    assert torch.allclose(sampled, clean, atol=1e-5)
    assert sigma_grid(4).shape == (5,)
    assert float(sigma_grid(1)[0]) == 1.0 and float(sigma_grid(1)[-1]) == 0.0


def test_h3_head_sign_recovery():
    """H3's head is ``x0 - P eps``. Eq. 9 needs the flip; a 1x1 basis cannot show it."""
    torch.manual_seed(5)
    adapter = LotVisualAdapter(4, 16, [(1, 2, 2)])
    with torch.no_grad():
        adapter.out_proj["1x2x2"].weight.copy_(torch.eye(16))
        adapter.out_proj["1x2x2"].bias.zero_()
    layout = layout_from_rects(1, 2, 2, [TokenRect(0, 0, 0, 1, 2, 2)])
    basis = adapter.bank.basis((1, 2, 2))
    y0 = torch.randn(1, 1, 2, 2, 4)
    eps = torch.randn_like(y0)
    sigma = 0.3
    y_t = (1 - sigma) * y0 + sigma * eps
    head = -asymmetric_target(y0.reshape(1, 16), eps.reshape(1, 16), basis)
    states = head.reshape(1, 1, 16)
    h3_velocity = adapter.velocity_from_states(states, y_t, sigma, layout, x0_minus_eps=True)
    assert torch.allclose(h3_velocity, y0 - eps, atol=1e-5)
    wrong = adapter.velocity_from_states(states, y_t, sigma, layout)
    assert not torch.allclose(-wrong, y0 - eps, atol=1e-2)
    # One H3 step to sigma 0 (Fizgig sampling.py) lands on the clean latent.
    assert torch.allclose(y_t + sigma * h3_velocity, y0, atol=1e-5)


def test_splice_y_space_contract():
    import sys as _sys
    from lot_paths import FIZGIG_SRC as fizgig
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.model import patchify_video

    torch.manual_seed(6)
    adapter = make_h3_adapter(hidden_size=8)
    guess = torch.linalg.qr(torch.randn(384, 96), mode="reduced").Q
    reference = torch.randn(400, 96)
    adapter.bank.fit_((1, 2, 2), 2.0 * reference @ guess.T, reference)
    layout = layout_from_rects(1, 2, 4, [TokenRect(0, 0, 0, 1, 2, 2), *(
        TokenRect(0, u, v, 1, 1, 1) for u in range(2) for v in range(2, 4))])
    x0 = torch.randn(1, 24, 1, 4, 8)
    rows = patchify_video(x0, (1, 2, 2))
    assert torch.equal(rows.reshape(1, 1, 2, 4, 96), patchify(x0))

    x_space = LotSplice(adapter, layout)
    assert not x_space.unit_scales()
    try:
        x_space.video_tokens(rows, 1, 4, 8)
    except ValueError as exc:
        assert "y_t" in str(exc)
    else:
        raise AssertionError("fitted scales accepted an x-space splice")

    splice = LotSplice(adapter, layout, y_space=True)
    y0 = splice.to_y(x0)
    assert torch.allclose(patchify(y0), adapter.bank.scale_clean(patchify(x0), layout))
    assert torch.allclose(y0[..., :4, :4], x0[..., :4, :4] / 2.0, atol=1e-4)
    assert torch.equal(y0[..., 4:], x0[..., 4:])
    assert torch.allclose(splice.to_x(y0), x0, atol=1e-5)
    assert splice.video_tokens(patchify_video(y0, (1, 2, 2)), 1, 4, 8).shape == (1, 1, 2, 4, 96)
    assert LotSplice(adapter, dense_layout(1, 2, 4)).unit_scales()


def test_procrustes_h3_fit_and_guards():
    from procrustes_h3 import check_pair, fit_bank, load_pairs, parse_extent

    try:
        load_pairs(Path("/nonexistent/lot_pairs"))
    except SystemExit as exc:
        assert "does not invent" in str(exc)
    else:
        raise AssertionError("a missing pair directory did not exit")
    assert parse_extent("1x4x2") == (1, 4, 2)
    for bad in ("1x4", "a_b_c"):
        try:
            parse_extent(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} parsed")
    try:
        check_pair((1, 2, 2), torch.randn(50, 384), torch.randn(50, 96))
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("too few rows were accepted")

    torch.manual_seed(7)
    adapter = make_h3_adapter(hidden_size=8)
    try:
        fit_bank(adapter, {})
    except RuntimeError:
        pass
    else:
        raise AssertionError("fit before init_from_pretrained")
    adapter.init_from_pretrained(torch.randn(8, 96), torch.randn(96, 8), torch.randn(8), torch.randn(96))
    pairs = {}
    for extent, true_scale in (((1, 2, 2), 2.0), ((1, 4, 2), 0.5)):
        dim = 96 * extent[1] * extent[2]
        lift = torch.linalg.qr(torch.randn(dim, 96), mode="reduced").Q
        reference = torch.randn(300, 96)
        pairs[extent] = check_pair(extent, true_scale * reference @ lift.T, reference)
    reports = {tuple(r["extent"]): r for r in fit_bank(adapter, pairs)}
    assert abs(reports[(1, 2, 2)]["scale"] - 2.0) < 1e-3
    assert abs(reports[(1, 4, 2)]["scale"] - 0.5) < 1e-3
    assert all(r["ortho_err"] < 1e-4 for r in reports.values())
    assert float(adapter.bank.scale((1, 1, 1))) == 1.0


def test_phase4_smoke_canvas():
    layout = clip_layout(7, 12, 20)
    assert layout.count == 7 * (4 * 20 + 2 * 10 + 10)
    assert {rect.extent for rect in layout.rects} == {(1, 1, 1), (1, 2, 2), (1, 4, 2)}
    assert clip_layout(37).count == 17094
    try:
        clip_layout(1, 10, 20)
    except ValueError:
        pass
    else:
        raise AssertionError("a 10-row grid was banded")


def test_pair_rows_align_with_coarse_grid():
    """Row i of ``dense`` must cover the pixels of reference token i (row-major)."""
    from make_pairs_h3 import block_rects

    tokens = torch.randn(1, 1, 8, 4, 3)
    for extent in ((1, 2, 2), (1, 4, 2), (1, 2, 4), (1, 1, 2)):
        _et, eh, ew = extent
        if 4 % ew:
            continue
        dense = gather_extent(tokens, block_rects(8, 4, extent))[0]
        pooled = tokens[0, 0].reshape(8 // eh, eh, 4 // ew, ew, 3).mean(dim=(1, 3)).reshape(-1, 3)
        site_mean = dense.reshape(dense.shape[0], eh * ew, 3).mean(dim=1)
        assert torch.allclose(site_mean, pooled, atol=1e-6), extent


def test_h3_loss_is_velocity_mse_in_y_space():
    torch.manual_seed(8)
    y0 = torch.randn(1, 24, 1, 4, 4)
    eps = torch.randn_like(y0)
    for sigma in (0.3, 0.92):
        y_t = (1 - sigma) * y0 + sigma * eps
        out = torch.randn_like(y0)
        velocity_mse = (out - (y0 - eps)).square().mean()
        assert torch.allclose(lot_h3_clean_loss(out, y_t, y0, sigma), velocity_mse, atol=1e-5)
        assert float(lot_h3_clean_loss(y0 - eps, y_t, y0, sigma)) < 1e-10
    # Below the floor the weight stops growing.
    small = 0.01
    y_t = (1 - small) * y0 + small * eps
    out = torch.randn_like(y0)
    assert float(lot_h3_clean_loss(out, y_t, y0, small)) < float((out - (y0 - eps)).square().mean())


def test_train_layouts_tile_and_mix():
    import random as _random
    from train_h3 import _tile, sample_layout

    torch.manual_seed(9)
    latent = torch.randn(1, 24, 1, 32, 56)                 # 16x28 tokens, 4x7 super-cells
    latent[..., :8, :8] *= 6.0                              # one detailed corner
    kinds = {}
    for seed in range(40):
        layout, kind = sample_layout(latent, _random.Random(seed))
        kinds.setdefault(kind, layout)
        assert layout.dense_count == 16 * 28 and layout.count <= layout.dense_count
        assert {rect.extent for rect in layout.rects} <= set(H3_EXTENTS)
    assert {"dense", "uniform", "mosaic", "box", "focus", "texture"} <= set(kinds) | {"mosaic"}
    for name in ("dense", "uniform", "mosaic", "box", "focus", "texture"):
        layout, kind = sample_layout(latent, _random.Random(1), force=name)
        assert kind == name and layout.dense_count == 16 * 28
        assert {rect.extent for rect in layout.rects} <= set(H3_EXTENTS)
    clip = torch.randn(1, 24, 7, 32, 56)
    moving, _ = sample_layout(clip, _random.Random(3), force="box")
    per_frame = [frozenset((r.u, r.v) for r in moving.rects if r.t == t and r.extent == (1, 1, 1)) for t in range(7)]
    assert len(set(per_frame)) > 1                          # the box drifts across latent frames
    from train_h3 import _legacy_mosaic, parse_mix
    assert sample_layout(latent, _random.Random(5), force="eval_mosaic")[0] == _legacy_mosaic(latent, _random.Random(5), 0.0)[0]
    assert parse_mix("box:0.5,focus:0.5") == {"box": 0.5, "focus": 0.5}
    mosaic = sample_layout(latent, _random.Random(11), force="mosaic")[0]
    kinds["mosaic"] = mosaic
    mosaic = kinds["mosaic"]
    assert len({rect.extent for rect in mosaic.rects}) > 1
    corner = {rect.extent for rect in mosaic.rects if rect.u < 4 and rect.v < 4}
    assert corner == {(1, 1, 1)}                            # the detailed cell stays fine
    for extent in H3_EXTENTS:
        assert _tile(1, 8, 8, lambda *_: extent).count == 64 // (extent[1] * extent[2])


def test_train_shift_moves_mass_to_low_sigma():
    import sys as _sys
    from lot_paths import FIZGIG_SRC as fizgig
    if fizgig not in _sys.path:
        _sys.path.insert(0, fizgig)
    from fizgig.minimax.trainer import sample_sigmas
    from train_h3 import parse_shift

    assert parse_shift("12") == 12.0 and parse_shift("sigmoid") == "sigmoid"
    assert parse_shift("lognorm:3") == "lognorm:3"
    gen = torch.Generator().manual_seed(0)
    h3 = sample_sigmas(20000, "cpu", shift=parse_shift("12"), generator=gen)
    low = sample_sigmas(20000, "cpu", shift=parse_shift("3"), generator=gen)
    assert float((h3 < 0.3).float().mean()) < 0.06            # H3's own: ~3.5% below 0.3
    assert float((low < 0.3).float().mean()) > 0.10            # shift 3: several times more
    assert float(low.median()) < float(h3.median())


def test_torch_eq9_matches_lakonlab():
    from flow import _broadcast_sigma, asymflow_velocity_torch

    torch.manual_seed(10)
    basis = torch.linalg.qr(torch.randn(16, 4), mode="reduced").Q
    u_a = torch.randn(3, 5, 16)
    x_t = torch.randn(3, 5, 16)
    for sigma in (0.05, torch.tensor([0.2, 0.5, 0.9])):
        lakon = recover_dense_velocity(u_a, x_t, basis, sigma)
        plain = asymflow_velocity_torch(u_a, x_t, basis, _broadcast_sigma(sigma, u_a))
        assert torch.allclose(lakon, plain, atol=1e-5)


def test_grid_layouts_any_shape():
    from h3 import GRID_LAYOUTS, grid_layout

    for height, width in ((36, 24), (19, 30), (24, 42), (12, 20)):
        for name in GRID_LAYOUTS:
            layout = grid_layout(name, 1, height, width)
            assert layout.dense_count == height * width
            if name == "dense":
                assert layout.count == layout.dense_count
            else:
                assert layout.count < layout.dense_count
    center = grid_layout("center", 1, 36, 24)
    middle = [r.extent for r in center.rects if 16 <= r.u < 20 and 8 <= r.v < 16]
    corner = [r.extent for r in center.rects if r.u < 4 and r.v < 4]
    assert set(middle) == {(1, 1, 1)} and set(corner) == {(1, 4, 4)}
    bands = grid_layout("bands", 1, 36, 24)
    assert {r.extent for r in bands.rects if r.u < 12} == {(1, 1, 1)}
    assert {r.extent for r in bands.rects if r.u >= 24} == {(1, 4, 2)}


def test_layout_sources():
    from layout_sources import (boxes_to_mask, compression, depth_layout, latent_frame_spans,
                                layout_from_desired_any, mask_layout, texture_layout, to_latent_frames)

    assert latent_frame_spans(7)[-1][1] == 22 and latent_frame_spans(1) == [(0, 1)]
    # mask: box in the middle of a 768x512 still -> fine inside, coarse outside
    mask = boxes_to_mask([(0.375, 0.375, 0.625, 0.625)], 768, 512)
    lay = mask_layout(mask, 1, 24, 16, dilate=0)
    inside = {r.extent for r in lay.rects if 10 <= r.u < 14 and 7 <= r.v < 9}
    corner = {r.extent for r in lay.rects if r.u < 4 and r.v < 4}
    assert inside == {(1, 1, 1)} and corner == {(1, 4, 4)}
    assert lay.dense_count == 24 * 16 and 1.0 < compression(lay) < 16
    # a moving box over 22 pixel frames -> the layout changes per latent frame
    frames = torch.zeros(22, 768, 512)
    for f in range(22):
        x = 0.1 + 0.6 * f / 21
        frames[f] = boxes_to_mask([(x, 0.4, x + 0.2, 0.6)], 768, 512)[0]
    moving = mask_layout(frames, 7, 24, 16, dilate=0)
    first = {(r.u, r.v) for r in moving.rects if r.t == 0 and r.extent == (1, 1, 1)}
    last = {(r.u, r.v) for r in moving.rects if r.t == 6 and r.extent == (1, 1, 1)}
    assert first and last and first != last
    # depth: in-focus band fine, far plane coarse
    depth = torch.linspace(0, 1, 768)[:, None].expand(768, 512)
    dl = depth_layout(depth, 1, 24, 16, focus=1.0, band=0.15)
    assert {r.extent for r in dl.rects if r.u >= 22} == {(1, 1, 1)}
    assert (1, 4, 4) in {r.extent for r in dl.rects if r.u < 8}
    # texture: noisy half fine, flat half coarse
    image = torch.zeros(1, 768, 512, 3)
    image[:, :, :256] = torch.rand(1, 768, 256, 3)
    tl = texture_layout(image, 1, 24, 16, fine_q=0.5, mid_q=0.3)
    assert {r.extent for r in tl.rects if r.v < 6} == {(1, 1, 1)}
    assert (1, 4, 4) in {r.extent for r in tl.rects if r.v >= 10}
    # odd grids: edges stay 1x1 and every token is covered exactly once
    odd = layout_from_desired_any(torch.full((2, 19, 30), 4))
    assert odd.dense_count == 2 * 19 * 30
    assert to_latent_frames(torch.rand(5, 3, 3), 2).shape == (2, 3, 3)


def main():
    tests = [
        test_shape_and_centers,
        test_partition_rules,
        test_regions_prefer_finer_level,
        test_blur_and_vrs,
        test_procrustes_and_mean_basis,
        test_velocity_recovery_matches_equation_9,
        test_unit_layout_reproduces_pretrained_head,
        test_extent_scale_and_backward,
        test_h3_patch_geometry,
        test_fit_extent_rebuilds_heads,
        test_h3_positions_match_base_grid,
        test_sanity_short_sequence_writes_nothing,
        test_vectorized_gather_and_scale,
        test_gate_band_and_sigma,
        test_final_layer_sees_modulated_states,
        test_splice_shortens_and_cached_refuses,
        test_euler_inference_recovers_clean,
        test_h3_head_sign_recovery,
        test_splice_y_space_contract,
        test_procrustes_h3_fit_and_guards,
        test_phase4_smoke_canvas,
        test_pair_rows_align_with_coarse_grid,
        test_h3_loss_is_velocity_mse_in_y_space,
        test_train_layouts_tile_and_mix,
        test_train_shift_moves_mass_to_low_sigma,
        test_torch_eq9_matches_lakonlab,
        test_grid_layouts_any_shape,
        test_layout_sources,
    ]
    for test in tests:
        test()
        print(test.__name__)
    print(f"ok {len(tests)}")


if __name__ == "__main__":
    main()
