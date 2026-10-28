"""H3 through musubi-tuner, with an optional Level-of-Token target-video tail.

musubi-tuner (https://github.com/kohya-ss/musubi-tuner, Apache-2.0) loads the
published ComfyUI ConvRot int8 checkpoints (pruned or full), streams blocks
from CPU, checkpoints activations, and trains LoRAs on the int8 base. This
module uses it as a library and changes none of its code. ``predict`` runs
the dense model, or, with ``lot=LotSplice(...)``, the same packed sequence
with three changes on the target video only:

1. its rows are embedded by the splice's extent heads (fewer, larger tokens),
2. their rotary positions are the rectangle centres,
3. the final norm + AdaLN feed the splice's heads instead of ``video_out``.

Text, audio, and the block stack are untouched. The output is H3's dense
``x0 - eps`` velocity in every case, so the samplers step it unchanged.

Audio: a still or clip trained without sound still packs H3's audio block,
because the base was trained with it. Those rows are silence (``x0 = 0``)
noised at the same schedule position on the audio shift (3) as the video
rows are on the video shift (12). They carry no loss.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import torch

from lot_paths import LOT_H3_CHECKPOINT, MUSUBI_SRC

if MUSUBI_SRC not in sys.path:
    sys.path.insert(0, MUSUBI_SRC)

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
AUDIO_CHANNELS = 32
LORA_INCLUDE = r"blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))"


# ---------------------------------------------------------------------------- schedules

def shift_sigma(sigma, shift: float):
    """``shift * sigma / (1 + (shift - 1) * sigma)``."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def unshift_sigma(sigma, shift: float):
    """Inverse of ``shift_sigma``."""
    return sigma / (shift + sigma * (1.0 - shift))


def audio_sigma(video_sigma):
    """The audio sigma at the same base-schedule position as ``video_sigma``."""
    return shift_sigma(unshift_sigma(video_sigma, VIDEO_SHIFT), AUDIO_SHIFT)


def sample_sigma(shift, device, generator: torch.Generator | None = None, count: int = 1) -> torch.Tensor:
    """``count`` training sigmas in (0, 1), shape ``(count,)``.

    A float ``S`` maps a uniform base through the shift (12 is H3's own
    density, the same as musubi's ``--h3_shift_video``). ``"sigmoid"`` is an
    unshifted logit-normal. ``"lognorm:S"`` is a logit-normal base under shift ``S``.
    """
    if shift == "sigmoid":
        return torch.sigmoid(torch.randn(count, device=device, generator=generator))
    if isinstance(shift, str) and shift.startswith("lognorm:"):
        base = torch.sigmoid(torch.randn(count, device=device, generator=generator))
        return shift_sigma(base, float(shift.split(":", 1)[1]))
    return shift_sigma(torch.rand(count, device=device, generator=generator), float(shift))


def pixel_frames(latent_t: int) -> int:
    """Pixel frames a ``latent_t``-frame latent decodes to: 1, 5, 9, 13, 17, 22, ..."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def latent_frames(num_frames: int) -> int:
    """Pixel frames (snapped down onto 17n+5) -> latent frames; 1 is a still."""
    if num_frames <= 1:
        return 1
    n = max(5, int(num_frames))
    n -= (n - 5) % 17
    return 5 * (n - 5) // 17 + 2


def audio_frames(latent_t: int) -> int:
    """40 Hz audio latents covering the clip, H3's ``(10 f + 3) // 6`` for ``f`` pixel frames."""
    return (10 * pixel_frames(latent_t) + 3) // 6


# ------------------------------------------------------------------------------ loading

def load_dit(checkpoint=LOT_H3_CHECKPOINT, *, device="cuda", swap: int = 0, train: bool = False):
    """The H3 transformer, frozen, ready for ``predict``.

    ``swap`` blocks stream from CPU (H2D only: the base is frozen). ``train``
    turns on activation checkpointing and lets the swap run a backward pass.
    """
    from musubi_tuner.minimax_h3.model import load_h3_transformer
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    device = torch.device(device)
    dit = load_h3_transformer(str(checkpoint), device="cpu" if swap else device)
    dit.requires_grad_(False)
    if swap:
        dit.enable_block_swap(swap, BlockSwapConfig(device=device, supports_backward=train, h2d_only=True))
        dit.move_to_device_except_swap_blocks(device)
        dit.prepare_block_swap_before_forward()
    if train:
        dit.enable_gradient_checkpointing()
        dit.train()
    return dit


def make_lora(dit, rank: int):
    """A rank-``rank`` LoRA (alpha = rank) on qkv, out, fc1, fc2 of every block, applied to ``dit``."""
    from musubi_tuner.networks import lora_minimax_h3

    network = lora_minimax_h3.create_arch_network(1.0, rank, float(rank), None, [], dit)
    network.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    expected = 4 * len(dit.blocks)
    if len(network.unet_loras) != expected:
        raise RuntimeError(f"{len(network.unet_loras)} LoRA modules wrapped, expected {expected}")
    return network


def set_lora(network, on: bool) -> None:
    for module in network.unet_loras:
        module.multiplier = 1.0 if on else 0.0


# ------------------------------------------------------------------------------ forward

def _sync_swap(dit) -> None:
    """Put a streaming offloader in forward-only mode under no_grad and back for a training forward."""
    if not dit.blocks_to_swap or not dit.offloader.supports_backward:
        return
    forward_only = not torch.is_grad_enabled()
    if dit.offloader.forward_only != forward_only:
        if forward_only:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()


def _layout(text_length: int, latent_t: int, lat_h: int, lat_w: int):
    from musubi_tuner.minimax_h3.packing import build_h3_layout

    return build_h3_layout(task="t2va", text_length=text_length, target_video=(latent_t, lat_h, lat_w),
                           target_audio_frames=audio_frames(latent_t), one_frame=latent_t == 1)


def _silence(latent_t: int, sigma_a: float, device, generator=None) -> torch.Tensor:
    """``(1, 32, 2, A)`` silent audio noised to ``sigma_a``."""
    eps = torch.randn(1, AUDIO_CHANNELS, 2, audio_frames(latent_t), device=device, generator=generator)
    return sigma_a * eps


def predict(dit, latent: torch.Tensor, t, text: torch.Tensor, *, lot=None,
            audio_generator: torch.Generator | None = None) -> torch.Tensor:
    """H3's velocity (``x0 - eps``) for ``latent`` ``(1, 24, T, H, W)`` at cleanness ``t``.

    ``text`` is ``(1, L, 5120)`` (or ``(L, 5120)``) Qwen3-VL layer-50 states.
    With ``lot`` (a ``LotSplice``), ``latent`` is the splice's ``y_t`` and the
    target video runs on its layout.
    """
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(f"latent must be (1, 24, T, H, W), got {tuple(latent.shape)}")
    if text.ndim == 2:
        text = text.unsqueeze(0)
    _sync_swap(dit)
    t_video = float(torch.as_tensor(t).reshape(-1)[0])
    sigma_a = float(audio_sigma(1.0 - t_video))
    _b, _c, latent_t, lat_h, lat_w = latent.shape
    layout = _layout(text.shape[1], latent_t, lat_h, lat_w)
    device = dit.video_patch_proj.weight.device
    audio = _silence(latent_t, sigma_a, device, audio_generator)
    tags = torch.ones(1, text.shape[1], dtype=torch.int64)
    if lot is None:
        return dit(video_latents=latent, audio_latents=audio, text_hidden_states=text, text_token_tags=tags,
                   layout=layout, model_t_video=t_video, model_t_audio=1.0 - sigma_a).video
    return _lot_forward(dit, lot, layout, latent, audio, text, tags, t_video, 1.0 - sigma_a)


def _lot_forward(dit, lot, layout, latent, audio, text, tags, t_video: float, t_audio: float) -> torch.Tensor:
    """``MiniMaxH3Model.forward`` for a T2VA layout, with the target video on ``lot.layout``."""
    from musubi_tuner.minimax_h3.packing import (
        H3RowSegment,
        build_position_grid,
        build_timestep_rows,
        pack_audio_rows,
        pack_video_rows,
        unpack_targets,
    )
    from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

    if layout.task != "t2va":
        raise ValueError("the LoT tail is wired for T2VA layouts only")
    video_segment = layout.target_video_segment
    if video_segment.stop != layout.row_count:
        raise RuntimeError("the target video must be the last segment of the packed sequence")
    device = dit.video_patch_proj.weight.device
    latent_t, lat_h, lat_w = latent.shape[2:]

    # 1. Target video: extent heads instead of video_patch_proj.
    tokens = lot.video_tokens(pack_video_rows(latent.to(device, torch.float32))[0], latent_t, lat_h, lat_w)
    video = lot.embed_rows(tokens).to(dit.dtype).unsqueeze(0)
    count = video.shape[1]
    segments = layout.segments[:-1] + (H3RowSegment(video_segment.role, video_segment.kind,
                                                    video_segment.start, video_segment.start + count),)
    lot_layout = replace(layout, segments=segments, row_count=video_segment.start + count)

    audio_rows = dit.audio_patch_proj(pack_audio_rows(audio.to(device)).to(torch.float32)).to(dit.dtype)
    text_rows = dit.token_refiner(dit.condition_proj(text.to(device, dit.dtype)))
    hidden = torch.cat([text_rows, audio_rows, video], dim=1)

    # 2. Rotary positions: the dense grid with the video tail replaced by rectangle centres.
    positions = lot.replace_video_positions(build_position_grid(layout)[0], latent_t, lat_h, lat_w)
    rotation = dit._rotation_table(positions.unsqueeze(0).to(device), hidden.dtype)
    rows = build_timestep_rows(lot_layout, text_token_tags=tags, model_t_video=t_video, model_t_audio=t_audio)
    temb = dit._timestep_embeddings(rows.unique_timesteps, device)

    for index, block in enumerate(dit.blocks):
        if dit.blocks_to_swap:
            dit.offloader.wait_for_block(index)
        if dit.gradient_checkpointing and dit.training:
            forward = block
            if dit.activation_cpu_offloading:
                forward = create_cpu_offloading_wrapper(forward, device)
            hidden = torch.utils.checkpoint.checkpoint(forward, hidden, temb, rows.block_segments, rotation,
                                                       use_reentrant=False)
        else:
            hidden = block(hidden, temb, rows.block_segments, rotation)
        if dit.blocks_to_swap:
            dit.offloader.submit_move_blocks_forward(dit.blocks, index)
    hidden = hidden.to(device)

    # 3. Final layer: shared norm + AdaLN, then the extent heads in place of video_out.
    final = dit.final_layer
    shift, scale = final.adaln_proj(temb)
    audio_index = rows.audio_timestep_index
    audio_states = final.norm(hidden[:, layout.target_audio_segment.row_slice])
    audio_states = audio_states * (1.0 + scale[audio_index]) + shift[audio_index]
    audio_out = final.audio_out(audio_states.to(torch.float32))
    modulated = lot.modulate(final, hidden[0, lot_layout.target_video_segment.row_slice], temb,
                             rows.video_timestep_index)
    velocity = lot.project_rows(modulated, tokens, max(0.0, min(1.0, 1.0 - t_video)))
    video_out, _audio = unpack_targets(layout, velocity.unsqueeze(0).to(torch.float32), audio_out)
    return video_out.to(latent.dtype)


# ------------------------------------------------------------------------------ sampling

def sigma_schedule(steps: int, shift: float = VIDEO_SHIFT) -> torch.Tensor:
    """``steps + 1`` sigmas from 1 to 0, linear in the base schedule, then shifted."""
    return shift_sigma(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64), shift)


@torch.no_grad()
def sample(dit, text: torch.Tensor, *, height: int, width: int, latent_t: int = 1, steps: int = 20,
           seed: int = 0, lot=None, device="cuda", on_step=None) -> torch.Tensor:
    """Euler sampling (``eta = 0``) of a ``(1, 24, T, height/16, width/16)`` latent.

    Same update as diffusers' ``MiniMaxH3Scheduler``: ``x0 = x + sigma * v``,
    then ``x <- r * x + (1 - r) * x0`` with ``r = sigma_next / sigma``. With
    ``lot`` the state is y-space; map the result back with ``lot.to_x``.
    """
    generator = torch.Generator(device).manual_seed(seed)
    x = torch.randn(1, 24, latent_t, height // 16, width // 16, device=device, generator=generator)
    audio_generator = torch.Generator(device).manual_seed(seed + 1)
    sigmas = sigma_schedule(steps)
    for index in range(steps):
        sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
        velocity = predict(dit, x, 1.0 - sigma, text, lot=lot, audio_generator=audio_generator).float()
        x0 = x + sigma * velocity
        ratio = sigma_next / sigma
        x = ratio * x + (1.0 - ratio) * x0
        if on_step is not None:
            on_step(index, sigma)
    return x


# ---------------------------------------------------------------------------------- VAE

def load_vae(path, device="cuda", *, decoder: bool = True, dtype: torch.dtype = torch.float16):
    """H3's video VAE.

    ``encode`` takes ``(1, 3, [F,] H, W)`` pixels in [-1, 1] and returns normalized
    latents; ``decode`` takes normalized latents and returns ``(1, 3, F, H, W)``
    pixels in [-1, 1]. ``decoder=False`` skips the decoder weights (encode only).
    """
    from musubi_tuner.minimax_h3.video_vae import load_video_vae

    return load_video_vae(str(path), device=device, dtype=dtype, load_decoder=decoder)
