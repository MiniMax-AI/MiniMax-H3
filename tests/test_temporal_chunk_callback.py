# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
import torch
import torch.distributed as dist

from FL2VA.video_vae.klvae import AutoencoderKL as FL2VAAutoencoderKL
from Ref2VA.video_vae.klvae import AutoencoderKL as Ref2VAAutoencoderKL


class _FakeTemporalDecoder:
    use_3d_conv = True
    tokens_chunk_size = 5
    token_overlap = 2
    token_drop = 3
    vae_ratio_t = 4
    frame_pre_padding = 3
    frame_overlap = 5
    clip_length = 17
    isolated_first_frame = False
    isolated_last_frame = False
    training = False

    def __init__(
        self,
        api: type,
        latent_t: int,
        *,
        collective: bool = False,
    ) -> None:
        for name in (
            "_decode_temporal_pad_frames",
            "_decode_temporal_output_chunk_plan",
            "_decode_temporal_output_frame_plan",
            "_decode_temporal_streaming",
            "decode_temporal",
            "decode_base",
        ):
            setattr(self, name, MethodType(getattr(api, name), self))

        pseudo_tokens = latent_t + self.token_drop
        remainder = pseudo_tokens % self.tokens_chunk_size
        self.pad_tokens = 0 if remainder == 0 else self.tokens_chunk_size - remainder
        self.num_chunks = (
            pseudo_tokens + self.pad_tokens
        ) // self.tokens_chunk_size - 1
        ideal_frames = self.num_chunks * 17 + 5
        output_frames = {
            33: 111,
            34: 115,
            35: 119,
            36: 120,
        }.get(latent_t, ideal_frames)
        values = torch.linspace(0, 1, ideal_frames).view(1, 1, ideal_frames, 1, 1)
        self.ideal = values.expand(1, 3, -1, 1, 1).contiguous()
        self.expected = self.ideal[:, :, :output_frames].clone()
        self.adaptive_decode_calls = 0
        self.collective = collective

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        call_index = self.adaptive_decode_calls
        self.adaptive_decode_calls += 1
        if self.collective:
            probe = torch.tensor([call_index], dtype=torch.int32)
            dist.all_reduce(probe)
        decoded = torch.full((1, 3, 28, 1, 1), -1.0)
        body_start = call_index * 17
        decoded[:, :, 3:20].copy_(self.ideal[:, :, body_start : body_start + 17])
        if call_index == self.num_chunks - 1:
            decoded[:, :, 23:28].copy_(self.ideal[:, :, -5:])
        return decoded

    def blend(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        extent: int,
        *,
        dim: int,
    ) -> torch.Tensor:
        del previous
        assert extent == self.frame_overlap
        assert dim == -3
        return current

    def trim_output(self, decoded: torch.Tensor, target_frames: int) -> torch.Tensor:
        return decoded[:, :, :target_frames]


class _EchoTemporalDecoder:
    use_3d_conv = True
    tokens_chunk_size = 3
    token_overlap = 2
    token_drop = 1
    vae_ratio_t = 2
    frame_pre_padding = 0
    frame_overlap = 4
    clip_length = 6
    training = False

    def __init__(
        self,
        api: type,
        *,
        isolated_first_frame: bool,
        isolated_last_frame: bool,
    ) -> None:
        for name in (
            "blend",
            "_decode_temporal_pad_frames",
            "_decode_temporal_output_chunk_plan",
            "_decode_temporal_output_frame_plan",
            "_decode_temporal_streaming",
            "decode_temporal",
        ):
            setattr(self, name, MethodType(getattr(api, name), self))
        self.isolated_first_frame = isolated_first_frame
        self.isolated_last_frame = isolated_last_frame

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.repeat_interleave(self.vae_ratio_t, dim=2)


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
@pytest.mark.parametrize(
    ("latent_t", "expected_chunk_sizes"),
    [
        (37, [17] * 7 + [5]),
        (36, [17] * 7 + [1]),
        (35, [17] * 7),
        (34, [17] * 6 + [13]),
        (33, [17] * 6 + [9]),
    ],
)
def test_callback_matches_legacy_temporal_decode(
    monkeypatch: pytest.MonkeyPatch,
    api: type,
    latent_t: int,
    expected_chunk_sizes: list[int],
) -> None:
    latent = torch.zeros(1, 3, latent_t, 1, 1)
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "0")
    baseline_model = _FakeTemporalDecoder(api, latent_t)
    baseline = baseline_model.decode_temporal(latent)

    callback_model = _FakeTemporalDecoder(api, latent_t)
    chunks: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, bool]] = []

    def consume(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        chunks.append(frames.clone())
        metadata.append((chunk_index, total_chunks, frame_start, is_final))

    output = callback_model.decode_temporal(
        latent,
        temporal_chunk_callback=consume,
    )

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), baseline, rtol=0, atol=0)
    assert [int(chunk.shape[2]) for chunk in chunks] == expected_chunk_sizes
    expected_starts = []
    frame_start = 0
    for index, chunk_size in enumerate(expected_chunk_sizes):
        expected_starts.append(
            (
                index,
                len(expected_chunk_sizes),
                frame_start,
                index == len(expected_chunk_sizes) - 1,
            )
        )
        frame_start += chunk_size
    assert metadata == expected_starts


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
def test_callback_publishes_only_after_overlap_blending(api: type) -> None:
    latent_t = 37
    model = _FakeTemporalDecoder(api, latent_t)
    chunks: list[torch.Tensor] = []

    def mark_overlap(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        extent: int,
        *,
        dim: int,
    ) -> torch.Tensor:
        del self, previous
        assert extent == 5
        assert dim == -3
        blended = current.clone()
        blended[:, :, :extent].fill_(0.25)
        return blended

    model.blend = MethodType(mark_overlap, model)
    output = model.decode_temporal(
        torch.zeros(1, 3, latent_t, 1, 1),
        temporal_chunk_callback=lambda frames, **metadata: chunks.append(
            frames.clone()
        ),
    )

    expected_overlap = torch.full_like(chunks[1][:, :, :5], 0.25)
    torch.testing.assert_close(chunks[1][:, :, :5], expected_overlap, rtol=0, atol=0)
    torch.testing.assert_close(output[:, :, 17:22], expected_overlap, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), output, rtol=0, atol=0)


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
@pytest.mark.parametrize("isolated_first_frame", [False, True])
@pytest.mark.parametrize("isolated_last_frame", [False, True])
def test_callback_preserves_isolated_head_and_tail_frames(
    monkeypatch: pytest.MonkeyPatch,
    api: type,
    isolated_first_frame: bool,
    isolated_last_frame: bool,
) -> None:
    latent = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1, 1)
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "0")
    baseline_model = _EchoTemporalDecoder(
        api,
        isolated_first_frame=isolated_first_frame,
        isolated_last_frame=isolated_last_frame,
    )
    baseline = baseline_model.decode_temporal(latent)

    callback_model = _EchoTemporalDecoder(
        api,
        isolated_first_frame=isolated_first_frame,
        isolated_last_frame=isolated_last_frame,
    )
    chunks: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, bool]] = []

    def consume(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        chunks.append(frames.clone())
        metadata.append((chunk_index, total_chunks, frame_start, is_final))

    output = callback_model.decode_temporal(
        latent,
        temporal_chunk_callback=consume,
    )

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), baseline, rtol=0, atol=0)
    assert metadata[-1][3]
    assert [item[0] for item in metadata] == list(range(len(metadata)))
    assert {item[1] for item in metadata} == {len(metadata)}
    assert [item[2] for item in metadata] == [
        sum(int(chunk.shape[2]) for chunk in chunks[:index])
        for index in range(len(chunks))
    ]


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
def test_callback_failure_waits_for_remaining_temporal_decodes(api: type) -> None:
    latent_t = 37
    model = _FakeTemporalDecoder(api, latent_t)
    callback_calls = 0

    def fail(*args, **kwargs) -> None:
        nonlocal callback_calls
        del args, kwargs
        callback_calls += 1
        raise LookupError("sink failed")

    with pytest.raises(LookupError, match="sink failed"):
        model.decode_temporal(
            torch.zeros(1, 3, latent_t, 1, 1),
            temporal_chunk_callback=fail,
        )

    assert callback_calls == 1
    assert model.adaptive_decode_calls == model.num_chunks
    recovered_model = _FakeTemporalDecoder(api, latent_t)
    recovered = recovered_model.decode_temporal(torch.zeros(1, 3, latent_t, 1, 1))
    torch.testing.assert_close(recovered, recovered_model.expected, rtol=0, atol=0)


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
def test_terminal_callback_failure_is_deferred(api: type) -> None:
    latent_t = 35
    model = _FakeTemporalDecoder(api, latent_t)
    seen: list[tuple[int, int, int, bool]] = []

    def fail_final(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        del frames
        seen.append((chunk_index, total_chunks, frame_start, is_final))
        if is_final:
            raise LookupError("final sink failed")

    with pytest.raises(LookupError, match="final sink failed"):
        model.decode_temporal(
            torch.zeros(1, 3, latent_t, 1, 1),
            temporal_chunk_callback=fail_final,
        )

    assert seen == [(index, 7, index * 17, index == 6) for index in range(7)]
    assert model.adaptive_decode_calls == model.num_chunks


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
def test_decode_base_forwards_callback_without_exposing_mutation(api: type) -> None:
    latent_t = 37
    model = _FakeTemporalDecoder(api, latent_t)
    metadata: list[tuple[int, int, int, bool]] = []

    def mutate(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        metadata.append((chunk_index, total_chunks, frame_start, is_final))
        frames.fill_(1234)

    output = model.decode_base(
        torch.zeros(1, 3, latent_t, 1, 1),
        temporal_chunk_callback=mutate,
    )

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    assert metadata == [(index, 8, index * 17, index == 7) for index in range(8)]


@pytest.mark.parametrize("api", [FL2VAAutoencoderKL, Ref2VAAutoencoderKL])
def test_decode_base_rejects_unsupported_callback_modes(api: type) -> None:
    latent_t = 37
    latent = torch.zeros(1, 3, latent_t, 1, 1)
    model = _FakeTemporalDecoder(api, latent_t)

    with pytest.raises(ValueError, match="untrimmed 3D video"):
        model.decode_base(
            latent, frame_num=1, temporal_chunk_callback=lambda *args, **kwargs: None
        )
    with pytest.raises(ValueError, match="untrimmed 3D video"):
        model.decode_base(
            latent,
            process_image=True,
            temporal_chunk_callback=lambda *args, **kwargs: None,
        )

    model.training = True
    with pytest.raises(ValueError, match="inference-only"):
        model.decode_temporal(
            latent, temporal_chunk_callback=lambda *args, **kwargs: None
        )


def _distributed_callback_worker(
    rank: int,
    init_file: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        model = _FakeTemporalDecoder(
            FL2VAAutoencoderKL,
            37,
            collective=True,
        )

        def consume(*args, **kwargs) -> None:
            del args, kwargs
            if rank == 0:
                raise LookupError("distributed sink failed")

        try:
            model.decode_temporal(
                torch.zeros(1, 3, 37, 1, 1),
                temporal_chunk_callback=consume,
            )
        except BaseException as exc:  # noqa: BLE001
            failure_type = type(exc).__name__
        else:
            failure_type = "none"

        recovered_model = _FakeTemporalDecoder(
            FL2VAAutoencoderKL,
            37,
            collective=True,
        )
        recovered = recovered_model.decode_temporal(torch.zeros(1, 3, 37, 1, 1))
        recovery_probe = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(recovery_probe)
        result_queue.put(
            {
                "rank": rank,
                "failure_type": failure_type,
                "failed_decode_calls": model.adaptive_decode_calls,
                "recovered": torch.equal(recovered, recovered_model.expected),
                "recovery_probe": int(recovery_probe.item()),
            }
        )
    finally:
        dist.destroy_process_group()


def test_callback_failure_does_not_strand_spatial_parallel_peers(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_distributed_callback_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.join(timeout=1)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "two-rank temporal callback test deadlocked"
    assert [process.exitcode for process in processes] == [0, 0]

    results = sorted(
        [result_queue.get(timeout=5) for _ in range(2)],
        key=lambda result: result["rank"],
    )
    assert [result["failure_type"] for result in results] == [
        "LookupError",
        "none",
    ]
    assert [result["failed_decode_calls"] for result in results] == [7, 7]
    assert all(result["recovered"] for result in results)
    assert [result["recovery_probe"] for result in results] == [3, 3]


def test_callback_api_version_is_public() -> None:
    assert FL2VAAutoencoderKL.temporal_chunk_callback_api_version == 1
    assert Ref2VAAutoencoderKL.temporal_chunk_callback_api_version == 1
