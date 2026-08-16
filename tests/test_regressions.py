"""
MiniMax-H3 Regression Tests — Fixed harness
==============================================
Uses sys.path injection instead of spec_from_file_location so relative
imports inside the audio_vae package resolve correctly.

Run with:
    cd d:\\minimax\\MiniMax-H3
    python -m pytest tests/test_regressions.py -v --tb=short
"""

import json
import sys
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent


def _load_audio_vae_package(task: str):
    """
    Add the task directory to sys.path so relative imports work,
    then import the modules as a flat namespace.
    Returns the dac_audio_vae module for that task.
    """
    pkg_dir = str(PROJECT_ROOT / task / "audio_vae")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    # Force fresh load by removing any cached versions
    mods_to_reload = [
        "dac_alias_free_filter", "dac_alias_free_resample",
        "dac_alias_free_act", "dac_activations", "dac_utils",
        "dac_attn_proj", "dac_bigvgan", "dac_audio_vae",
    ]
    for m in mods_to_reload:
        sys.modules.pop(m, None)
    import dac_audio_vae
    return dac_audio_vae


def _load_filter_module(task: str):
    pkg_dir = str(PROJECT_ROOT / task / "audio_vae")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    sys.modules.pop("dac_alias_free_filter", None)
    import dac_alias_free_filter
    return dac_alias_free_filter


def _load_attn_proj(task: str):
    pkg_dir = str(PROJECT_ROOT / task / "audio_vae")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    sys.modules.pop("dac_attn_proj", None)
    import dac_attn_proj
    return dac_attn_proj


# ─── Bug 1 ────────────────────────────────────────────────────────────────────

class TestBug1KaiserSincFilter:

    @pytest.fixture(params=["FL2VA", "Ref2VA"])
    def fmod(self, request):
        return _load_filter_module(request.param)

    def test_cutoff_zero_no_nameerror(self, fmod):
        r = fmod.kaiser_sinc_filter1d(cutoff=0, half_width=0.6, kernel_size=12)
        assert r is not None

    def test_cutoff_zero_shape(self, fmod):
        r = fmod.kaiser_sinc_filter1d(cutoff=0, half_width=0.6, kernel_size=12)
        assert r.shape == (1, 1, 12)

    def test_cutoff_zero_all_zeros(self, fmod):
        r = fmod.kaiser_sinc_filter1d(cutoff=0, half_width=0.6, kernel_size=12)
        assert torch.all(r == 0)

    def test_cutoff_positive_shape(self, fmod):
        for ks in [8, 12, 24]:
            r = fmod.kaiser_sinc_filter1d(cutoff=0.3, half_width=0.3, kernel_size=ks)
            assert r.shape == (1, 1, ks)

    def test_cutoff_positive_normalized(self, fmod):
        r = fmod.kaiser_sinc_filter1d(cutoff=0.5, half_width=0.6, kernel_size=12)
        assert abs(r.sum().item() - 1.0) < 1e-5


# ─── Bug 2 ────────────────────────────────────────────────────────────────────

class TestBug2DuplicateSampleRate:

    def test_source_has_exactly_one_assignment(self):
        for task in ("FL2VA", "Ref2VA"):
            src = (PROJECT_ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
            count = src.count("self.sample_rate = sample_rate")
            assert count == 1, f"{task}: expected 1 assignment, found {count}"

    @pytest.mark.parametrize("task", ["FL2VA", "Ref2VA"])
    def test_sample_rate_attribute_correct(self, task):
        vae_mod = _load_audio_vae_package(task)
        for sr in (16000, 32000):
            vae = vae_mod.DacAudioVAE(sample_rate=sr)
            assert vae.sample_rate == sr


# ─── Bug 3 ────────────────────────────────────────────────────────────────────

class TestBug3InitWeightsConsistency:

    def test_uses_trunc_normal_std02(self):
        for task in ("FL2VA", "Ref2VA"):
            src = (PROJECT_ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
            assert "trunc_normal_" in src
            assert "std=0.02" in src
            assert "m.bias" in src

    def test_apply_called_once_in_live_code(self):
        for task in ("FL2VA", "Ref2VA"):
            src = (PROJECT_ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
            # Count only real code lines (not comment or docstring lines)
            code_apply = sum(
                1 for l in src.splitlines()
                if "self.apply(init_weights)" in l
                and not l.strip().startswith("#")
                and not l.strip().startswith("`")
                and not l.strip().startswith("\"\"\"")
            )
            assert code_apply == 1, f"{task}: {code_apply} apply calls in live code"

    @pytest.mark.parametrize("task", ["FL2VA", "Ref2VA"])
    def test_conv_biases_zeroed_after_build(self, task):
        vae_mod = _load_audio_vae_package(task)
        vae = vae_mod.DacAudioVAE(sample_rate=32000)
        checked = 0
        for name, param in vae.named_parameters():
            if "bias" in name and param is not None and param.numel() > 0:
                assert torch.all(param.data == 0), f"{task}: {name} not zeroed"
                checked += 1
        assert checked > 0, "No biases found to check"


# ─── Bug 4 ────────────────────────────────────────────────────────────────────

class TestBug4CausalAttentionShape:

    @pytest.fixture(params=["FL2VA", "Ref2VA"])
    def amod(self, request):
        return _load_attn_proj(request.param)

    def _fwd(self, amod, in_d, out_d, nh, B=2, N=8):
        ca = amod.CausalAttention(in_d, out_d, nh)
        ca.eval()
        with torch.no_grad():
            return ca(torch.randn(B, N, in_d)).shape

    def test_in_eq_out(self, amod):
        assert self._fwd(amod, 64, 64, 8) == (2, 8, 64)

    def test_in_less_than_out(self, amod):
        assert self._fwd(amod, 32, 64, 8) == (2, 8, 64)

    def test_in_greater_than_out(self, amod):
        """This was the broken case — Bug 4."""
        assert self._fwd(amod, 64, 32, 8) == (2, 8, 32)

    def test_large_batch(self, amod):
        assert self._fwd(amod, 64, 32, 8, B=4, N=16) == (4, 16, 32)

    def test_attn_projection_end_to_end(self, amod):
        proj = amod.AttnProjection(in_dim=64, out_dim=32, num_heads=8)
        proj.eval()
        with torch.no_grad():
            out = proj(torch.randn(2, 10, 64))
        assert out.shape == (2, 10, 32)


# ─── Bug 5 ────────────────────────────────────────────────────────────────────

class TestBug5EncoderBlockPadding:

    def test_source_formula_correct(self):
        for task in ("FL2VA", "Ref2VA"):
            src = (PROJECT_ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
            # New formula present
            assert "padding=stride // 2" in src, f"{task}: new formula missing"
            # Old formula must NOT appear in live code (only in comments)
            old_in_live = any(
                "padding=math.ceil(stride" in l and not l.strip().startswith("#")
                for l in src.splitlines()
            )
            assert not old_in_live, f"{task}: old math.ceil formula still in live code"

    @pytest.mark.parametrize("stride", [1, 2, 3, 4, 5, 8])
    def test_output_length(self, stride):
        vae_mod = _load_audio_vae_package("FL2VA")
        dim = 32
        block = vae_mod.EncoderBlock(dim=dim, stride=stride)
        block.eval()
        T = stride * 20
        with torch.no_grad():
            out = block(torch.randn(1, dim // 2, T))
        assert out.shape[1] == dim, f"stride={stride}: channels {out.shape[1]}"
        assert out.shape[2] == T // stride, f"stride={stride}: T {out.shape[2]} != {T//stride}"


# ─── Bug 6 ────────────────────────────────────────────────────────────────────

class TestBug6ResidualUnit:

    @pytest.fixture
    def ru_mod(self):
        return _load_audio_vae_package("FL2VA")

    def test_source_uses_explicit_crop(self):
        for task in ("FL2VA", "Ref2VA"):
            src = (PROJECT_ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
            assert "pad_right = diff - pad_left" in src

    @pytest.mark.parametrize("T", [30, 31, 50, 51, 99, 100, 101])
    def test_no_crash_various_lengths(self, ru_mod, T):
        unit = ru_mod.ResidualUnit(dim=16, dilation=1)
        unit.eval()
        with torch.no_grad():
            _ = unit(torch.randn(2, 16, T))

    def test_dilation3_no_crash(self, ru_mod):
        unit = ru_mod.ResidualUnit(dim=16, dilation=3)
        unit.eval()
        for T in [31, 33, 47, 51]:
            with torch.no_grad():
                _ = unit(torch.randn(1, 16, T))

    def test_output_is_3d(self, ru_mod):
        unit = ru_mod.ResidualUnit(dim=16, dilation=1)
        unit.eval()
        with torch.no_grad():
            out = unit(torch.randn(2, 16, 51))
        assert out.ndim == 3


# ─── Bug 7 ────────────────────────────────────────────────────────────────────

class TestBug7TransformerConfig:

    @pytest.mark.parametrize("task", ["FL2VA", "Ref2VA"])
    def test_ffn_both_keys(self, task):
        cfg = json.loads((PROJECT_ROOT / task / "transformer" / "config.json").read_text())
        assert "ffn_hidden_size" in cfg and "ffn_dim" in cfg
        assert cfg["ffn_hidden_size"] == cfg["ffn_dim"]

    @pytest.mark.parametrize("task", ["FL2VA", "Ref2VA"])
    def test_rope_both_keys(self, task):
        cfg = json.loads((PROJECT_ROOT / task / "transformer" / "config.json").read_text())
        assert "rope_inv_freq_len" in cfg and "rope_freq_dim" in cfg
        assert cfg["rope_inv_freq_len"] == cfg["rope_freq_dim"]

    @pytest.mark.parametrize("task", ["FL2VA", "Ref2VA"])
    def test_refiner_both_keys(self, task):
        cfg = json.loads((PROJECT_ROOT / task / "transformer" / "config.json").read_text())
        assert "token_refiner_num_layers" in cfg and "num_refiner_layers" in cfg
        assert cfg["token_refiner_num_layers"] == cfg["num_refiner_layers"]


# ─── Bug 8 ────────────────────────────────────────────────────────────────────

class TestBug8DiffusersVersion:

    def test_task_indexes_match(self):
        fl = json.loads((PROJECT_ROOT / "FL2VA" / "model_index.json").read_text())["_diffusers_version"]
        ref = json.loads((PROJECT_ROOT / "Ref2VA" / "model_index.json").read_text())["_diffusers_version"]
        assert fl == ref

    def test_root_indexes_match(self):
        root = json.loads((PROJECT_ROOT / "model_index.json").read_text())["_diffusers_version"]
        mod = json.loads((PROJECT_ROOT / "modular_model_index.json").read_text())["_diffusers_version"]
        assert root == mod


# ─── Bug 9 ────────────────────────────────────────────────────────────────────

class TestBug9PyYAML:
    def test_listed(self):
        req = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        assert "PyYAML" in req


# ─── Bug 10 ───────────────────────────────────────────────────────────────────

class TestBug10CRLF:
    def test_no_crlf(self):
        bad = [f.name for f in (PROJECT_ROOT / "scripts" / "readme").glob("*.sh")
               if b"\r\n" in f.read_bytes()]
        assert bad == [], f"CRLF found in: {bad}"


# ─── Bug 11 ───────────────────────────────────────────────────────────────────

class TestBug11CurlFlag:
    @pytest.mark.parametrize("script", [
        "full-2k-t2va-h3-base.sh",
        "full-2k-i2va-h3-base.sh",
        "full-2k-ref2va-h3-base.sh",
    ])
    def test_flag_present(self, script):
        txt = (PROJECT_ROOT / "scripts" / "readme" / script).read_text(encoding="utf-8")
        assert "--fail-with-body" in txt


# ─── Bug 12 ───────────────────────────────────────────────────────────────────

class TestBug12SampleRate:
    def test_both_keys_present_and_equal(self):
        cfg = json.loads((PROJECT_ROOT / "audio_vae" / "config.json").read_text())
        assert "sample_rate" in cfg
        assert "sampling_rate" in cfg
        assert cfg["sample_rate"] == cfg["sampling_rate"]

    def test_fl2va_config_has_sample_rate(self):
        cfg = json.loads((PROJECT_ROOT / "FL2VA" / "audio_vae" / "config.json").read_text())
        assert "sample_rate" in cfg


# ─── Bug 13 ───────────────────────────────────────────────────────────────────

class TestBug13README:
    def test_download_steps_present(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
        assert "Step 1" in readme and "Step 2" in readme and "Step 3" in readme
        assert "hf download MiniMaxAI/MiniMax-H3" in readme


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
