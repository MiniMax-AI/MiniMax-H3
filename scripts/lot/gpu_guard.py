"""One H3 GPU job at a time. The guard refuses; it never kills anything."""

from __future__ import annotations

import os
import subprocess

GPU_SCRIPTS = ("train_synth.py", "parity_h3.py", "procrustes_h3.py", "make_pairs_h3.py", "render_h3.py", "time_decode_h3.py", "train_h3.py", "build_cache_h3.py", "time_lot_h3.py", "mixed_res_h3.py", "two_track_h3.py", "train_tt.py", "build_pairs_tt.py")


def other_gpu_job() -> str | None:
    """Command line of another LoT GPU script on this machine, or ``None``."""
    listing = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    me = os.getpid()
    for line in listing.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv = cmd.split()
        if not pid.isdigit() or int(pid) == me or not argv:
            continue
        # argv[0] must be the interpreter: a shell whose command string names a
        # script is a wrapper, not the job.
        if not os.path.basename(argv[0]).startswith("python"):
            continue
        if any(os.path.basename(arg) in GPU_SCRIPTS for arg in argv[1:]):
            return cmd
    return None


def refuse_if_busy(name: str) -> None:
    job = other_gpu_job()
    if job is not None:
        raise SystemExit(f"{name}: another LoT GPU job is running ({job}). It will not be killed.")


def card_gb() -> float:
    """Free memory on the current CUDA device, GB (other processes count against it)."""
    import torch

    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def auto_swap(value: str) -> int:
    """``--swap auto``: DiT blocks to stream from CPU, from the card's free memory.

    Measured with ``train_tt.py lora --check`` (two-track LoRA, 1344x768 pairs): 8 blocks peak at
    19.6 GB on an RTX 5090; 24 blocks peak at 13.3 GB on an RTX PRO 4000 with 6 GB already taken.
    0 (all resident) on 64 GB+ free, e.g. an RTX PRO 6000 (96 GB), is not measured yet."""
    if value != "auto":
        return int(value)
    gb = card_gb()
    return 0 if gb >= 64 else 8 if gb >= 28 else 16 if gb >= 21 else 24
