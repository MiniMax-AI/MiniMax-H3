"""One H3 GPU job at a time. The guard refuses; it never kills anything."""

from __future__ import annotations

import os
import subprocess

GPU_SCRIPTS = ("train_synth.py", "parity_h3.py", "procrustes_h3.py", "make_pairs_h3.py", "render_h3.py", "time_decode_h3.py", "train_h3.py", "build_cache_h3.py", "time_lot_h3.py")


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
