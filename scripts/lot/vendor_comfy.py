#!/usr/bin/env python3
"""Copy the LoT core into a ComfyUI node pack as a self-contained package.

    python3 scripts/lot/vendor_comfy.py /media/2TB/ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Image-Lane

Writes ``<pack>/lot_core/`` with the six modules the nodes need, rewriting their
top-level imports (``from layout import ...``) to package-relative ones so they
cannot collide with another node pack's ``layout`` or ``flow``. ``scripts/lot``
stays the source of truth: edit there, then re-run this.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MODULES = ("layout", "flow", "adapter", "h3", "h3_positions", "h3_splice", "layout_sources")
HERE = Path(__file__).resolve().parent


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]) / "lot_core"
    target.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    pattern = re.compile(r"^from (" + "|".join(MODULES) + r") import ", re.MULTILINE)
    for name in MODULES:
        text = (HERE / f"{name}.py").read_text()
        text = pattern.sub(r"from .\1 import ", text)
        header = (f"# Vendored from johndpope/MiniMax-H3 scripts/lot/{name}.py at {commit}.\n"
                  f"# Do not edit here: change scripts/lot and re-run scripts/lot/vendor_comfy.py.\n")
        (target / f"{name}.py").write_text(header + text)
    (target / "__init__.py").write_text(
        f'"""Level-of-Token core for MiniMax H3, vendored from johndpope/MiniMax-H3 at {commit}."""\n')
    print(f"vendored {len(MODULES)} modules into {target} from {commit}")


if __name__ == "__main__":
    main()
