#!/usr/bin/env python3
"""Publish a train_h3 run to the Hugging Face Hub (private by default).

    python3 scripts/lot/upload_hf.py --run scripts/lot/runs/train_h3_nikki \
        --repo johndpope/MiniMax-H3-LoT --title "Nikki talk + pose" [--public]

Stages ``adapter.safetensors`` (the LoT adapter: extent heads, shape MLP and the
fitted extent bank, converted from ``adapter.pt``), ``lora.safetensors`` (rank-16
LoRA on the DiT's qkv / out / fc1 / fc2), ``train_log.jsonl`` and a model card
with the held-out eval curve, then uploads the folder. Nothing is uploaded
until staging succeeds. The repo is created private unless ``--public``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

LICENSE_LINK = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
CODE = "https://github.com/johndpope/MiniMax-H3/tree/main/scripts/lot"
ISSUE = "https://github.com/johndpope/MiniMax-H3/issues/1"
HELP = "https://github.com/scrya-com/lot-diffusion/discussions/1"
# Section III.4 of the H3 Community License: every distribution carries this NOTICE text.
NOTICE = ("MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, "
          "Copyright © 2026 MiniMax. All Rights Reserved.\n\n"
          "The files in this repository (adapter.safetensors, lora.safetensors) are modifications derived "
          "from MiniMax H3 weights: a Level-of-Token adapter and a rank-16 LoRA trained on the MiniMax H3 "
          "FL2VA DiT. They are not released or endorsed by MiniMax.\n")


def warnings(repo: str) -> str:
    """Help-wanted, license/territory and data notes at the top of every card."""
    base = f"https://huggingface.co/{repo}/blob/main"
    return f"""> [!WARNING]
> # ⚠️ THIS MODEL NEEDS MORE TRAINING — HELP WANTED
> **These are research checkpoints, not a finished model.** LoT output is still visibly worse than dense MiniMax H3: coarse regions are soft, faces in talking-head shots distort, and the LoRA can bend bodies even with LoT off. If you have GPU time, data, or ideas, join the discussion: **[help-wanted discussion]({HELP})**.

> [!CAUTION]
> **License and territory.** Derived from MiniMax H3 under the [MiniMax H3 Community License Agreement]({base}/LICENSE) (see also [`NOTICE`]({base}/NOTICE)). That license allows use and redistribution **only outside its Excluded Territories: the European Union, the United Kingdom, the Republic of Korea and the United States of America.** Do not download or use these files there. Commercial use above US$20M yearly revenue needs MiniMax's written authorization. The [Acceptable Use Policy]({base}/LICENSE) applies, including: no impersonation without consent, and machine-generated output must be disclosed as such.

> [!NOTE]
> **Training data.** Check the Training section below for what this run saw. Do not use these weights to impersonate anyone.

**Train it yourself / help improve it:** [cheatsheet]({CODE}/CHEATSHEET.md) · [train from your own mp4s]({CODE}/README.md#training-h3-for-lot-from-your-own-mp4s) · [results and renders (issue 1)]({ISSUE})
"""


def eval_table(log_path: Path) -> str:
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    evals = [row for row in rows if "eval" in row]
    if not evals:
        return "_No held-out evals recorded._"
    keys = list(evals[0]["eval"].keys())
    lines = ["| step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for row in evals:
        lines.append(f"| {row['step']} | " + " | ".join(f"{row['eval'][k]:.4f}" for k in keys) + " |")
    return "\n".join(lines)


def model_card(repo: str, title: str, settings: str, table: str, subfolder: str = "") -> str:
    run = f"runs/lot_hf/{subfolder}" if subfolder else "runs/lot_hf"
    return f"""---
license: other
license_name: minimax-h3-community-license
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- level-of-token
- lora
- video-generation
- research
---

# MiniMax-H3 Level-of-Token adapter: {title}

{warnings(repo)}

Research weights for running **MiniMax-H3**'s DiT on a **Level-of-Token (LoT)** layout: fewer, larger tokens where detail is low, with the full-resolution velocity recovered afterwards ([Nakayama et al., arXiv 2610.05816](https://arxiv.org/abs/2610.05816)). The VAE latent stays full size; only the transformer sequence shrinks. On one 24 GB card a 124-frame 768×1344 DiT forward (37 latent frames) measured 52.3 s dense vs 18.3 s LoT (2.85×, `center` layout). The VAE decode is unchanged.

**Status: research.** Coarse regions are still softer than dense H3. Progress, renders and caveats are tracked in [the issue]({ISSUE}).

## Files

| File | What |
|---|---|
| `adapter.safetensors` | LoT adapter: per-extent input/output heads, shape MLP, and the Procrustes extent bank (`A`, `s`) it was trained with |
| `lora.safetensors` | Rank-16 LoRA on `blocks.*.attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2` (200 modules) of the FL2VA DiT |
| `train_log.jsonl` | Per-step loss and held-out evals |

## Use

Needs the code in [`scripts/lot`]({CODE}), a [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) checkout (`MUSUBI_SRC`), and the pruned int8 FL2VA DiT (`hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`).

```bash
hf download {repo} --local-dir runs/lot_hf
python3 scripts/lot/time_lot_h3.py --shape clip --trained {run}      # timing per layout
python3 scripts/lot/render_h3.py --trained {run} \\
    --layout bands --variants dense,dense_lora,lot_trained   # needs a cache of encoded prompts
```

ComfyUI (stills and video): [ComfyUI-MiniMax-H3-Image-Lane](https://github.com/johndpope/ComfyUI-MiniMax-H3-Image-Lane). Quick start and training from your own mp4s: [cheatsheet]({CODE}/CHEATSHEET.md).

## Training

{settings}

Loss: eq. 17 with H3's `x0 − ε` head (velocity MSE in y-space), plus a distillation term that pulls LoT's clean estimate toward the frozen dense base (LoRA off) from the same noisy input.

## Held-out eval

`dense` / `uniform2` / `mosaic` are the data loss on those layouts. `gap_*` is the weighted distance from LoT to the frozen dense teacher. Lower is better.

{table}

## License

Derived from MiniMax H3 weights; use is governed by the [MiniMax H3 Community License Agreement]({LICENSE_LINK}).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="train_h3 --out directory")
    parser.add_argument("--repo", required=True, help="e.g. johndpope/MiniMax-H3-LoT")
    parser.add_argument("--title", default="LoT adapter")
    parser.add_argument("--settings", default="See `train_log.jsonl`.", help="markdown for the Training section")
    parser.add_argument("--subfolder", default="", help="upload into this folder of the repo")
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()

    adapter_pt = args.run / "adapter.pt"
    lora = args.run / "lora.safetensors"
    log = args.run / "log.jsonl"
    for path in (adapter_pt, lora, log):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    from safetensors.torch import save_file
    from huggingface_hub import HfApi

    stage = args.run / "hf_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    state = torch.load(adapter_pt, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(stage / "adapter.safetensors"),
              metadata={"format": "pt", "source": "train_h3 adapter.pt"})
    shutil.copy2(lora, stage / "lora.safetensors")
    shutil.copy2(log, stage / "train_log.jsonl")
    (stage / "README.md").write_text(model_card(args.repo, args.title, args.settings, eval_table(log), args.subfolder))

    api = HfApi()
    api.create_repo(args.repo, private=not args.public, exist_ok=True)
    # The license copy and NOTICE are repo-level (Section III.1 and III.4); upload them every time.
    license_path = stage / "LICENSE"
    from huggingface_hub import hf_hub_download
    shutil.copy2(hf_hub_download("MiniMaxAI/MiniMax-H3", "LICENSE"), license_path)
    (stage / "NOTICE").write_text(NOTICE)
    for name in ("LICENSE", "NOTICE"):
        api.upload_file(path_or_fileobj=str(stage / name), path_in_repo=name, repo_id=args.repo,
                        commit_message=f"{name} (MiniMax H3 Community License, Section III)")
        (stage / name).unlink()
    api.upload_folder(repo_id=args.repo, folder_path=str(stage), path_in_repo=args.subfolder or None,
                      commit_message=f"LoT adapter + LoRA: {args.title}")
    sizes = {p.name: p.stat().st_size for p in stage.iterdir()}
    print(f"LOT_H3 kind=hf_upload repo={args.repo} private={not args.public} "
          f"subfolder={args.subfolder or '/'} files={sizes} ok=1", flush=True)


if __name__ == "__main__":
    main()
