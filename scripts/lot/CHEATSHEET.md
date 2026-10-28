# Level-of-Token on H3: cheatsheet

> **Maintained at [scrya-com/lot-diffusion](https://github.com/scrya-com/lot-diffusion)** (same code, with the musubi-tuner H3 runtime bundled in `third_party/`).

> **⚠️ Research checkpoints: LoT output is still worse than dense H3, and it needs more training. Help wanted: [help-wanted discussion](https://github.com/scrya-com/lot-diffusion/discussions/1).** Weights: [HF johndpope/MiniMax-H3-LoT](https://huggingface.co/johndpope/MiniMax-H3-LoT) (MiniMax H3 Community License; not for use in the EU, UK, South Korea or USA).

LoT gives the H3 DiT fewer, larger tokens where detail is low, then recovers the full-resolution velocity (Nakayama et al., [arXiv 2610.05816](https://arxiv.org/abs/2610.05816)). The VAE latent stays full size; only the transformer sequence shrinks. Full walkthrough: [README → Training H3 for LoT from your own mp4s](README.md#training-h3-for-lot-from-your-own-mp4s). Results and history: [issue 1](https://github.com/johndpope/MiniMax-H3/issues/1).

Run everything from the repo root. One GPU job at a time; the scripts refuse to start beside another one and never kill it.

## Quick start: download → test → time → train

**Path A: ComfyUI (no Python setup).** Install [ComfyUI-MiniMax-H3-Image-Lane](https://github.com/johndpope/ComfyUI-MiniMax-H3-Image-Lane) and download the weights:

```bash
cd ComfyUI/custom_nodes && git clone https://github.com/johndpope/ComfyUI-MiniMax-H3-Image-Lane.git && cd ../..
hf download johndpope/MiniMax-H3-LoT run4_nikki_distill/adapter.safetensors run4_nikki_distill/lora.safetensors --local-dir /tmp/lot
mkdir -p ComfyUI/models/lot
cp /tmp/lot/run4_nikki_distill/adapter.safetensors ComfyUI/models/lot/run4_nikki_distill_adapter.safetensors
cp /tmp/lot/run4_nikki_distill/lora.safetensors    ComfyUI/models/loras/minimax_h3_lot_run4_nikki_lora.safetensors
```

Load `workflows/lot_video_api.json` (clip) or `workflows/t1_lot_image_api.json` (still). Each samples the same seed twice, LoT and dense. **Validate the timing** by comparing the console's it/s for the two sampler runs (LoT runs first). Expect about 2.6× on the 22-frame clip and about 1.35× on a still.

**Path B: Python (this repo + [musubi-tuner](https://github.com/kohya-ss/musubi-tuner)).** musubi-tuner supplies H3 itself (int8 loader, block swap, LoRA, VAE), used unmodified. Point the scripts at your files (defaults are this machine; see `lot_paths.py`):

```bash
git clone https://github.com/kohya-ss/musubi-tuner && pip install -e musubi-tuner
export MUSUBI_SRC=$PWD/musubi-tuner/src
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models
export LOT_H3_CHECKPOINT=$PWD/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
export LOT_H3_VAE=/path/to/minimax_h3_video_vae_fp16.safetensors
hf download johndpope/MiniMax-H3-LoT --local-dir runs/lot_hf

python3 scripts/lot/test_lot.py                                                   # 1. CPU tests -> ok 28
python3 scripts/lot/parity_h3.py                                                  # 2. GPU: LoT 1x1 == dense -> ok=1
python3 scripts/lot/time_lot_h3.py --shape still --trained runs/lot_hf/run4_nikki_distill            # 3. timing
python3 scripts/lot/time_lot_h3.py --shape clip  --trained runs/lot_hf/run4_nikki_distill            #    all resident, 21 GB
python3 scripts/lot/render_h3.py --trained runs/lot_hf/run4_nikki_distill --swap 4 \
    --layout bands --variants dense,dense_lora,lot_trained                       # 4. look (held-out prompts)
```

Step 4 needs prompts already encoded by Qwen: a cache with `_te` files (`LOT_H3_STILLS_CACHE`, or `--prompt-cache DIR`, built in sections 1–2). Without one, use Path A for pictures; steps 1–3 need no text encoder.

`time_lot_h3.py` prints one `LOT_H3 kind=lot_timing` line per layout with `ms` and `ratio` against dense, on random inputs. Measured here (RTX PRO 4000 Blackwell, 24 GB, run-4 weights, all blocks resident):

| Shape | dense | `center` | `bands` | `uniform2` |
|---|---|---|---|---|
| still 768×1152 | 657 ms | 306 ms (**2.15×**) | 395 ms (1.67×) | 265 ms (2.48×) |
| clip 512×768×22 | 2,154 ms | 844 ms (**2.55×**) | 1,124 ms (1.92×) | 665 ms (3.24×) |

Both fit fully resident (peak 20.8 / 21.1 GB). With `--swap 4` the same clip only shows 1.44–1.51×: the streamed blocks, not attention, set the time. The long clip (`--shape long --swap 48`, 124 frames at 768×1344) streams 48 blocks and still gains: dense 52.3 s · `center` 18.3 s (**2.85×**) · `bands` 20.4 s (2.57×), peak 13.3 GB.

**Train:** sections 1–4 below. Best checkpoint so far: **`run4_nikki_distill`** (see [Runs so far](#runs-so-far)). Publish with `upload_hf.py`.

## 0. Does it work here?

```bash
python3 scripts/lot/test_lot.py            # CPU, no weights -> "ok 28"
python3 scripts/lot/parity_h3.py           # GPU: LoT 1x1 == dense H3 -> rel_rms=0 ok=1
```

## 1. Clips (any mp4)

```bash
W=640; H=384   # portrait footage: W=384 H=640
ffmpeg -v error -y -ss 0 -i talk.mp4 \
  -vf "fps=24,scale=$W:$H:force_original_aspect_ratio=increase,crop=$W:$H" \
  -frames:v 22 -an -c:v libx264 -crf 16 -pix_fmt yuv420p /data/lot_clips/talk_000_mute.mp4
echo "a woman talking to camera in a kitchen" > /data/lot_clips/talk_000_mute.txt
```

Spec: 24 fps · 22 frames (17n+5 grid) · sides ×32 · no audio needed · same-stem `.txt` caption. Stills (`.png`/`.jpg`/`.webp` + `.txt`) can share the folder.

## 2. Cache

```bash
python3 scripts/lot/build_cache_h3.py media --src /data/lot_clips --out /data/lot_clips/cache --size 640 384
python3 scripts/lot/build_cache_h3.py text  --out /data/lot_clips/cache     # loads Qwen; run it alone on the GPU
```

`media` encodes clips and stills with the H3 video VAE (first `--frames 22`, cover-resized and centre-cropped to `--size`); `text` adds the `_te` files, resumable.

**Already have H3 latents (`z.pt`) and prompts?** Skip the VAE and use `build_cache_h3.py`. It built the Nikki cache (50 talking-head + 466 pose clips, first 7 latent frames, `latent_7x48x32`):

```bash
python3 scripts/lot/build_cache_h3.py latents   # CPU: z.pt -> cache + captions.json
python3 scripts/lot/build_cache_h3.py text      # GPU, alone: Qwen3-VL captions, resumable (~9.5 s each)
```

## 3. Extent bank (once)

```bash
P=scripts/lot/runs/pairs_nikki_scrya_ref2va_768x1152    # existing Nikki pairs, or build your own:
# python3 scripts/lot/make_pairs_h3.py --src DIR_WITH_<clip>/comfy_t00.png
python3 scripts/lot/procrustes_h3.py --pairs $P --smoke  # every line ok=1
```

## 4. Train

```bash
python3 scripts/lot/train_h3.py --pairs $P --cache /data/lot_clips/cache --swap 4 --shift 3 --distill 1 --check

LOT_H3_TRAIN=1 python3 scripts/lot/train_h3.py --pairs $P \
  --cache /data/lot_clips/cache --cache /media/2TB/lora-data/h3_lot/cache_iso3d \
  --swap 4 --shift 3 --distill 1 --steps 1500 --save-every 250 --holdout 12 \
  --init scripts/lot/runs/train_h3_nikki --out scripts/lot/runs/my_run
```

`--distill 1` is the recipe that first gave visible gains (runs 3 and 4). `--init` takes a run dir with `adapter.pt` or `adapter.safetensors` plus `lora.safetensors`, so an HF download works as a warm start. Publish with `python3 scripts/lot/upload_hf.py --run scripts/lot/runs/my_run --repo you/name --subfolder my_run`; it adds the license, `NOTICE` and warnings.

Long runs: start detached with `setsid nohup ... > /tmp/my_run.log 2>&1 &`. Then watch:

```bash
grep -E "train_eval|kind=train step" /tmp/my_run.log | tail -3             # progress, s/step
tail -f scripts/lot/runs/my_run/log.jsonl | grep --line-buffered '"eval"'  # held-out evals
```

## 5. Look

```bash
python3 scripts/lot/render_h3.py --pairs $P --trained scripts/lot/runs/my_run --swap 4 \
  --layout bands --variants dense,dense_lora,lot_fit,lot_trained   # -> runs/render/grid.png
# held-out prompts from another cache, chosen items (use the --holdout the run trained with):
python3 scripts/lot/render_h3.py --trained scripts/lot/runs/my_run --prompt-cache scripts/lot/runs/cache_nikki \
  --holdout 12 --pick 7,3 --prompts 2 --swap 4 --layout bands --variants dense,dense_lora,lot_trained
python3 scripts/lot/time_lot_h3.py --shape long --swap 48   # 124-frame 768x1344 DiT timing, dense vs LoT
python3 scripts/lot/time_decode_h3.py    # VAE decode (LoT does not speed this up)
```

## ComfyUI (stills and video)

The LoT nodes live in [ComfyUI-MiniMax-H3-Image-Lane](https://github.com/johndpope/ComfyUI-MiniMax-H3-Image-Lane): **MiniMax H3 LoT Apply** (layouts `center` / `bands` / `uniform2` / `uniform4` / `dense`) and **MiniMax H3 LoT Unscale Latent**, which is required before VAE Decode. Workflows (API format; they also ship in the pack's `workflows/`), each a LoT branch plus a dense A/B branch from one seed:

- [`comfyui/h3_lot_image_api.json`](../../comfyui/h3_lot_image_api.json): 768×1152 still, image VAE.
- [`comfyui/h3_lot_video_api.json`](../../comfyui/h3_lot_video_api.json): 512×768, 22-frame clip from the stock **MiniMax H3 Image to Video** node (t2va; add `first_frame` for i2va), video VAE, animated WebP.

LoT Apply works for any target length: the layout repeats per latent frame, and keyframe and reference rows stay dense.

**Content-aware layouts** (the paper's layout sources), each feeding LoT Apply's `layout_override`:

| Node | Fine tokens go | Use for |
|---|---|---|
| LoT Layout from **Boxes** | inside normalised `x0,y0,x1,y1` boxes (secondary boxes → 2×2) | faces and subjects in portraits and talking heads |
| LoT Layout from **Mask** | on a MASK, per frame when it is a batch (tracked object) | segmented subjects |
| LoT Layout from **Depth** | near a focus depth (Depth Anything map) | depth-of-field shots |
| LoT Layout from **Texture** | where an image is busy (i2v first frame, or a quick dense preview) | scenery, two-pass t2v |
| LoT Layout **Preview** | draws the layout and logs the compression | checking a layout before sampling |

[`comfyui/h3_lot_boxes_api.json`](../../comfyui/h3_lot_boxes_api.json): face box 1×1, shoulders 2×2, rest 4×4 on the held-out talk prompt. **It came out nearly identical to dense** (576/864 tokens, 1.5×), where the fixed `center` layout blurred and blocked the face. Tighter boxes give more speed for some quality.

```bash
hf download johndpope/MiniMax-H3-LoT run4_nikki_distill/adapter.safetensors run4_nikki_distill/lora.safetensors --local-dir /tmp/lot
cp /tmp/lot/run4_nikki_distill/adapter.safetensors ComfyUI/models/lot/run4_nikki_distill_adapter.safetensors
cp /tmp/lot/run4_nikki_distill/lora.safetensors    ComfyUI/models/loras/minimax_h3_lot_run4_nikki_lora.safetensors
```

Inside ComfyUI, a 1×1 `dense` layout reproduces the stock forward exactly (relative error 0 with ComfyUI's bf16 head weights). `center` ran at 5.6 vs 4.1 it/s on a 768×1152 still (1.35×) and **3.45 vs 1.33 it/s on a 22-frame 512×768 clip (2.6×)**. The output is visibly softer than dense: **research weights, more training wanted** ([help-wanted discussion](https://github.com/scrya-com/lot-diffusion/discussions/1)).

## Flags that matter

| Flag | Default | What it does |
|---|---|---|
| `LOT_H3_TRAIN=1` | unset | Required to train. `--check` runs without it and writes nothing. |
| `--cache DIR` | `cache_iso3d` | Cache of latents + text from `build_cache_h3.py`. Repeat to mix clips and stills. |
| `--swap N` | 4 | Blocks streamed from CPU (H2D only, with activation checkpointing). 4 fits 22-frame 512×768 clips with `--distill 1` (22.3 GB peak). |
| `--shift` | 12 | Noise density for LoT steps. **3** trains low σ, where detail forms. |
| `--dense-shift` | 12 | Noise density for the 20% dense anchor steps (H3's own). |
| `--init DIR` | none | Warm-start `adapter.pt` + `lora.safetensors`, fresh optimizer. |
| `--distill W` | 0 | Pull LoT's clean estimate toward frozen dense H3 (LoRA off) from the same `x_t`. Costs one no-grad dense forward per LoT step. Evals add `gap_*`. |
| `--data-weight` | 1.0 | Scale of the eq. 17 data term (0 = distillation only). |
| `--dense-p` | 0.2 | Share of steps on the plain dense layout. |
| `--rank` | 16 | LoRA rank on qkv / out / fc1 / fc2 × 50 blocks (200 modules). |
| `--max-latent-hw H W` | none | Random spatial crop cap (latent px, multiples of 8) for clips too big to train whole. |
| `--holdout N` | 24 | Held-out items per cache. Pass the same value to `render_h3.py --holdout`. |
| `--layout-mix` | `uniform:0.1,mosaic:0.15,box:0.2,focus:0.2,texture:0.15` | Non-dense layout families per step (dense is `--dense-p`). `box` / `focus` / `texture` change per latent frame like the paper's mask, depth and texture layouts. `mosaic:1` reproduces runs 1–4. Evals add a fixed `focus` layout and `gap_focus`. |

## Measured on one 24 GB card (RTX PRO 4000 Blackwell)

| Thing | Number |
|---|---|
| DiT forward, 37 latent frames 768×1344 (48 blocks streamed), `time_lot_h3.py --shape long` | dense 52.3 s · `center` 18.3 s (**2.85×**) · `bands` 20.4 s (2.57×); peak 13.3 GB |
| DiT forward, 768×1152 still, all resident, run-4 adapter + LoRA | dense 0.81 s · bands 0.47 s (**1.75×**) · all-2×2 0.36 s (**2.29×**); LoRA costs ~1% |
| ComfyUI sampler, 768×1152 still, `center` (300/864 tokens) | 5.6 it/s vs dense 4.1 it/s (1.35×; Comfy's fused kernels make dense steps fast) |
| ComfyUI sampler, 512×768 22-frame clip, `center` | 3.45 it/s vs dense 1.33 it/s (**2.6×**) |
| `time_lot_h3.py`, run-4 weights, all resident (musubi-tuner runtime) | still: center 2.15× · bands 1.67× · uniform2 2.48×; clip 512×768×22: center **2.55×** · bands 1.92× · uniform2 3.24× (peak 21.1 GB) |
| 20-step 37-frame clip, end to end (sum of measured parts) | dense ~1,110 s · LoT ~446 s · ~2.5× incl. the 31.8 s decode |
| VAE decode, 37 latent frames | 31.8 s, same with or without LoT |
| Train step | runs 1–4: stills ~2.7 s (swap 4) · 22-frame clips 13–17 s (swap 8). musubi-tuner runtime `--check`: 512×768 clips 10–14 s with the distill teacher (swap 4) |
| Training peak | stills 20.5 GB · clips 20.0 GB |

## Runs so far

| Run | Data | Recipe | Result |
|---|---|---|---|
| 1 | 777 isometric stills | shift 12, data loss | Removed the frozen model's streaks; coarse regions soft. Dense loss −43% was mostly style learning. |
| 2 | same | `--shift 3` from run 1 | LoT loss −5%, renders barely changed |
| 3 | same | `--distill 1` from run 2 | **First visible gain**: teacher gap −16–18%, the prompt's subject comes back |
| 4 | + 516 Nikki talk/pose clips | `--distill 1` from run 3, swap 8 | Gap −27%, data −19–21%. Poses clearly better; talking-head mouths still distort in coarse bands. On HF. |

Next ideas: more steps and mixed-style data, a higher `--dense-p` (the LoRA can bend bodies even with LoT off), a larger rank or nonlinear extent heads. Discussion: [help-wanted discussion](https://github.com/scrya-com/lot-diffusion/discussions/1).

## Gotchas

- **Sign:** H3's head predicts `x0 − ε`. The clean estimate is `y_t + σ·out`, not `flow.clean_from_velocity` (the toy model's `ε − x0`).
- **y-space:** after a Procrustes fit the scales are no longer 1, so the DiT input is `splice.to_y(x0)` noised, and output goes back through `to_x`. A splice without `y_space=True` refuses fitted scales.
- **Swap costs about 0.37 s per streamed block per forward.** On small inputs it hides the LoT speedup, so keep everything resident (`--swap 0`) when it fits.
- **Frozen base + coarse tokens = artifacts.** LoT needs the fine-tune; a fitted bank alone gives streaks and grid texture.
- **Style vs LoT:** single-style data makes the dense loss drop as well. Compare `lot_trained` with `dense_lora`, not with `dense`.
- **No fork of the runtime:** `h3_musubi.py` uses musubi-tuner's H3 model, loader, LoRA and VAE as a library. If a musubi update changes `MiniMaxH3Model.forward`, rerun `parity_h3.py` (it must stay `rel_rms=0`).
- **Layouts for faces:** fixed `bands` put the mouth in a coarse band on headshots. Use `center` (ComfyUI) for portraits.
- **The shell here is zsh:** an unquoted `$FLAGS` is *not* word-split, so `cmd $FLAGS` passes one argument and argparse fails. Write flags out, or use `${=FLAGS}`.
- **`pkill -f PATTERN` matches the shell running it** when the pattern appears in that command line. Look up the PID first, kill it by number in a second command.
- **Checkpoint names:** on this machine ComfyUI's `minimax_h3_fl2va_pruned_int8_convrot.safetensors` is a symlink to the *nvfp4* file. The adapters were trained on the real int8 file, Comfy-Org's 20,970,379,616-byte `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (linked here as `/media/2TB/models/minimax_h3/...`).
- **License:** the weights are derived from MiniMax H3 (Community License). Keep `LICENSE` + `NOTICE` with any copy, and not for use in the EU, UK, South Korea or USA.
