#!/usr/bin/env bash
# Two-track (MPEG-Diffusion) training on H3, every stage, resumable. See lot/CHEATSHEET.md section 7.
#
#   WORK=/data/tt bash lot/train_tt_all.sh            # all stages
#   WORK=/data/tt STAGES="lora render" bash lot/train_tt_all.sh
#
# Needs LOT_H3_CHECKPOINT, LOT_H3_VAE and LOT_H3_TEXT_ENCODER (lot/lot_paths.py). One GPU job at a
# time: on a multi-GPU box pick the card with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N.
set -euo pipefail
cd "$(dirname "$0")/.."

WORK=${WORK:?set WORK to a data directory with ~10 GB free}
PY=${PY:-python3}
COUNT=${COUNT:-2000}                 # photos to download
SIZE=${SIZE:-"1344 768"}             # training canvas (W H), multiples of 64
STEPS=${STEPS:-2000}                 # LoRA steps
SWITCH=${SWITCH:-12}                 # sampler switch the region head is trained for (of 20 steps)
STAGES=${STAGES:-"fetch cache pairs fitkv lora region decoder render"}
TE_FROM=${TE_FROM:-}                 # HF dataset with te/*_minimaxh3_te.safetensors (optional)
export LOT_H3_TRAIN=1

has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$WORK"

has fetch   && $PY lot/fetch_pexels.py --out "$WORK/photos" --count "$COUNT"
has cache   && $PY lot/build_cache_h3.py media --src "$WORK/photos" --out "$WORK/cache" --size $SIZE
# Optional: precomputed caption embeddings (skips the 32B text encoder for captions it covers), e.g.
# TE_FROM=johndpope/MiniMax-H3-pexels-people-text
if has cache && [[ -n "${TE_FROM:-}" ]]; then
  hf download "$TE_FROM" --repo-type dataset --include "te/*" --local-dir "$WORK/te_download"
  cp -n "$WORK"/te_download/te/*_minimaxh3_te.safetensors "$WORK/cache/"
fi
has cache   && $PY lot/build_cache_h3.py text  --out "$WORK/cache"
has pairs   && $PY lot/build_pairs_tt.py --cache "$WORK/cache" --out "$WORK/pairs" --subject yolo
has fitkv   && $PY lot/two_track_h3.py fit-kv --pairs "$WORK/pairs" --out "$WORK/fit" --items 200
has lora    && $PY lot/train_tt.py lora    --pairs "$WORK/pairs" --out "$WORK/lora" --maps "$WORK/fit/up_maps_tt.pt" \
                 --steps "$STEPS"
has region  && $PY lot/train_tt.py region  --pairs "$WORK/pairs" --out "$WORK/region" --train "$WORK/lora" \
                 --switch "$SWITCH"
has decoder && $PY lot/train_tt.py decoder --pairs "$WORK/pairs" --out "$WORK/decoder"
has render  && $PY lot/two_track_h3.py render --out "$WORK/render" --cache "$WORK/cache" --holdout 8 --prompts 4 \
                 --switch "$SWITCH" --hr-steps 0 --train "$WORK/lora" --region "$WORK/region/region_head_tt.pt" \
                 --decoder "$WORK/decoder/decoder.pt" --outline
echo "train_tt_all: done ($STAGES) in $WORK"
