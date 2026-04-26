#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main"
cd "$REPO_DIR"

TARGET_MODEL="$REPO_DIR/model/Qwen2.5-7B-Instruct"
DRAFT_MODEL="$REPO_DIR/model/Qwen2.5-0.5B-Instruct"

for L in 1 2 6; do
  echo "[start] $(date '+%F %T %Z') lookahead=${L}"
  CUDA_VISIBLE_DEVICES=0 python "$REPO_DIR/experiments/run_mc_uwm_quality.py" \
    --dataset gsm8k \
    --samples 1000 \
    --lookahead "$L" \
    --max-length 128 \
    --warmup 2 \
    --methods mc_uwm_speed \
    --reweight deltagumbel \
    --seed 1 \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --target-device cuda:0 \
    --draft-device cuda:0 \
    --output "$REPO_DIR/outputs/gsm8k_mc_uwm_speed_l${L}_1000_clean.json" \
    --rows-output "$REPO_DIR/outputs/gsm8k_mc_uwm_speed_l${L}_1000_clean.jsonl" \
    --progress-every 50
  echo "[done] $(date '+%F %T %Z') lookahead=${L}"
done
