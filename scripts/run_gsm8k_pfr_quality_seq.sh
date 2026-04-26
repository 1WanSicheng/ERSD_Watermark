#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main"
cd "$REPO_DIR"

TARGET_MODEL="$REPO_DIR/model/Qwen2.5-7B-Instruct"
DRAFT_MODEL="$REPO_DIR/model/Qwen2.5-0.5B-Instruct"
LOG_DIR="$REPO_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

for L in 1 2 4 6; do
  echo "[start] $(date '+%F %T %Z') lookahead=${L}"
  python experiments/run_pfr_quality.py \
    --dataset gsm8k \
    --samples 1000 \
    --lookahead "$L" \
    --max-length 128 \
    --warmup 2 \
    --target-model "$TARGET_MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --target-device cuda:0 \
    --draft-device cuda:0 \
    --output "outputs/gsm8k_pfr_quality_l${L}_1000.json" \
    --rows-output "outputs/gsm8k_pfr_quality_l${L}_1000.jsonl" \
    --progress-every 50
  echo "[done] $(date '+%F %T %Z') lookahead=${L}"
done
