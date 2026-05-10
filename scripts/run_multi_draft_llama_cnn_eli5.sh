#!/usr/bin/env bash
# Multi-draft on llama-7b / llama-68m, cnn + eli5, n=50, L=4, B=[2,4,8]
# 4 decoders: ms_pfr_cached, mpfr_torchgen_cached, invariant_multi, strong_multi
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
mkdir -p "$LOGDIR"
cd "$REPO"

for ds in cnn eli5; do
  CFG="experiments/configs/multi_draft_llama_${ds}_n50_L4_B248.json"
  LOG="$LOGDIR/multi_draft_llama_${ds}_n50_L4_B248.log"
  echo "[start] $(date '+%F %T') dataset=${ds}" | tee -a "$LOG"
  "$PY" -u -m experiments.run_multi_draft --config "$CFG" >> "$LOG" 2>&1
  rc=$?
  echo "[done ] $(date '+%F %T') dataset=${ds}  rc=${rc}" | tee -a "$LOG"
done
echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/multi_draft_llama_cnn_eli5.summary.log"
