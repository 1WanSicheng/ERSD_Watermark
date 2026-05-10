#!/usr/bin/env bash
# Single-draft on llama-7b / llama-68m, cnn + eli5, n=100, L=[1,2,3,4]
# 6 decoders: mc, basic_uwm, mc_uwm_speed, mc_uwm_strength, pfr, pfr_no_watermark
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
mkdir -p "$LOGDIR"
cd "$REPO"

for ds in cnn eli5; do
  CFG="experiments/configs/single_draft_llama_${ds}_n100_L1234.json"
  LOG="$LOGDIR/single_draft_llama_${ds}_n100_L1234.log"
  echo "[start] $(date '+%F %T') dataset=${ds}" | tee -a "$LOG"
  "$PY" -u -m experiments.run_single_draft --config "$CFG" >> "$LOG" 2>&1
  rc=$?
  echo "[done ] $(date '+%F %T') dataset=${ds}  rc=${rc}" | tee -a "$LOG"
done
echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/single_draft_llama_cnn_eli5.summary.log"
