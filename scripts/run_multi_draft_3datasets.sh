#!/usr/bin/env bash
# Run multi-draft experiment on cnn_dailymail, gsm8k, eli5
# samples=100, L=4, B=[2,4,6,8], decoders={ms_pfr_cached, mpfr_torchgen_cached, invariant_multi}
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
mkdir -p "$LOGDIR"
cd "$REPO"

for ds in cnn gsm8k eli5; do
  CFG="experiments/configs/multi_draft_${ds}_n100_L4_B2468.json"
  LOG="$LOGDIR/multi_draft_${ds}_n100_L4_B2468.log"
  echo "[start] $(date '+%F %T') dataset=${ds}  cfg=${CFG}" | tee -a "$LOG"
  "$PY" -m experiments.run_multi_draft --config "$CFG" >> "$LOG" 2>&1
  rc=$?
  echo "[done ] $(date '+%F %T') dataset=${ds}  rc=${rc}" | tee -a "$LOG"
done
echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/multi_draft_3datasets.summary.log"
