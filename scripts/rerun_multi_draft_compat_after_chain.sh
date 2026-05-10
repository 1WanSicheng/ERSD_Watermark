#!/usr/bin/env bash
# Wait for the in-flight llama compat chain to finish, then re-run
# multi_draft cnn + eli5 under the fixed runner (with finalize_labels call).
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
cd "$REPO"

# Wait until the [ALL_DONE] marker appears in the chain summary log.
until [ -f "$LOGDIR/llama_compat_all.summary.log" ] && grep -q ALL_DONE "$LOGDIR/llama_compat_all.summary.log"; do
  sleep 60
done
echo "[chain done $(date '+%F %T')] launching multi_draft re-runs"

run() {
  local label="$1"
  local cfg="$2"
  local out="$3"
  local log="$LOGDIR/${label}.log"
  echo "[start] $(date '+%F %T') ${label}" | tee -a "$log"
  "$PY" -u -m experiments.run_multi_draft --config "$cfg" --output "$out" >> "$log" 2>&1
  rc=$?
  echo "[done ] $(date '+%F %T') ${label}  rc=${rc}" | tee -a "$log"
}

# Re-run with the same config files but write to *_FIXED.json so we don't
# overwrite the previous (buggy NaN-U) compat outputs.
run multi_draft_llama_cnn_n50_L4_B248_COMPAT_FIXED  experiments/configs/multi_draft_llama_cnn_n50_L4_B248_COMPAT.json  outputs/multi_draft_llama_cnn_n50_L4_B2-4-8_COMPAT_FIXED.json
run multi_draft_llama_eli5_n50_L4_B248_COMPAT_FIXED experiments/configs/multi_draft_llama_eli5_n50_L4_B248_COMPAT.json outputs/multi_draft_llama_eli5_n50_L4_B2-4-8_COMPAT_FIXED.json

echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/multi_draft_compat_FIXED.summary.log"
