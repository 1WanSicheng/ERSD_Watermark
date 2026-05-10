#!/usr/bin/env bash
# Re-run all 4 llama experiments under use_chat_template=false (compat mode):
#   1. multi_draft cnn  n=50  L=4  B=[2,4,8]   (4 decoders)  ~25 min
#   2. multi_draft eli5 n=50  L=4  B=[2,4,8]   (4 decoders)  ~20 min
#   3. single_draft cnn  n=100 L=[1..4]        (6 decoders)  ~75 min
#   4. single_draft eli5 n=100 L=[1..4]        (6 decoders)  ~75 min
# Order: multi-draft first (faster) so we get results to inspect early.
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
mkdir -p "$LOGDIR"
cd "$REPO"

run() {
  local label="$1"
  local mode="$2"   # "multi" or "single"
  local cfg="$3"
  local log="$LOGDIR/${label}.log"
  echo "[start] $(date '+%F %T') ${label}" | tee -a "$log"
  if [ "$mode" = "multi" ]; then
    "$PY" -u -m experiments.run_multi_draft  --config "$cfg" >> "$log" 2>&1
  else
    "$PY" -u -m experiments.run_single_draft --config "$cfg" >> "$log" 2>&1
  fi
  rc=$?
  echo "[done ] $(date '+%F %T') ${label}  rc=${rc}" | tee -a "$log"
}

run multi_draft_llama_cnn_n50_L4_B248_COMPAT  multi  experiments/configs/multi_draft_llama_cnn_n50_L4_B248_COMPAT.json
run multi_draft_llama_eli5_n50_L4_B248_COMPAT multi  experiments/configs/multi_draft_llama_eli5_n50_L4_B248_COMPAT.json
run single_draft_llama_cnn_n100_L1234_COMPAT  single experiments/configs/single_draft_llama_cnn_n100_L1234_COMPAT.json
run single_draft_llama_eli5_n100_L1234_COMPAT single experiments/configs/single_draft_llama_eli5_n100_L1234_COMPAT.json

echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/llama_compat_all.summary.log"
