#!/usr/bin/env bash
# Patched draft-KV-cache version of run_mpfr_cached_benchmark on gsm8k + eli5
# samples=100, L=4, B=[2,4,6,8], methods=ms_pfr_batched_cached + mpfr_batched_torchgen_cached
set -u
REPO="/root/PFR/ERSD_Watermark"
PY="/root/miniconda3/envs/pfr/bin/python"
LOGDIR="$REPO/outputs/logs"
mkdir -p "$LOGDIR"
cd "$REPO"

for ds in gsm8k eli5; do
  OUT="outputs/mpfr_cached_benchmark_${ds}_n100_L4_B2-4-6-8_DRAFTCACHE.json"
  LOG="$LOGDIR/mpfr_cached_benchmark_${ds}_n100_L4_B2-4-6-8_DRAFTCACHE.log"
  echo "[start] $(date '+%F %T') dataset=${ds}" | tee -a "$LOG"
  "$PY" -u MPFR_spec/run_mpfr_cached_benchmark.py \
    --samples 100 --warmup 1 --lookahead 4 --max-new-tokens 128 \
    --num-drafts 2 4 6 8 --top-k 50 --top-p 1.0 --temperature 1.0 \
    --dataset "$ds" \
    --methods ms_pfr_batched_cached mpfr_batched_torchgen_cached \
    --output "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "[done ] $(date '+%F %T') dataset=${ds}  rc=${rc}" | tee -a "$LOG"
done
echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$LOGDIR/mpfr_cached_benchmark_gsm8k_eli5_DRAFTCACHE.summary.log"
