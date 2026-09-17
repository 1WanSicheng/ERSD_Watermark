#!/usr/bin/env bash
# ICLR token-rate refresh driver (run on the A100 server, one uncontended GPU).
#
# Phase 1 (pilot, ~3-4 GPU-hours total): confirm the optimized MPFR closes the
#   TR gap vs INVARIANT in all four cells at n=100 before committing to n=1000.
#   Priority order: vicuna_cnn (largest paper gap) -> qwen_eli5 -> vicuna_eli5
#   -> qwen_cnn (already evidenced at B=4/8, reruns for B=2/6 coverage).
#
# Phase 2 (priority production, ~1 GPU-day): Qwen x {CNN, ELI5}, B={4,6,8},
#   n=1000, full metrics (watermark data collected in the same pass).
#
# Phase 3 (deferred, run when GPUs free up): Qwen B=2 completion, then the two
#   Vicuna cells with the existing full_vicuna_*_n1000.json configs.
#
# Usage:  bash scripts/run_iclr_tr_refresh.sh pilot
#         bash scripts/run_iclr_tr_refresh.sh prio
#         bash scripts/run_iclr_tr_refresh.sh deferred
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
    local cfg="$1"
    echo "=== $(date '+%F %T')  $cfg ==="
    python -m experiments.run_multi_draft --config "$cfg" 2>&1 \
        | tee "outputs/logs/$(basename "${cfg%.json}").log"
}

mkdir -p outputs/logs outputs/pilot outputs/multi

case "${1:-pilot}" in
  pilot)
    run experiments/configs/pilot_tr_n100/pilot_tr_vicuna_cnn_n100.json
    run experiments/configs/pilot_tr_n100/pilot_tr_qwen_eli5_n100.json
    run experiments/configs/pilot_tr_n100/pilot_tr_vicuna_eli5_n100.json
    run experiments/configs/pilot_tr_n100/pilot_tr_qwen_cnn_n100.json
    python scripts/summarize_pilot_tr.py outputs/pilot/*.json
    ;;
  prio)
    run experiments/configs/prio_qwen_cnn_n1000_B468.json
    run experiments/configs/prio_qwen_eli5_n1000_B468.json
    ;;
  deferred)
    run experiments/configs/prio_qwen_cnn_n1000_B2.json
    run experiments/configs/prio_qwen_eli5_n1000_B2.json
    run experiments/configs/full_vicuna_cnn_n1000.json
    run experiments/configs/full_vicuna_eli5_n1000.json
    ;;
  *)
    echo "usage: $0 {pilot|prio|deferred}" >&2; exit 1;;
esac
