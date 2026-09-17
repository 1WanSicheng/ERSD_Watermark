#!/usr/bin/env bash
# 8-GPU launcher for the full Tables 5-8 token-rate refresh.
#
# 16 shards = 4 cells x B={2,4,6,8}; each shard runs both decoders (MPFR then
# INVARIANT) sequentially in ONE process on ONE GPU, so the MPFR-vs-INVARIANT
# pairing within a shard is measured under identical conditions even when the
# node is fully loaded.
#
# Static schedule (longest-first): each GPU gets one Qwen shard (~3.6-4.2 h)
# then one Vicuna shard (~2.3 h)  =>  ~6-7 h wall clock total.
#
# Usage:  bash scripts/run_8gpu_refresh.sh            # all 16 shards on GPUs 0-7
#         GPUS="0,1,2,3" bash scripts/run_8gpu_refresh.sh   # fewer GPUs, queued
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/multi/shards
PY="${PYTHON:-python}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4,5,6,7}"
NGPU=${#GPU_LIST[@]}

# Longest shards first so stragglers start early.
SHARDS=(
  experiments/configs/shards_8gpu/shard_qwen_cnn_B2.json
  experiments/configs/shards_8gpu/shard_qwen_eli5_B2.json
  experiments/configs/shards_8gpu/shard_qwen_cnn_B4.json
  experiments/configs/shards_8gpu/shard_qwen_eli5_B4.json
  experiments/configs/shards_8gpu/shard_qwen_cnn_B6.json
  experiments/configs/shards_8gpu/shard_qwen_eli5_B6.json
  experiments/configs/shards_8gpu/shard_qwen_cnn_B8.json
  experiments/configs/shards_8gpu/shard_qwen_eli5_B8.json
  experiments/configs/shards_8gpu/shard_vicuna_cnn_B2.json
  experiments/configs/shards_8gpu/shard_vicuna_eli5_B2.json
  experiments/configs/shards_8gpu/shard_vicuna_cnn_B4.json
  experiments/configs/shards_8gpu/shard_vicuna_eli5_B4.json
  experiments/configs/shards_8gpu/shard_vicuna_cnn_B6.json
  experiments/configs/shards_8gpu/shard_vicuna_eli5_B6.json
  experiments/configs/shards_8gpu/shard_vicuna_cnn_B8.json
  experiments/configs/shards_8gpu/shard_vicuna_eli5_B8.json
)

# Pre-warm the HF dataset/model cache once so 8 concurrent first-runs don't
# race on downloads (skipped instantly if already cached).
"$PY" - <<'PY'
from experiments import _shared as S
S.load_prompts("cnn_dailymail", 1000)
S.load_prompts("eli5", 1000)
print("[prewarm] prompt caches ready")
PY

declare -a GPU_PID
launch() {  # launch <gpu> <config>
    local gpu="$1" cfg="$2" name
    name=$(basename "${cfg%.json}")
    echo "[$(date '+%F %T')] GPU $gpu <- $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.run_multi_draft \
        --config "$cfg" > "outputs/logs/${name}.log" 2>&1 &
    GPU_PID[$gpu]=$!
}

next=0
# Fill every GPU, then hand each finishing GPU the next queued shard.
while (( next < ${#SHARDS[@]} )); do
    launched=0
    for gpu in "${GPU_LIST[@]}"; do
        if [[ -z "${GPU_PID[$gpu]:-}" ]] || ! kill -0 "${GPU_PID[$gpu]}" 2>/dev/null; then
            if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
                wait "${GPU_PID[$gpu]}" || echo "[warn] a shard on GPU $gpu exited non-zero (see logs)"
            fi
            launch "$gpu" "${SHARDS[$next]}"
            next=$((next + 1)); launched=1
            (( next < ${#SHARDS[@]} )) || break
        fi
    done
    (( launched )) || sleep 60
done
wait || true

echo "[$(date '+%F %T')] all shards finished"
"$PY" scripts/summarize_pilot_tr.py outputs/multi/shards/*.json
