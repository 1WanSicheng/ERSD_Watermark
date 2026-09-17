#!/usr/bin/env bash
# Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct 3-seed production:
# 24 shards = 3 seeds x {cnn, eli5} x B={2,4,6,8}, n=1000, full metrics.
#
# Usage:  PYTHON=/path/to/python bash scripts/run_llama_3seed.sh
#         GPUS="0,1,2,3" bash scripts/run_llama_3seed.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/multi/shards_llama
PY="${PYTHON:-python}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4,5,6,7}"

SHARDS=()
for seed in s1 s2 s3; do
  for B in 2 4 6 8; do
    for ds in cnn eli5; do
      SHARDS+=("experiments/configs/shards_llama_seeds/shard_llama_${ds}_B${B}_${seed}.json")
    done
  done
done

declare -a GPU_PID
launch() {
    local gpu="$1" cfg="$2" name
    name=$(basename "${cfg%.json}")
    echo "[$(date '+%F %T')] GPU $gpu <- $name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.run_multi_draft \
        --config "$cfg" > "outputs/logs/${name}.log" 2>&1 &
    GPU_PID[$gpu]=$!
}

next=0
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

echo "[$(date '+%F %T')] all llama shards finished"
"$PY" scripts/summarize_pilot_tr.py outputs/multi/shards_llama/*.json
