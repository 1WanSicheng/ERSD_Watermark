#!/usr/bin/env bash
# Qwen multi-seed replicates: seeds 2 and 3 (seed_offset 10000 / 20000) for
# both Qwen cells, B={2,4,6,8}.  16 shards, default 4 GPUs => 4 rounds of
# ~2h each => ~8-9 h wall clock.  Seed 1 is the existing production run in
# outputs/multi/shards/.
#
# Usage:  PYTHON=/path/to/python bash scripts/run_qwen_3seed.sh
#         GPUS="4,5,6,7" bash scripts/run_qwen_3seed.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/multi/shards_seeds
PY="${PYTHON:-python}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"

SHARDS=()
for seed in s2 s3; do
  for B in 2 4 6 8; do
    for ds in cnn eli5; do
      SHARDS+=("experiments/configs/shards_qwen_seeds/shard_qwen_${ds}_B${B}_${seed}.json")
    done
  done
done

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

echo "[$(date '+%F %T')] all seed shards finished"
"$PY" scripts/summarize_pilot_tr.py outputs/multi/shards_seeds/*.json
