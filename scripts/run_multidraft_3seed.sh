#!/usr/bin/env bash
# Run the 48 multi-draft shards behind Table 1 / Tables 6-9:
#   {qwen, llama} x {cnn, eli5} x B={2,4,6,8} x 3 seeds, n=1000 prompts each.
# Each shard runs MPFR and GLS sequentially in one process on one GPU, so the
# token-rate comparison within a shard is paired on identical hardware.
#
# Usage:  PYTHON=/path/to/python GPUS="0,1,2,3" bash scripts/run_multidraft_3seed.sh
#         then: python scripts/aggregate_3seed_tables.py --pair qwen  --results-dir outputs/multidraft_3seed
#               python scripts/aggregate_3seed_tables.py --pair llama --results-dir outputs/multidraft_3seed
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/logs outputs/multidraft_3seed
PY="${PYTHON:-python}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0}"
SHARDS=(experiments/configs/multidraft_3seed/*.json)

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
echo "[$(date '+%F %T')] all shards finished"
