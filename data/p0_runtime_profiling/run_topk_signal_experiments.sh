#!/usr/bin/env bash
set -u

REPO=/data/opc/mpfr_p0/repo
PYTHON=/data/wansicheng3/envdllm/bin/python
RUN_ROOT=/data/opc/mpfr_p0/rebuttal_topk_signal_n100
OUT_DIR="${RUN_ROOT}/json"
LOG_DIR="${RUN_ROOT}/logs"

export HF_HOME=/data/wansicheng3/hf_cache
export HF_DATASETS_CACHE=/data/wansicheng3/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${REPO}" || exit 1

PIDS=()
NAMES=()

launch() {
    local gpu=$1
    local mode=$2
    local top_k=$3
    local label=$4
    local name="${mode}_k${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        data/p0_runtime_profiling/run_topk_signal_only.py \
        --mode "${mode}" \
        --top-k "${top_k}" \
        --samples 100 \
        --output "${OUT_DIR}/${name}.json" \
        >"${LOG_DIR}/${name}.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${name}")
    echo "started ${name} gpu=${gpu} pid=$!"
}

launch 0 single 50 50
launch 1 single 100 100
launch 2 single 500 500
launch 3 single 0 none
launch 4 multi 50 50
launch 5 multi 100 100
launch 6 multi 500 500
launch 7 multi 0 none

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "completed ${NAMES[$index]}"
    else
        echo "failed ${NAMES[$index]}"
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi
echo "all top-k signal-only experiments finished"
