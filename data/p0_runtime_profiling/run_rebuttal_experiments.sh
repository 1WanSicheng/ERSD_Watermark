#!/usr/bin/env bash
set -u

REPO=/data/opc/mpfr_p0/repo
PYTHON=/data/wansicheng3/envdllm/bin/python
RUN_ROOT=/data/opc/mpfr_p0/rebuttal_runtime_n100_cnn_paired
OUT_DIR="${RUN_ROOT}/json"
LOG_DIR="${RUN_ROOT}/logs"

export HF_HOME=/data/wansicheng3/hf_cache
export HF_DATASETS_CACHE=/data/wansicheng3/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${REPO}" || exit 1

PIDS=()
NAMES=()

launch_single() {
    local gpu=$1
    local name=$2
    local dataset=$3
    local lookahead=$4
    local top_k=$5
    shift 5
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        data/p0_runtime_profiling/profile_single_fourway.py \
        --dataset "${dataset}" \
        --samples 100 \
        --profile-samples 100 \
        --lookaheads "${lookahead}" \
        --max-new-tokens 128 \
        --top-k "${top_k}" \
        --methods "$@" \
        --output "${OUT_DIR}/${name}.json" \
        >"${LOG_DIR}/${name}.log" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    NAMES+=("${name}")
    echo "started ${name} gpu=${gpu} pid=${pid}"
}

launch_multi() {
    local gpu=$1
    local name=$2
    local dataset=$3
    local drafts=$4
    local top_k=$5
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        data/p0_runtime_profiling/profile_multi_two_way.py \
        --dataset "${dataset}" \
        --samples 100 \
        --profile-samples 100 \
        --lookahead 4 \
        --max-new-tokens 128 \
        --draft-counts "${drafts}" \
        --top-k "${top_k}" \
        --methods mpfr invariant \
        --output "${OUT_DIR}/${name}.json" \
        >"${LOG_DIR}/${name}.log" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    NAMES+=("${name}")
    echo "started ${name} gpu=${gpu} pid=${pid}"
}

wait_wave() {
    local failed=0
    local index
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            echo "completed ${NAMES[$index]}"
        else
            echo "failed ${NAMES[$index]}"
            failed=1
        fi
    done
    PIDS=()
    NAMES=()
    return "${failed}"
}

# Wave 1: the headline single- and multi-draft configurations.
# One CNN/DailyMail configuration per GPU. The k=50 cells are reused by the
# top-k sweep below.
gpu=0
for lookahead in 1 2 3 4; do
    launch_single \
        "${gpu}" "single_cnn_dailymail_L${lookahead}_k50" \
        cnn_dailymail "${lookahead}" 50 \
        vsps mse mws pfr_nowm pfr
    gpu=$((gpu + 1))
done

for drafts in 4 8; do
    launch_multi \
        "${gpu}" "multi_cnn_dailymail_B${drafts}_k50" \
        cnn_dailymail "${drafts}" 50
    gpu=$((gpu + 1))
done

launch_single 6 single_cnn_dailymail_L4_k100 cnn_dailymail 4 100 \
    mse mws pfr_nowm pfr
launch_single 7 single_cnn_dailymail_L4_k500 cnn_dailymail 4 500 \
    mse mws pfr_nowm pfr
wait_wave || true

# Wave 2: the remaining top-k cells. k=50 comes from Wave 1.
launch_single 0 single_cnn_dailymail_L4_knone cnn_dailymail 4 0 \
    mse mws pfr_nowm pfr
launch_multi 1 multi_cnn_dailymail_B4_k100 cnn_dailymail 4 100
launch_multi 2 multi_cnn_dailymail_B4_k500 cnn_dailymail 4 500
launch_multi 3 multi_cnn_dailymail_B4_knone cnn_dailymail 4 0
launch_multi 4 multi_cnn_dailymail_B8_k100 cnn_dailymail 8 100
launch_multi 5 multi_cnn_dailymail_B8_k500 cnn_dailymail 8 500
launch_multi 6 multi_cnn_dailymail_B8_knone cnn_dailymail 8 0
wait_wave || true

echo "all experiment waves finished"
