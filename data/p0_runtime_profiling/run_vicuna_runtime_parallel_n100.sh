#!/usr/bin/env bash
set -euo pipefail

repo=/data/opc/mpfr_p0/repo
python_bin=/data/wansicheng3/envdllm/bin/python
output_dir=/data/opc/mpfr_p0/rebuttal_vicuna_runtime_parallel_n100

export HF_HOME=/data/wansicheng3/hf_cache
export HF_DATASETS_CACHE=/data/wansicheng3/hf_cache/datasets
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$repo"
mkdir -p "$output_dir"

pids=()
names=()

launch_single() {
  local gpu=$1
  local cpus=$2
  local method=$3
  local name="single_${method}"
  CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$cpus" "$python_bin" \
    data/p0_runtime_profiling/profile_single_fourway.py \
    --target lmsys/vicuna-7b-v1.5 \
    --draft double7/vicuna-68m \
    --dtype float16 \
    --dataset cnn_dailymail \
    --samples 100 \
    --profile-samples 100 \
    --warmup 1 \
    --lookaheads 4 \
    --max-new-tokens 128 \
    --top-k 50 \
    --methods "$method" \
    --output "$output_dir/${name}_n100.json" \
    > "$output_dir/${name}_n100.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "started $name gpu=$gpu cpus=$cpus pid=$!"
}

launch_multi() {
  local gpu=$1
  local cpus=$2
  local method=$3
  local b=$4
  local name="multi_${method}_B${b}"
  CUDA_VISIBLE_DEVICES="$gpu" taskset -c "$cpus" "$python_bin" \
    data/p0_runtime_profiling/profile_multi_two_way.py \
    --target lmsys/vicuna-7b-v1.5 \
    --draft double7/vicuna-68m \
    --dtype float16 \
    --dataset cnn_dailymail \
    --samples 100 \
    --profile-samples 100 \
    --warmup 1 \
    --lookahead 4 \
    --max-new-tokens 128 \
    --top-k 50 \
    --draft-counts "$b" \
    --methods "$method" \
    --output "$output_dir/${name}_n100.json" \
    > "$output_dir/${name}_n100.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "started $name gpu=$gpu cpus=$cpus pid=$!"
}

# Match each process to CPUs local to its GPU's NUMA node. GPU pairs share a
# node, so each process receives a disjoint half of that node's logical CPUs.
launch_single 0 "24-27,88-91" vsps
launch_single 1 "28-31,92-95" pfr_nowm
launch_single 2 "8-11,72-75" pfr
launch_multi 3 "12-15,76-79" mpfr 4
launch_multi 4 "56-59,120-123" invariant 4
launch_multi 5 "60-63,124-127" mpfr 8
launch_multi 6 "40-47,104-111" invariant 8

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${names[$index]}"
  else
    echo "failed ${names[$index]}"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "all Vicuna runtime cells finished"
