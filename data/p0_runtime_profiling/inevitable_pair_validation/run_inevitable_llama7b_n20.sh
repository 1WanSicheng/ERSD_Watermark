#!/usr/bin/env bash
set -euo pipefail

repo=/data/opc/mpfr_p0/repo
python_bin=/data/wansicheng3/envdllm/bin/python
target=/data/opc/mpfr_p0/inevitable_pair/models/llama-7b
draft=/data/opc/mpfr_p0/inevitable_pair/models/llama-68m
output_dir=/data/opc/mpfr_p0/inevitable_pair/n20

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/wansicheng3/hf_cache
export HF_HUB_OFFLINE=1

cd "$repo"
mkdir -p "$output_dir"

common_args=(
  --target "$target"
  --draft "$draft"
  --dtype float16
  --dataset cnn_paper_summarization
  --samples 20
  --profile-samples 5
  --warmup 1
  --max-new-tokens 128
  --top-k 0
)

"$python_bin" data/p0_runtime_profiling/profile_single_fourway.py \
  "${common_args[@]}" \
  --lookaheads 4 \
  --methods basic basic_uwm \
  --output "$output_dir/baselines.json" \
  > "$output_dir/baselines.log" 2>&1

"$python_bin" data/p0_runtime_profiling/profile_single_fourway.py \
  "${common_args[@]}" \
  --lookaheads 1 2 3 4 \
  --methods vsps pfr_nowm pfr \
  --output "$output_dir/vsps_pfr.json" \
  > "$output_dir/vsps_pfr.log" 2>&1

"$python_bin" data/p0_runtime_profiling/profile_single_fourway.py \
  "${common_args[@]}" \
  --lookaheads 1 2 3 4 \
  --methods mse mws \
  --output "$output_dir/mse_mws.json" \
  > "$output_dir/mse_mws.log" 2>&1

echo "all inevitable-pair validation runs completed"
