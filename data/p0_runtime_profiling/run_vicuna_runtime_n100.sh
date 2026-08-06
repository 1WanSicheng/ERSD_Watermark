#!/usr/bin/env bash
set -euo pipefail

repo=/data/opc/mpfr_p0/repo
python_bin=/data/wansicheng3/envdllm/bin/python
output_dir=/data/opc/mpfr_p0/rebuttal_vicuna_runtime_n100

export HF_HOME=/data/wansicheng3/hf_cache
export HF_DATASETS_CACHE=/data/wansicheng3/hf_cache/datasets
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$repo"
mkdir -p "$output_dir"

"$python_bin" data/p0_runtime_profiling/profile_single_fourway.py \
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
  --methods vsps pfr_nowm pfr \
  --output "$output_dir/single_n100.json" \
  > "$output_dir/single_n100.log" 2>&1

"$python_bin" data/p0_runtime_profiling/profile_multi_two_way.py \
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
  --draft-counts 4 8 \
  --methods mpfr invariant \
  --output "$output_dir/multi_n100.json" \
  > "$output_dir/multi_n100.log" 2>&1
