#!/usr/bin/env bash
set -euo pipefail

repo=/data/opc/mpfr_p0/repo
python_bin=/data/wansicheng3/envdllm/bin/python
target=/data/wansicheng3/hf_cache/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
draft=/data/wansicheng3/hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
output_dir=/data/opc/mpfr_p0/qwen72_pilot

export HF_HOME=/data/wansicheng3/hf_cache
export HF_HUB_OFFLINE=1

cd "$repo"
mkdir -p "$output_dir"

"$python_bin" data/p0_runtime_profiling/profile_single_fourway.py \
  --target "$target" \
  --draft "$draft" \
  --target-device-map balanced \
  --dtype bfloat16 \
  --dataset cnn_dailymail \
  --samples 10 \
  --profile-samples 10 \
  --warmup 1 \
  --lookaheads 4 \
  --max-new-tokens 128 \
  --top-k 50 \
  --methods vsps pfr_nowm pfr \
  --output "$output_dir/single_n10.json" \
  > "$output_dir/single_n10.log" 2>&1

"$python_bin" data/p0_runtime_profiling/profile_multi_two_way.py \
  --target "$target" \
  --draft "$draft" \
  --target-device-map balanced \
  --dtype bfloat16 \
  --dataset cnn_dailymail \
  --samples 10 \
  --profile-samples 10 \
  --warmup 1 \
  --lookahead 4 \
  --max-new-tokens 128 \
  --top-k 50 \
  --draft-counts 4 8 \
  --methods mpfr invariant \
  --output "$output_dir/multi_n10.json" \
  > "$output_dir/multi_n10.log" 2>&1
