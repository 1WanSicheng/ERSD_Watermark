#!/usr/bin/env bash
set -euo pipefail

repo=/data/opc/mpfr_p0/repo
python_bin=/data/wansicheng3/envdllm/bin/python
target=/data/wansicheng3/hf_cache/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
draft=/data/wansicheng3/hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
output_dir=/data/opc/mpfr_p0/rebuttal_qwen72_metrics_n100

export HF_HOME=/data/wansicheng3/hf_cache
export HF_DATASETS_CACHE=/data/wansicheng3/hf_cache/datasets
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/data/wansicheng3/env/lib/python3.12/site-packages

cd "$repo"
mkdir -p "$output_dir"

"$python_bin" data/p0_runtime_profiling/run_qwen72_metrics.py \
  --mode single \
  --target "$target" \
  --draft "$draft" \
  --samples 100 \
  --max-new-tokens 128 \
  --output "$output_dir/single_metrics_n100.json" \
  > "$output_dir/single_metrics_n100.log" 2>&1

"$python_bin" data/p0_runtime_profiling/run_qwen72_metrics.py \
  --mode multi \
  --target "$target" \
  --draft "$draft" \
  --samples 100 \
  --max-new-tokens 128 \
  --output "$output_dir/multi_metrics_n100.json" \
  > "$output_dir/multi_metrics_n100.log" 2>&1
