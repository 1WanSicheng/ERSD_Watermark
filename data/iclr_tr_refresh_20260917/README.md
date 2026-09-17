# ICLR token-rate refresh (2026-09) — curated results

Refresh of the multi-draft tables (paper Tables 5–8) with the optimized MPFR
implementation (`MPFR_spec/mpfr_batched_torchgen_cached.py`, post
sampling-path optimization). Raw per-prompt outputs (~950 MB, incl.
`u_per_token` detector inputs) are NOT in git — they live in
`outputs/multi/shards*/` on the run machine and local mirrors; this folder
keeps the per-shard `config` + cell-level `summary` blocks plus the final
aggregated tables, which is everything the paper tables need.

## Protocol

- L=4, B ∈ {2,4,6,8}, 1000 prompts/cell, max 128 new tokens, top-k 50, temp 1.
- 3 seeds via `seed_offset` ∈ {0, 10000, 20000} (per-prompt seed = idx+7+offset;
  offset 0 is the original single-seed protocol).
- Decoders: `mpfr_torchgen_cached` (MPFR, watermark on) vs `invariant_multi`
  (unkeyed baseline, no watermark). Paired: both run sequentially in the same
  process on the same dedicated GPU per (cell, B, seed) shard.
- Model pairs: Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct (Tables 5–6) and
  Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct via unsloth mirrors
  (Tables 7–8; replaces Vicuna-7B-v1.5 / vicuna-68m of the NeurIPS version).
- Datasets: CNN/DailyMail summarization; ELI5 long-form QA.
- Hardware: NVIDIA A100-SXM4-40GB (driver 595.71, CUDA 13.2), 2× AMD EPYC
  7542, 2 TB RAM; float16; Python 3.12.13, torch 2.11.0+cu130,
  transformers 4.46.3, datasets 5.0.0.

## Headline result

MPFR's token rate exceeds the unwatermarked INVARIANT baseline in **48/48**
(seed × cell × B) paired comparisons: +0.7%..+4.9% on Qwen, +0.4%..+5.9% on
Llama. AATPS is also ≥ baseline everywhere; ANLPPT/LPPL are stable to
±0.001–0.005 across seeds. (NeurIPS-version implementation was 3.5–17.1%
slower in TR across Tables 5–8.)

## Contents

- `shards_qwen_s1/` — Qwen seed 1 (from the 8-GPU full 2×2 production run,
  2026-09-11; also contains the 8 Vicuna shards of that run for reference).
- `shards_qwen_s2s3/` — Qwen seeds 2–3 (4-GPU run, 2026-09-13).
- `shards_llama/` — Llama seeds 1–3 (4-GPU run, 2026-09-14/15).
- `pilots/` — n=100 pilot summaries (4 old cells + 2 Llama cells).
- `tables_5_6_qwen_3seed.txt`, `tables_7_8_llama_3seed.txt` — final
  mean ± std tables (plain text + LaTeX rows), regenerable via
  `python scripts/aggregate_3seed_tables.py --pair {qwen,llama} --latex`.
