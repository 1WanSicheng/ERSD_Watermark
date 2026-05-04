# 2x2 multi_draft sweep (n=1000) - partial report

_generated 2026-05-04 04:35, qwen_eli5 still re-running_

## Setup

- **Models**: Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct, lmsys/vicuna-7b-v1.5 / double7/vicuna-68m
- **Datasets**: cnn_dailymail, eli5
- **Decoders**: `mpfr_torchgen_cached` (PFR + multi-draft, watermark) and `invariant_multi` (no watermark, baseline)
- **Sweep**: L=4, B in {2,4,6,8}, samples = 1000 per (decoder, B)
- **process_logits**: top_k=50, top_p=1.0, temperature=1.0  (private_key="1234")
- **Hardware**: 4x RTX A6000 (48 GB), one cell per GPU; NUMA-pinned to socket 0/1 per GPU.
- **Caveat**: this pod is roughly 30% slower in absolute TR than the 0502 reference machine, but **relative comparisons within this run are valid**.

## Cells completed

| Cell | Status |
|---|---|
| **qwen_cnn** | done |
| **qwen_eli5** | re-running (split mpfr+inv on GPU 0/1) |
| **vicuna_cnn** | done |
| **vicuna_eli5** | done |

## Per-cell results

### qwen_cnn

| decoder | B | AATPS | TR | ANLPPT_U | ANLPPT_Li | ANLPPT_PL | LPPL | KL/WS | TPR_U |
|---|---|---|---|---|---|---|---|---|---|
| mpfr_torchgen_cached | 2 | 2.799 | 24.88 | 0.052 | 0.037 | 0.055 | 0.475 | 0.998 | 0.219 |
| mpfr_torchgen_cached | 4 | 3.123 | 26.51 | 0.052 | 0.037 | 0.055 | 0.474 | 0.997 | 0.220 |
| mpfr_torchgen_cached | 6 | 3.299 | 26.88 | 0.052 | 0.037 | 0.055 | 0.475 | 0.999 | 0.222 |
| mpfr_torchgen_cached | 8 | 3.410 | 26.91 | 0.052 | 0.037 | 0.055 | 0.476 | 0.998 | 0.221 |
| invariant_multi | 2 | 2.804 | 26.89 | - | - | - | 0.487 | - | - |
| invariant_multi | 4 | 3.114 | 28.00 | - | - | - | 0.481 | - | - |
| invariant_multi | 6 | 3.275 | 28.29 | - | - | - | 0.482 | - | - |
| invariant_multi | 8 | 3.387 | 27.89 | - | - | - | 0.479 | - | - |

### qwen_eli5

_Pending qwen_eli5 re-run completion._

### vicuna_cnn

| decoder | B | AATPS | TR | ANLPPT_U | ANLPPT_Li | ANLPPT_PL | LPPL | KL/WS | TPR_U |
|---|---|---|---|---|---|---|---|---|---|
| mpfr_torchgen_cached | 2 | 2.422 | 53.96 | 0.022 | 0.016 | 0.022 | 0.269 | 0.987 | 0.066 |
| mpfr_torchgen_cached | 4 | 2.664 | 50.16 | 0.022 | 0.017 | 0.023 | 0.271 | 0.990 | 0.066 |
| mpfr_torchgen_cached | 6 | 2.803 | 46.51 | 0.022 | 0.016 | 0.023 | 0.270 | 0.987 | 0.066 |
| mpfr_torchgen_cached | 8 | 2.900 | 42.87 | 0.022 | 0.017 | 0.023 | 0.270 | 0.988 | 0.066 |
| invariant_multi | 2 | 2.478 | 57.63 | - | - | - | 0.287 | - | - |
| invariant_multi | 4 | 2.690 | 52.58 | - | - | - | 0.286 | - | - |
| invariant_multi | 6 | 2.831 | 48.05 | - | - | - | 0.283 | - | - |
| invariant_multi | 8 | 2.925 | 43.76 | - | - | - | 0.288 | - | - |

### vicuna_eli5

| decoder | B | AATPS | TR | ANLPPT_U | ANLPPT_Li | ANLPPT_PL | LPPL | KL/WS | TPR_U |
|---|---|---|---|---|---|---|---|---|---|
| mpfr_torchgen_cached | 2 | 1.889 | 47.45 | 0.058 | 0.042 | 0.061 | 0.504 | 0.995 | 0.362 |
| mpfr_torchgen_cached | 4 | 2.132 | 49.76 | 0.058 | 0.042 | 0.061 | 0.504 | 0.995 | 0.363 |
| mpfr_torchgen_cached | 6 | 2.278 | 50.36 | 0.058 | 0.042 | 0.061 | 0.503 | 0.996 | 0.365 |
| mpfr_torchgen_cached | 8 | 2.377 | 49.85 | 0.059 | 0.043 | 0.062 | 0.504 | 0.994 | 0.366 |
| invariant_multi | 2 | 1.884 | 49.55 | - | - | - | 0.555 | - | - |
| invariant_multi | 4 | 2.108 | 52.48 | - | - | - | 0.552 | - | - |
| invariant_multi | 6 | 2.248 | 54.11 | - | - | - | 0.546 | - | - |
| invariant_multi | 8 | 2.361 | 55.16 | - | - | - | 0.539 | - | - |

## Cross-cell pivots (mpfr_torchgen_cached only)

### AATPS (acceptance rate)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 2.799 | 3.123 | 3.299 | 3.410 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 2.422 | 2.664 | 2.803 | 2.900 |
| vicuna_eli5 | 1.889 | 2.132 | 2.278 | 2.377 |

### token_rate (tok/s)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 24.884 | 26.505 | 26.878 | 26.909 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 53.964 | 50.159 | 46.514 | 42.873 |
| vicuna_eli5 | 47.445 | 49.761 | 50.357 | 49.851 |

### ANLPPT_U (per-token watermark detectability)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 0.052 | 0.052 | 0.052 | 0.052 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 0.022 | 0.022 | 0.022 | 0.022 |
| vicuna_eli5 | 0.058 | 0.058 | 0.058 | 0.059 |

### LPPL (mean per-token NLL under target's processed dist)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 0.475 | 0.474 | 0.475 | 0.476 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 0.269 | 0.271 | 0.270 | 0.270 |
| vicuna_eli5 | 0.504 | 0.504 | 0.503 | 0.504 |

### KL/WS ratio (PFR detector recovery; 1.0 = saturated)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 0.998 | 0.997 | 0.999 | 0.998 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 0.987 | 0.990 | 0.987 | 0.988 |
| vicuna_eli5 | 0.995 | 0.995 | 0.996 | 0.994 |

### TPR_U @ 64 tokens @ FPR=1%

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 0.219 | 0.220 | 0.222 | 0.221 |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 0.066 | 0.066 | 0.066 | 0.066 |
| vicuna_eli5 | 0.362 | 0.363 | 0.365 | 0.366 |

### PFR overhead (TR loss vs `invariant_multi` same B)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 7.5% | 5.3% | 5.0% | 3.5% |
| qwen_eli5 | - | - | - | - |
| vicuna_cnn | 6.4% | 4.6% | 3.2% | 2.0% |
| vicuna_eli5 | 4.3% | 5.2% | 6.9% | 9.6% |

## Key observations

1. **Watermark strength independent of B**: ANLPPT_U / KL / TPR_U all flat across B in {2,4,6,8} (within Monte-Carlo noise at n=1000). Confirms PFR's promise that multi-draft scaling does not erode the watermark.
2. **KL/WS ratio ~ 0.99-1.00 in every cell**: PFR pushes the watermark almost to the entropy-budget upper bound. The trade-off frontier is achieved tightly.
3. **PFR engineering overhead < 10% in every cell**: vs `invariant_multi` baseline the TR penalty ranges 2-10%, much smaller than the ~17% measured in early smoke tests. The B1+B2+B3 optimisations and the batched logits patch are paying off.
4. **B affects throughput non-trivially per dataset**: vicuna_cnn TR DECREASES monotonically with B (54 -> 43 tok/s) - multi-draft compute cost outpaces the AATPS gain. vicuna_eli5 peaks around B=6. Qwen TR rises slowly with B (because the 0.5B draft already dominates per-step time).
5. **AATPS comparison: mpfr is approximately equal to invariant** (within +-0.05): PFR's reweighting does not measurably hurt acceptance.
6. **LPPL: mpfr <= invariant systematically** (e.g. vicuna_eli5 mpfr 0.504 vs invariant 0.539-0.555, ~5 sigma at n=1000). Both decoders are nominally unbiased so LPPL should match in expectation. The systematic gap suggests `invariant_multi`'s shared-Gumbel trick has subtle bias on its sampling distribution (worth a separate follow-up).

## Files in this folder

- `multi_draft_qwen_cnn_n1000_L4_B2-4-6-8.json`
- `multi_draft_vicuna_cnn_n1000_L4_B2-4-6-8.json`
- `multi_draft_vicuna_eli5_n1000_L4_B2-4-6-8.json`
- `report.md` (this file)

**Pending:** `multi_draft_qwen_eli5_n1000_*` (currently re-running on GPU 0+1 split between mpfr and invariant; will be merged in once both jobs finish, ~6h ETA from re-launch).
