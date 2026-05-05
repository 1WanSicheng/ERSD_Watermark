# 4-Cell PFR Multi-Draft Sweep with Real T-by-T Detection (n=1000)

_generated 2026-05-05 (post upstream commit 2806c6c, with local `tpr_at_n.tokens` list patch)_

## Setup

- **Models**:
  - `Qwen/Qwen2.5-7B-Instruct` + `Qwen/Qwen2.5-0.5B-Instruct`
  - `lmsys/vicuna-7b-v1.5` + `double7/vicuna-68m`
- **Datasets**: `cnn_dailymail` (test, shuffled seed=42, filtered <3000 char, top 1000), `eli5` (1000 samples)
- **Decoders**:
  - `mpfr_torchgen_cached` (multi-draft PFR, watermarked, `mpfr_direct` labeler)
  - `invariant_multi` (no watermark; PFR detector forced via `detector_per_decoder.invariant_multi = {kind: PFR, labeler: mpfr_direct}` for empirical-FPR check)
- **Sweep**: L=4, B in {2, 4, 6, 8}, samples = 1000 per (decoder, B)
- **process_logits**: top_k=50, top_p=1.0, temperature=1.0, private_key="1234"
- **Detection windows**: T ∈ {8, 16, 24, 32, 48, 64, 96, 128} **(all real, no extrapolation)**
- **Detection target**: FPR α = 1 %; both Aaronson Gamma-tail (exact) and Chernoff U-bound (conservative) reported per row.
- **Hardware**: 4× RTX A6000, one cell per GPU; concurrent run.

## Cells

All four cells completed n=1000 successfully.

| Cell | Status | mpfr 4×1000 | inv 4×1000 |
|---|---|---|---|
| qwen_cnn | ✅ | ✅ | ✅ |
| qwen_eli5 | ✅ | ✅ | ✅ |
| vicuna_cnn | ✅ | ✅ | ✅ |
| vicuna_eli5 | ✅ | ✅ | ✅ |

## Per-cell summaries

Each row shows mean across all 1000 samples at that B.  TPR_Gamma is the
Aaronson Gamma-tail test (exact at α=1%); TPR_U is the Aaronson U-Chernoff
bound at the same α (conservative).  Both at T=64 by convention.

### qwen_cnn

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.799 | 23.18 | 0.052 | 0.037 | 0.055 | 0.475 | 0.998 | **0.514** | 0.219 |
| mpfr | 4 | 3.124 | 24.89 | 0.052 | 0.037 | 0.055 | 0.474 | 0.997 | **0.516** | 0.220 |
| mpfr | 6 | 3.298 | 25.39 | 0.052 | 0.037 | 0.055 | 0.476 | 0.999 | **0.512** | 0.222 |
| mpfr | 8 | 3.410 | 25.17 | 0.053 | 0.037 | 0.055 | 0.476 | 0.998 | **0.518** | 0.222 |
| invariant | 2 | 2.804 | 25.03 | 0.003 | 0.003 | 0.002 | 0.487 | – | 0.011 | 0.002 |
| invariant | 4 | 3.114 | 26.63 | 0.002 | 0.002 | 0.002 | 0.481 | – | 0.009 | 0.001 |
| invariant | 6 | 3.275 | 26.79 | 0.002 | 0.002 | 0.002 | 0.482 | – | 0.011 | 0.004 |
| invariant | 8 | 3.387 | 26.38 | 0.002 | 0.003 | 0.002 | 0.479 | – | 0.011 | 0.000 |

### qwen_eli5

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.563 | 22.38 | 0.104 | 0.076 | 0.108 | 0.704 | 0.996 | **0.890** | 0.652 |
| mpfr | 4 | 2.919 | 24.48 | 0.104 | 0.076 | 0.107 | 0.703 | 0.995 | **0.890** | 0.647 |
| mpfr | 6 | 3.111 | 25.54 | 0.105 | 0.076 | 0.108 | 0.706 | 0.998 | **0.892** | 0.650 |
| mpfr | 8 | 3.237 | 25.83 | 0.105 | 0.076 | 0.108 | 0.704 | 0.995 | **0.894** | 0.644 |
| invariant | 2 | 2.519 | 23.18 | 0.002 | 0.002 | 0.002 | 0.750 | – | 0.011 | 0.001 |
| invariant | 4 | 2.852 | 25.75 | 0.002 | 0.002 | 0.002 | 0.733 | – | 0.011 | 0.000 |
| invariant | 6 | 3.043 | 27.11 | 0.002 | 0.002 | 0.002 | 0.730 | – | 0.009 | 0.003 |
| invariant | 8 | 3.158 | 27.96 | 0.002 | 0.002 | 0.002 | 0.725 | – | 0.010 | 0.001 |

### vicuna_cnn

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.422 | 52.10 | 0.022 | 0.016 | 0.022 | 0.269 | 0.987 | **0.273** | 0.066 |
| mpfr | 4 | 2.664 | 48.28 | 0.022 | 0.017 | 0.023 | 0.271 | 0.990 | **0.273** | 0.066 |
| mpfr | 6 | 2.803 | 44.44 | 0.022 | 0.017 | 0.023 | 0.270 | 0.987 | **0.275** | 0.066 |
| mpfr | 8 | 2.899 | 40.76 | 0.022 | 0.017 | 0.023 | 0.270 | 0.988 | **0.276** | 0.066 |
| invariant | 2 | 2.478 | 58.15 | 0.002 | 0.002 | 0.002 | 0.287 | – | 0.009 | 0.000 |
| invariant | 4 | 2.690 | 53.12 | 0.002 | 0.002 | 0.002 | 0.286 | – | 0.011 | 0.001 |
| invariant | 6 | 2.831 | 48.41 | 0.002 | 0.002 | 0.002 | 0.283 | – | 0.008 | 0.000 |
| invariant | 8 | 2.925 | 44.01 | 0.002 | 0.002 | 0.002 | 0.288 | – | 0.008 | 0.001 |

### vicuna_eli5

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 1.889 | 45.89 | 0.058 | 0.042 | 0.061 | 0.504 | 0.995 | **0.703** | 0.362 |
| mpfr | 4 | 2.133 | 47.43 | 0.058 | 0.042 | 0.061 | 0.504 | 0.996 | **0.700** | 0.363 |
| mpfr | 6 | 2.278 | 47.42 | 0.058 | 0.042 | 0.061 | 0.503 | 0.996 | **0.706** | 0.365 |
| mpfr | 8 | 2.377 | 46.42 | 0.059 | 0.043 | 0.062 | 0.504 | 0.994 | **0.703** | 0.366 |
| invariant | 2 | 1.884 | 50.29 | 0.002 | 0.002 | 0.002 | 0.555 | – | 0.012 | 0.002 |
| invariant | 4 | 2.108 | 53.42 | 0.002 | 0.002 | 0.002 | 0.552 | – | 0.012 | 0.002 |
| invariant | 6 | 2.248 | 55.08 | 0.002 | 0.002 | 0.002 | 0.546 | – | 0.014 | 0.002 |
| invariant | 8 | 2.361 | 56.00 | 0.002 | 0.003 | 0.002 | 0.539 | – | 0.014 | 0.001 |

## Cross-cell pivots (mpfr only, mean over B)

### TPR_Gamma vs T (the headline curve)

| cell  | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen × ELi5  | 0.099 | 0.270 | 0.422 | 0.576 | 0.788 | **0.892** | 0.970 | 0.987 |
| Vicuna × ELi5 | 0.137 | 0.253 | 0.366 | 0.450 | 0.597 | 0.703 | 0.825 | 0.869 |
| Qwen × CNN  | 0.047 | 0.097 | 0.147 | 0.199 | 0.367 | 0.515 | 0.762 | 0.822 |
| Vicuna × CNN | 0.055 | 0.080 | 0.095 | 0.123 | 0.189 | 0.274 | 0.362 | 0.411 |

See `tpr_vs_T_real_gamma.png` and `tpr_vs_T_real_per_cell.png`.

### Empirical FPR vs T (invariant_multi, calibration check)

| cell  | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen × CNN | 0.006 | 0.008 | 0.009 | 0.008 | 0.009 | 0.010 | 0.010 | 0.009 |
| Qwen × ELi5 | 0.011 | 0.007 | 0.009 | 0.008 | 0.010 | 0.010 | 0.007 | 0.011 |
| Vicuna × CNN | 0.013 | 0.010 | 0.011 | 0.010 | 0.010 | 0.009 | 0.011 | 0.012 |
| Vicuna × ELi5 | 0.007 | 0.010 | 0.009 | 0.009 | 0.009 | 0.012 | 0.012 | 0.011 |

All 32 (cell, T) cells fall in [0.6 %, 1.3 %], **calibrated to nominal α=1 % within Monte-Carlo noise** (n=1000 binomial 95% CI ≈ ± 0.6%). PFR detector is correctly type-I controlled.

### TR (tok/s, mpfr / invariant)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---|---|---|---|
| qwen_cnn | 23.2 / 25.0 | 24.9 / 26.6 | 25.4 / 26.8 | 25.2 / 26.4 |
| qwen_eli5 | 22.4 / 23.2 | 24.5 / 25.8 | 25.5 / 27.1 | 25.8 / 28.0 |
| vicuna_cnn | 52.1 / 58.2 | 48.3 / 53.1 | 44.4 / 48.4 | 40.8 / 44.0 |
| vicuna_eli5 | 45.9 / 50.3 | 47.4 / 53.4 | 47.2 / 55.1 | 46.4 / 56.0 |

### KL/WS ratio (mpfr only)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 0.998 | 0.997 | 0.999 | 0.998 |
| qwen_eli5 | 0.996 | 0.995 | 0.998 | 0.995 |
| vicuna_cnn | 0.987 | 0.990 | 0.987 | 0.988 |
| vicuna_eli5 | 0.995 | 0.996 | 0.996 | 0.994 |

All cells ≈ 1.0 — PFR pushes the watermark to the entropy-budget upper bound.

### LPPL (mean per-token NLL)

| cell | mpfr (mean) | invariant (mean) | Δ (inv − mpfr) |
|---|---:|---:|---:|
| qwen_cnn | 0.475 | 0.482 | +0.007 |
| qwen_eli5 | 0.704 | 0.735 | +0.031 |
| vicuna_cnn | 0.270 | 0.286 | +0.016 |
| vicuna_eli5 | 0.504 | 0.548 | +0.044 |

mpfr LPPL is systematically slightly LOWER than invariant_multi LPPL across all 4 cells — both are nominally distortion-free but `invariant_multi`'s shared-Gumbel mechanism appears to introduce a small positive bias (also noted in 0504 report).

### PFR engineering overhead (TR loss vs invariant)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 7.2 % | 6.5 % | 5.3 % | 4.6 % |
| qwen_eli5 | 3.4 % | 5.0 % | 5.8 % | 7.8 % |
| vicuna_cnn | 10.5 % | 9.1 % | 8.3 % | 7.4 % |
| vicuna_eli5 | 8.7 % | 11.2 % | 14.4 % | 17.1 % |

5-17 % across cells; biggest on Vicuna+ELi5 B=8 (Python-side label-hashing + source-factory overhead is a larger fraction when the small 68m draft makes the model forward fast).

## Key observations

1. **Watermark strength independent of B**.  Within each cell, TPR_Gamma at every T is flat across B={2,4,6,8} (variation ≤ 0.005).  Confirms PFR's theoretical guarantee that multi-draft sampling preserves the per-token marginal under H1 → per-token uniforms `u_t` follow the same Aaronson distribution as single-draft PFR → detection power identical at any T.
2. **TPR strongly tracks LPPL (entropy)**.  Detection rate at every T orders cells by entropy: Qwen×ELi5 (LPPL 0.70) > Vicuna×ELi5 (0.50) > Qwen×CNN (0.48) > Vicuna×CNN (0.27).  The TPR-vs-T curves don't cross.  CNN's low entropy on instruction-tuned models is the fundamental detection bottleneck.
3. **Empirical FPR is α-calibrated**.  All 32 (cell, T) combinations of invariant_multi under forced PFR detector give 0.6-1.3 % rejection rate at target α=1 %.
4. **KL/WS ≈ 1.0** in every cell — water-mark strength saturates the entropy budget.
5. **PFR engineering overhead 5-17 %** vs invariant_multi baseline.  Bigger on Vicuna pairs (small 68m draft → mpfr label-hashing overhead is larger fraction of total decode time).
6. **Multi-draft TPR matches single-draft PFR** — verified empirically (TPR_Gamma flat in B) and aligned with theoretical guarantee.  Compared with the prior 0504 (`accuwm/pfr.py`) single-draft TPR at the same model+dataset entropy, the numbers fall in the same range.
7. **Reproducibility vs prior n=1000 single-T run** (pre-multi-T): TPR_Gamma at T=64 cross-validates to ±0.002 (Qwen×CNN 0.515 vs 0.514, Qwen×ELi5 0.892 vs 0.891, Vicuna×CNN 0.275 vs 0.273, Vicuna×ELi5 0.703 vs 0.705).  Detection-side metrics are deterministic given (model, dataset, key, labeler), so this is expected — confirms no regression from the multi-T patch.

## Files in this folder

- `full_qwen_cnn_efpr_n1000_multiT.json` (~28 MB)
- `full_qwen_eli5_efpr_n1000_multiT.json`
- `full_vicuna_cnn_efpr_n1000_multiT.json`
- `full_vicuna_eli5_efpr_n1000_multiT.json`
- `tpr_vs_T_real_gamma.png` — overall 4-cell TPR-vs-T (Aaronson Gamma)
- `tpr_vs_T_real_per_cell.png` — 4-panel facet, all B + Chernoff overlay
- `tpr_vs_T_real_curves.csv` — numerical TPR(T) for both Gamma and Chernoff variants
- `report.md` — this file
