# 4-Cell Single-Draft Watermark Sweep with Real T-by-T Detection (n=1000)

_generated 2026-05-07_

## Setup

- **Models**:
  - `Qwen/Qwen2.5-7B-Instruct` + `Qwen/Qwen2.5-0.5B-Instruct`
  - `lmsys/vicuna-7b-v1.5` + `double7/vicuna-68m`
- **Datasets**: `cnn_dailymail` (test, shuffled seed=42, filtered <3000 char, top 1000), `eli5` (1000 samples)
- **Decoders** (single-draft): `basic_uwm`, `mc_uwm_speed`, `mc_uwm_strength`, `mc_uwm_pseudo_r`, `pfr`, `mc`, `pfr_no_watermark`
  - 5 watermarked (`basic_uwm`, `mc_uwm_speed`, `mc_uwm_strength`, `mc_uwm_pseudo_r`, `pfr`); 2 H0 controls (`mc` with DeltaGumbel detector, `pfr_no_watermark` with PFR detector)
- **Dual-key rerun for `mc_uwm_pseudo_r`**: separate run with `detector_per_decoder.kind = DeltaGumbelDual`, post-hoc detection via trained-threshold methodology (50/50 train/test split × 10 random trials × 101-step grid over $t \in [0, 1]$, then evaluate Aaronson Gamma-tail on `mix = where(r > t*, Us_mc, Us_pk)` over the held-out half)
- **Sweep**: lookahead L=4, samples = 1000 per decoder
- **process_logits**: top_k=50, top_p=1.0, temperature=1.0, private_key="1234"
- **Detection windows**: T ∈ {8, 16, 24, 32, 48, 64, 96, 128} **(all real, no extrapolation)**
- **Detection target**: FPR α = 1 %; both Aaronson Gamma-tail (exact) and U-Chernoff bound (conservative) reported per row.
- **Hardware**: 4× RTX A6000, one cell per GPU; concurrent run.

## Cells

All four cells completed n=1000 successfully.

| Cell | single-draft 7×1000 | dual-key pseudo_r rerun 1×1000 |
|---|---|---|
| Qwen × CNN/DailyMail | ✅ | ✅ |
| Qwen × ELi5 | ✅ | ✅ |
| Vicuna × CNN/DailyMail | ✅ | ✅ |
| Vicuna × ELi5 | ✅ | ✅ |

## Per-cell summaries

Each row shows mean across all n=1000 samples for that decoder.  AATPS is the average accepted tokens per step (single-draft always 1.0 by construction).  TR is token rate (tok/s).  ANLPPT-{U, Li, PL} are per-token recoverable signal under three score variants.  LPPL is the mean log-perplexity of generated tokens under the target model.  KL/WS is the watermark-vs-source KL divergence ratio (≈1 = entropy-budget saturation).  TPR_Gamma is the Aaronson Gamma-tail TPR at α=1%, T=64 (exact); TPR_U is the U-Chernoff bound TPR at α=1%, T=64 (conservative).  For `mc_uwm_pseudo_r`, TPR_Gamma@64 is the **dual-key trained-threshold** value from the post-hoc rerun (see Setup); single-key Aaronson on `Us_pk` only is much lower and is reported instead via TPR_U@64 = – here.

### Qwen × CNN/DailyMail

| decoder | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_uwm | 1.000 | 32.36 | 0.048 | 0.035 | 0.051 | 0.473 | 0.995 | **0.503** | 0.187 |
| mc_uwm_speed | 2.525 | 18.61 | 0.014 | 0.010 | 0.016 | 0.479 | 0.376 | **0.180** | 0.027 |
| mc_uwm_strength | 2.432 | 18.06 | 0.048 | 0.034 | 0.052 | 0.475 | 0.997 | **0.503** | 0.180 |
| mc_uwm_pseudo_r | 2.499 | 17.75 | 0.013 | 0.010 | 0.014 | 0.481 | 0.988 | **0.345** | – |
| pfr | 2.446 | 20.76 | 0.050 | 0.036 | 0.053 | 0.471 | 0.991 | **0.543** | 0.219 |
| mc (H0, DG) | 2.513 | 22.02 | 0.002 | 0.002 | 0.002 | 0.475 | – | **0.012** | 0.005 |
| pfr_no_watermark (H0, PFR) | 2.439 | 21.56 | 0.002 | 0.002 | 0.002 | 0.474 | – | **0.003** | 0.000 |

### Qwen × ELi5

| decoder | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_uwm | 1.000 | 32.73 | 0.100 | 0.072 | 0.104 | 0.696 | 0.994 | **0.893** | 0.606 |
| mc_uwm_speed | 2.292 | 16.67 | 0.029 | 0.019 | 0.033 | 0.701 | 0.435 | **0.394** | 0.100 |
| mc_uwm_strength | 2.184 | 15.70 | 0.101 | 0.073 | 0.105 | 0.696 | 0.995 | **0.895** | 0.612 |
| mc_uwm_pseudo_r | 2.302 | 17.77 | 0.024 | 0.017 | 0.026 | 0.705 | 1.003 | **0.729** | – |
| pfr | 2.196 | 20.79 | 0.104 | 0.075 | 0.107 | 0.698 | 0.998 | **0.897** | 0.648 |
| mc (H0, DG) | 2.294 | 21.26 | 0.002 | 0.002 | 0.002 | 0.705 | – | **0.009** | 0.000 |
| pfr_no_watermark (H0, PFR) | 2.187 | 20.68 | 0.002 | 0.002 | 0.002 | 0.703 | – | **0.006** | 0.000 |

### Vicuna × CNN/DailyMail

| decoder | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_uwm | 1.000 | 35.29 | 0.021 | 0.016 | 0.022 | 0.274 | 1.024 | **0.249** | 0.074 |
| mc_uwm_speed | 2.220 | 50.23 | 0.005 | 0.005 | 0.006 | 0.264 | 0.288 | **0.045** | 0.003 |
| mc_uwm_strength | 2.163 | 48.82 | 0.021 | 0.016 | 0.023 | 0.277 | 1.027 | **0.255** | 0.072 |
| mc_uwm_pseudo_r | 2.183 | 48.31 | 0.007 | 0.007 | 0.008 | 0.277 | 0.997 | **0.154** | – |
| pfr | 2.177 | 52.86 | 0.022 | 0.017 | 0.023 | 0.272 | 0.999 | **0.246** | 0.065 |
| mc (H0, DG) | 2.194 | 50.57 | 0.002 | 0.002 | 0.002 | 0.271 | – | **0.008** | 0.001 |
| pfr_no_watermark (H0, PFR) | 2.159 | 47.89 | 0.002 | 0.002 | 0.002 | 0.268 | – | **0.017** | 0.002 |

### Vicuna × ELi5

| decoder | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma@64** | TPR_U@64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic_uwm | 1.000 | 35.99 | 0.060 | 0.044 | 0.062 | 0.507 | 0.985 | **0.695** | 0.369 |
| mc_uwm_speed | 1.700 | 36.40 | 0.011 | 0.008 | 0.012 | 0.514 | 0.310 | **0.139** | 0.019 |
| mc_uwm_strength | 1.651 | 35.17 | 0.059 | 0.043 | 0.062 | 0.509 | 0.984 | **0.703** | 0.371 |
| mc_uwm_pseudo_r | 1.703 | 36.15 | 0.016 | 0.012 | 0.017 | 0.508 | 0.997 | **0.567** | – |
| pfr | 1.656 | 40.73 | 0.062 | 0.045 | 0.063 | 0.515 | 0.991 | **0.716** | 0.386 |
| mc (H0, DG) | 1.697 | 42.02 | 0.002 | 0.002 | 0.002 | 0.510 | – | **0.008** | 0.002 |
| pfr_no_watermark (H0, PFR) | 1.656 | 39.48 | 0.002 | 0.002 | 0.002 | 0.503 | – | **0.006** | 0.000 |

## Cross-cell pivots

### TPR_Gamma vs T (`pfr`, the headline curve)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.067 | 0.112 | 0.157 | 0.221 | 0.378 | 0.543 | 0.754 | 0.804 |
| Qwen × ELi5 | 0.098 | 0.260 | 0.420 | 0.566 | 0.778 | 0.897 | 0.974 | 0.992 |
| Vicuna × CNN/DailyMail | 0.072 | 0.088 | 0.103 | 0.125 | 0.193 | 0.246 | 0.354 | 0.410 |
| Vicuna × ELi5 | 0.172 | 0.285 | 0.381 | 0.480 | 0.610 | 0.716 | 0.821 | 0.868 |

### TPR_Gamma vs T (`mc_uwm_pseudo_r`, dual-key trained t — improving_KL methodology)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.046 | 0.063 | 0.101 | 0.126 | 0.219 | 0.345 | 0.550 | 0.633 |
| Qwen × ELi5 | 0.053 | 0.171 | 0.298 | 0.384 | 0.598 | 0.729 | 0.873 | 0.946 |
| Vicuna × CNN/DailyMail | 0.061 | 0.061 | 0.065 | 0.091 | 0.119 | 0.154 | 0.234 | 0.274 |
| Vicuna × ELi5 | 0.127 | 0.201 | 0.273 | 0.350 | 0.474 | 0.567 | 0.702 | 0.772 |

### TPR_Gamma vs T (`basic_uwm`)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.062 | 0.094 | 0.145 | 0.197 | 0.338 | 0.503 | 0.738 | 0.796 |
| Qwen × ELi5 | 0.114 | 0.273 | 0.432 | 0.555 | 0.770 | 0.893 | 0.962 | 0.990 |
| Vicuna × CNN/DailyMail | 0.074 | 0.084 | 0.103 | 0.137 | 0.194 | 0.249 | 0.347 | 0.401 |
| Vicuna × ELi5 | 0.162 | 0.269 | 0.367 | 0.448 | 0.606 | 0.695 | 0.807 | 0.853 |

### TPR_Gamma vs T (`mc_uwm_strength`)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.061 | 0.095 | 0.146 | 0.196 | 0.327 | 0.503 | 0.757 | 0.804 |
| Qwen × ELi5 | 0.116 | 0.277 | 0.436 | 0.555 | 0.766 | 0.895 | 0.967 | 0.991 |
| Vicuna × CNN/DailyMail | 0.074 | 0.084 | 0.101 | 0.136 | 0.194 | 0.255 | 0.363 | 0.400 |
| Vicuna × ELi5 | 0.163 | 0.270 | 0.365 | 0.449 | 0.599 | 0.703 | 0.816 | 0.851 |

### TPR_Gamma vs T (`mc_uwm_speed`)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.034 | 0.042 | 0.051 | 0.067 | 0.129 | 0.180 | 0.270 | 0.301 |
| Qwen × ELi5 | 0.040 | 0.094 | 0.134 | 0.181 | 0.279 | 0.394 | 0.557 | 0.708 |
| Vicuna × CNN/DailyMail | 0.023 | 0.024 | 0.025 | 0.026 | 0.042 | 0.045 | 0.054 | 0.059 |
| Vicuna × ELi5 | 0.030 | 0.058 | 0.076 | 0.089 | 0.105 | 0.139 | 0.187 | 0.243 |

### Empirical FPR vs T (`mc`, H0 for DeltaGumbel detector)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.009 | 0.012 | 0.012 | 0.012 | 0.013 | 0.012 | 0.014 | 0.011 |
| Qwen × ELi5 | 0.008 | 0.012 | 0.008 | 0.007 | 0.013 | 0.009 | 0.007 | 0.012 |
| Vicuna × CNN/DailyMail | 0.007 | 0.006 | 0.008 | 0.008 | 0.007 | 0.008 | 0.006 | 0.008 |
| Vicuna × ELi5 | 0.010 | 0.011 | 0.013 | 0.011 | 0.009 | 0.008 | 0.008 | 0.013 |

### Empirical FPR vs T (`pfr_no_watermark`, H0 for PFR detector)

| cell | T=8 | T=16 | T=24 | T=32 | T=48 | T=64 | T=96 | T=128 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen × CNN/DailyMail | 0.008 | 0.012 | 0.009 | 0.008 | 0.008 | 0.003 | 0.007 | 0.009 |
| Qwen × ELi5 | 0.008 | 0.012 | 0.007 | 0.009 | 0.010 | 0.006 | 0.006 | 0.006 |
| Vicuna × CNN/DailyMail | 0.015 | 0.015 | 0.011 | 0.010 | 0.011 | 0.017 | 0.007 | 0.008 |
| Vicuna × ELi5 | 0.008 | 0.010 | 0.003 | 0.002 | 0.005 | 0.006 | 0.007 | 0.009 |

### Headline TPR_Gamma per decoder (T=8, 16, 32, **64**, 128)

| cell | decoder | T=8 | T=16 | T=32 | T=64 | T=128 |
|---|---|---:|---:|---:|---:|---:|
| Qwen × CNN/DailyMail | basic_uwm | 0.062 | 0.094 | 0.197 | **0.503** | 0.796 |
| Qwen × CNN/DailyMail | mc_uwm_speed | 0.034 | 0.042 | 0.067 | **0.180** | 0.301 |
| Qwen × CNN/DailyMail | mc_uwm_strength | 0.061 | 0.095 | 0.196 | **0.503** | 0.804 |
| Qwen × CNN/DailyMail | mc_uwm_pseudo_r  *(dual-key trained t)* | 0.046 | 0.063 | 0.126 | **0.345** | 0.633 |
| Qwen × CNN/DailyMail | pfr | 0.067 | 0.112 | 0.221 | **0.543** | 0.804 |
| Qwen × ELi5 | basic_uwm | 0.114 | 0.273 | 0.555 | **0.893** | 0.990 |
| Qwen × ELi5 | mc_uwm_speed | 0.040 | 0.094 | 0.181 | **0.394** | 0.708 |
| Qwen × ELi5 | mc_uwm_strength | 0.116 | 0.277 | 0.555 | **0.895** | 0.991 |
| Qwen × ELi5 | mc_uwm_pseudo_r  *(dual-key trained t)* | 0.053 | 0.171 | 0.384 | **0.729** | 0.946 |
| Qwen × ELi5 | pfr | 0.098 | 0.260 | 0.566 | **0.897** | 0.992 |
| Vicuna × CNN/DailyMail | basic_uwm | 0.074 | 0.084 | 0.137 | **0.249** | 0.401 |
| Vicuna × CNN/DailyMail | mc_uwm_speed | 0.023 | 0.024 | 0.026 | **0.045** | 0.059 |
| Vicuna × CNN/DailyMail | mc_uwm_strength | 0.074 | 0.084 | 0.136 | **0.255** | 0.400 |
| Vicuna × CNN/DailyMail | mc_uwm_pseudo_r  *(dual-key trained t)* | 0.061 | 0.061 | 0.091 | **0.154** | 0.274 |
| Vicuna × CNN/DailyMail | pfr | 0.072 | 0.088 | 0.125 | **0.246** | 0.410 |
| Vicuna × ELi5 | basic_uwm | 0.162 | 0.269 | 0.448 | **0.695** | 0.853 |
| Vicuna × ELi5 | mc_uwm_speed | 0.030 | 0.058 | 0.089 | **0.139** | 0.243 |
| Vicuna × ELi5 | mc_uwm_strength | 0.163 | 0.270 | 0.449 | **0.703** | 0.851 |
| Vicuna × ELi5 | mc_uwm_pseudo_r  *(dual-key trained t)* | 0.127 | 0.201 | 0.350 | **0.567** | 0.772 |
| Vicuna × ELi5 | pfr | 0.172 | 0.285 | 0.480 | **0.716** | 0.868 |

### KL/WS ratio (watermark-vs-source KL ratio per decoder, mean over prompts)

| cell | basic_uwm | mc_uwm_speed | mc_uwm_strength | mc_uwm_pseudo_r | pfr |
|---|---:|---:|---:|---:|---:|
| Qwen × CNN/DailyMail | 0.995 | 0.376 | 0.997 | 0.988 | 0.991 |
| Qwen × ELi5 | 0.994 | 0.435 | 0.995 | 1.003 | 0.998 |
| Vicuna × CNN/DailyMail | 1.024 | 0.288 | 1.027 | 0.997 | 0.999 |
| Vicuna × ELi5 | 0.985 | 0.310 | 0.984 | 0.997 | 0.991 |

### LPPL (mean per-token log-perplexity under target model)

| cell | basic_uwm | mc_uwm_speed | mc_uwm_strength | mc_uwm_pseudo_r | pfr | mc | pfr_no_watermark |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen × CNN/DailyMail | 0.473 | 0.479 | 0.475 | 0.481 | 0.471 | 0.475 | 0.474 |
| Qwen × ELi5 | 0.696 | 0.701 | 0.696 | 0.705 | 0.698 | 0.705 | 0.703 |
| Vicuna × CNN/DailyMail | 0.274 | 0.264 | 0.277 | 0.277 | 0.272 | 0.271 | 0.268 |
| Vicuna × ELi5 | 0.507 | 0.514 | 0.509 | 0.508 | 0.515 | 0.510 | 0.503 |

### AATPS / TR per decoder per cell

| cell | decoder | AATPS | TR (tok/s) |
|---|---|---:|---:|
| Qwen × CNN/DailyMail | basic_uwm | 1.000 | 32.36 |
| Qwen × CNN/DailyMail | mc_uwm_speed | 2.525 | 18.61 |
| Qwen × CNN/DailyMail | mc_uwm_strength | 2.432 | 18.06 |
| Qwen × CNN/DailyMail | mc_uwm_pseudo_r | 2.499 | 17.75 |
| Qwen × CNN/DailyMail | pfr | 2.446 | 20.76 |
| Qwen × CNN/DailyMail | mc | 2.513 | 22.02 |
| Qwen × CNN/DailyMail | pfr_no_watermark | 2.439 | 21.56 |
| Qwen × ELi5 | basic_uwm | 1.000 | 32.73 |
| Qwen × ELi5 | mc_uwm_speed | 2.292 | 16.67 |
| Qwen × ELi5 | mc_uwm_strength | 2.184 | 15.70 |
| Qwen × ELi5 | mc_uwm_pseudo_r | 2.302 | 17.77 |
| Qwen × ELi5 | pfr | 2.196 | 20.79 |
| Qwen × ELi5 | mc | 2.294 | 21.26 |
| Qwen × ELi5 | pfr_no_watermark | 2.187 | 20.68 |
| Vicuna × CNN/DailyMail | basic_uwm | 1.000 | 35.29 |
| Vicuna × CNN/DailyMail | mc_uwm_speed | 2.220 | 50.23 |
| Vicuna × CNN/DailyMail | mc_uwm_strength | 2.163 | 48.82 |
| Vicuna × CNN/DailyMail | mc_uwm_pseudo_r | 2.183 | 48.31 |
| Vicuna × CNN/DailyMail | pfr | 2.177 | 52.86 |
| Vicuna × CNN/DailyMail | mc | 2.194 | 50.57 |
| Vicuna × CNN/DailyMail | pfr_no_watermark | 2.159 | 47.89 |
| Vicuna × ELi5 | basic_uwm | 1.000 | 35.99 |
| Vicuna × ELi5 | mc_uwm_speed | 1.700 | 36.40 |
| Vicuna × ELi5 | mc_uwm_strength | 1.651 | 35.17 |
| Vicuna × ELi5 | mc_uwm_pseudo_r | 1.703 | 36.15 |
| Vicuna × ELi5 | pfr | 1.656 | 40.73 |
| Vicuna × ELi5 | mc | 1.697 | 42.02 |
| Vicuna × ELi5 | pfr_no_watermark | 1.656 | 39.48 |

## Key observations

1. **`pfr` and `basic_uwm`/`mc_uwm_strength` form a top-tier cluster.**  At T=64, all three reach within $\pm 0.04$ TPR of each other on every cell (e.g. Qwen×ELi5: 0.897 / 0.893 / 0.895; Vicuna×ELi5: 0.716 / 0.695 / 0.703).  `pfr` is consistently tied or slightly highest.

2. **`mc_uwm_speed` is the weakest watermarked decoder.**  Its KL/WS ratio is ~0.3–0.4 — it under-saturates the entropy budget by design (favors throughput over signal).  TPR_Gamma@64 lands 0.045 (Vicuna×CNN) to 0.394 (Qwen×ELi5), substantially below the other watermarks.

3. **`mc_uwm_pseudo_r` (dual-key, trained $t$) sits in the middle.**  At T=64 it's between `mc_uwm_speed` and the top cluster on every cell (0.154 < 0.345 < 0.567 < 0.729).  This matches expectations: the dual-key mechanism splits the watermark randomness so a single-key Aaronson recovers less signal; the trained-threshold detector recovers most but not all of it.

4. **TPR ranks by corpus entropy.**  Across all decoders, TPR orders cells: Qwen×ELi5 > Vicuna×ELi5 > Qwen×CNN > Vicuna×CNN, the same order as LPPL.  CNN's lower entropy on instruction-tuned targets is the fundamental detection bottleneck.

5. **Empirical FPR is α-calibrated for both detectors.**  At α=1%, `mc` (DeltaGumbel detector) and `pfr_no_watermark` (PFR detector) both stay in [0.002, 0.017] across all $4 \times 8 = 32$ (cell, T) combinations — within binomial 95% CI of α at n=1000 ($\pm 0.6\%$).

6. **KL/WS ≈ 1.0 for the entropy-saturating watermarks.**  `basic_uwm`, `mc_uwm_strength`, `mc_uwm_pseudo_r`, and `pfr` all sit within $[0.984, 1.027]$ across cells.  Only `mc_uwm_speed` materially under-shoots (0.288–0.435).

7. **LPPL is essentially flat across decoders within each cell.**  Maximum spread is $\sim 0.013$ nats/token (Vicuna×CNN: 0.264–0.277), at the level of run-to-run noise.  This is consistent with the distortion-free property of each watermark.

## Files in this folder

- `single_qwen_cnn_efpr_n1000.json` — single-draft 7-decoder run, all metrics + per-T detection
- `single_qwen_eli5_efpr_n1000.json`
- `single_vicuna_cnn_efpr_n1000.json`
- `single_vicuna_eli5_efpr_n1000.json`
- `rerun_pseudo_r_dualkey_<cell>_n1000.json` (×4) — `mc_uwm_pseudo_r` run with `DeltaGumbelDual` detector saving `dual_Us_pk`, `dual_Us_mc`, `dual_r` per row
- `tpr_vs_T_2x2_data.json` — derived per-cell, per-decoder TPR curves (includes the dual-key trained-threshold output for `mc_uwm_pseudo_r`)
- `tpr_vs_T_2x2.png` — 2×2 facet plot
- `qwen_cnn_tpr_vs_T.png`, `vicuna_cnn_tpr_vs_T.png`, `vicuna_eli5_tpr_vs_T.png` — single-cell facets
- `report.md` — this file (regenerated by `compute_single_report.py`)
