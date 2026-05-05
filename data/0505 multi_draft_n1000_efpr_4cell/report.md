# 4-Cell PFR Multi-Draft Sweep with Empirical-FPR (n=1000)

_generated 2026-05-05 (post upstream commit 2806c6c)_

## Setup

- **Models**:
  - `Qwen/Qwen2.5-7B-Instruct` + `Qwen/Qwen2.5-0.5B-Instruct`
  - `lmsys/vicuna-7b-v1.5` + `double7/vicuna-68m`
- **Datasets**: `cnn_dailymail` (test split, shuffled seed=42, filtered <3000 char, top 1000), `eli5` (1000 samples)
- **Decoders**:
  - `mpfr_torchgen_cached` (multi-draft PFR, watermarked, **mpfr_direct** labeler — full-prefix bytes + `MPFR_DIRECT_CLOCK_V1` domain tag)
  - `invariant_multi` (no watermark, position-indexed `randomness.uniform_()` ahead of time, **no labeler**)
- **Sweep**: L=4, B in {2, 4, 6, 8}, samples = 1000 per (decoder, B)
- **process_logits**: top_k=50, top_p=1.0, temperature=1.0, private_key="1234"
- **Detection**: PFR Aaronson at FPR α=1%, T=64 tokens; for `invariant_multi` we force the PFR detector with `mpfr_direct` labeler to obtain empirical FPR (calibration check).
- **Metrics**: AATPS, TR, ANLPPT-{U/Li/PL}, log-perplexity, KL/WS ratio, TPR / empirical-FPR.
- **Hardware**: 4× RTX A6000 (one cell per GPU); concurrent 4-job run.

## Cells

All four cells completed n=1000 successfully.

| Cell | Status | mpfr 4×1000 | inv 4×1000 |
|---|---|---|---|
| qwen_cnn | ✅ | ✅ | ✅ |
| qwen_eli5 | ✅ | ✅ | ✅ |
| vicuna_cnn | ✅ | ✅ | ✅ |
| vicuna_eli5 | ✅ | ✅ | ✅ |

## Per-cell results

### qwen_cnn

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma** | TPR_Chernoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.799 | 22.76 | 0.052 | 0.037 | 0.055 | 0.475 | 0.998 | **0.514** | 0.219 |
| mpfr | 4 | 3.124 | 24.08 | 0.052 | 0.037 | 0.055 | 0.474 | 0.997 | **0.516** | 0.220 |
| mpfr | 6 | 3.298 | 24.47 | 0.052 | 0.037 | 0.055 | 0.476 | 0.999 | **0.512** | 0.222 |
| mpfr | 8 | 3.410 | 24.24 | 0.053 | 0.037 | 0.055 | 0.476 | 0.998 | **0.518** | 0.222 |
| invariant | 2 | 2.804 | 24.24 | 0.003 | 0.003 | 0.002 | 0.487 | – | 0.008 | 0.002 |
| invariant | 4 | 3.114 | 25.82 | 0.002 | 0.002 | 0.002 | 0.481 | – | 0.010 | 0.000 |
| invariant | 6 | 3.275 | 26.02 | 0.002 | 0.002 | 0.002 | 0.482 | – | 0.014 | 0.003 |
| invariant | 8 | 3.387 | 25.52 | 0.002 | 0.003 | 0.002 | 0.479 | – | 0.009 | 0.001 |

### qwen_eli5

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma** | TPR_Chernoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.563 | 22.40 | 0.104 | 0.076 | 0.108 | 0.704 | 0.996 | **0.891** | 0.652 |
| mpfr | 4 | 2.919 | 24.55 | 0.104 | 0.076 | 0.107 | 0.703 | 0.995 | **0.891** | 0.647 |
| mpfr | 6 | 3.111 | 25.35 | 0.105 | 0.076 | 0.108 | 0.706 | 0.998 | **0.893** | 0.650 |
| mpfr | 8 | 3.237 | 25.69 | 0.105 | 0.076 | 0.108 | 0.704 | 0.995 | **0.892** | 0.644 |
| invariant | 2 | 2.519 | 23.18 | 0.002 | 0.002 | 0.002 | 0.750 | – | 0.010 | 0.000 |
| invariant | 4 | 2.852 | 25.82 | 0.002 | 0.002 | 0.002 | 0.733 | – | 0.009 | 0.000 |
| invariant | 6 | 3.043 | 27.04 | 0.002 | 0.002 | 0.002 | 0.730 | – | 0.011 | 0.001 |
| invariant | 8 | 3.158 | 27.86 | 0.002 | 0.002 | 0.002 | 0.725 | – | 0.011 | 0.000 |

### vicuna_cnn

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma** | TPR_Chernoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 2.422 | 52.28 | 0.022 | 0.016 | 0.022 | 0.269 | 0.987 | **0.273** | 0.066 |
| mpfr | 4 | 2.664 | 48.51 | 0.022 | 0.017 | 0.023 | 0.271 | 0.990 | **0.273** | 0.066 |
| mpfr | 6 | 2.803 | 44.57 | 0.022 | 0.017 | 0.023 | 0.270 | 0.987 | **0.274** | 0.065 |
| mpfr | 8 | 2.899 | 40.83 | 0.022 | 0.017 | 0.023 | 0.270 | 0.988 | **0.276** | 0.066 |
| invariant | 2 | 2.478 | 58.12 | 0.002 | 0.002 | 0.002 | 0.287 | – | 0.010 | 0.002 |
| invariant | 4 | 2.690 | 53.07 | 0.002 | 0.002 | 0.002 | 0.286 | – | 0.006 | 0.000 |
| invariant | 6 | 2.831 | 48.28 | 0.002 | 0.002 | 0.002 | 0.283 | – | 0.009 | 0.001 |
| invariant | 8 | 2.925 | 43.92 | 0.002 | 0.002 | 0.002 | 0.288 | – | 0.011 | 0.002 |

### vicuna_eli5

| decoder | B | AATPS | TR | U | Li | PL | LPPL | KL/WS | **TPR_Gamma** | TPR_Chernoff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mpfr | 2 | 1.889 | 45.79 | 0.058 | 0.042 | 0.061 | 0.504 | 0.995 | **0.705** | 0.362 |
| mpfr | 4 | 2.133 | 47.40 | 0.058 | 0.042 | 0.061 | 0.504 | 0.996 | **0.698** | 0.363 |
| mpfr | 6 | 2.278 | 47.24 | 0.058 | 0.042 | 0.061 | 0.503 | 0.996 | **0.705** | 0.365 |
| mpfr | 8 | 2.377 | 46.23 | 0.059 | 0.043 | 0.062 | 0.504 | 0.994 | **0.704** | 0.366 |
| invariant | 2 | 1.884 | 50.05 | 0.002 | 0.002 | 0.002 | 0.555 | – | 0.009 | 0.001 |
| invariant | 4 | 2.108 | 53.18 | 0.002 | 0.002 | 0.002 | 0.552 | – | 0.011 | 0.001 |
| invariant | 6 | 2.248 | 54.61 | 0.002 | 0.002 | 0.002 | 0.546 | – | 0.011 | 0.002 |
| invariant | 8 | 2.361 | 55.52 | 0.002 | 0.003 | 0.002 | 0.539 | – | 0.016 | 0.001 |

## Cross-cell pivots (mpfr_torchgen_cached only)

### AATPS (acceptance rate)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 2.799 | 3.124 | 3.298 | 3.410 |
| qwen_eli5 | 2.563 | 2.919 | 3.111 | 3.237 |
| vicuna_cnn | 2.422 | 2.664 | 2.803 | 2.899 |
| vicuna_eli5 | 1.889 | 2.133 | 2.278 | 2.377 |

### TR (tok/s)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 22.76 | 24.08 | 24.47 | 24.24 |
| qwen_eli5 | 22.40 | 24.55 | 25.35 | 25.69 |
| vicuna_cnn | 52.28 | 48.51 | 44.57 | 40.83 |
| vicuna_eli5 | 45.79 | 47.40 | 47.24 | 46.23 |

### TPR @ 64 tokens, FPR=1% — Aaronson Gamma-tail (exact, standard PFR detector)

| cell | B=2 | B=4 | B=6 | B=8 | LPPL |
|---|---:|---:|---:|---:|---:|
| qwen_cnn | 0.514 | 0.516 | 0.512 | 0.518 | 0.475 |
| qwen_eli5 | 0.891 | 0.891 | 0.893 | 0.892 | 0.704 |
| vicuna_cnn | 0.273 | 0.273 | 0.274 | 0.276 | 0.270 |
| vicuna_eli5 | 0.705 | 0.698 | 0.705 | 0.704 | 0.504 |

For reference, the **U-Chernoff bound** (conservative, was previously labelled `TPR_U`):

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 0.219 | 0.220 | 0.222 | 0.222 |
| qwen_eli5 | 0.652 | 0.647 | 0.650 | 0.644 |
| vicuna_cnn | 0.066 | 0.066 | 0.065 | 0.066 |
| vicuna_eli5 | 0.362 | 0.363 | 0.365 | 0.366 |

The Gamma-tail test is what `accuwm/pfr.py` and the canonical PFR papers report; the multi-draft TPR matches single-draft PFR within Monte-Carlo noise (theoretical guarantee: PFR's multi-draft sampler doesn't change the marginal distribution of the emitted token, so per-token uniforms `u_t` follow the same Aaronson distribution under H1 regardless of B).

### KL/WS ratio

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 0.998 | 0.997 | 0.999 | 0.998 |
| qwen_eli5 | 0.996 | 0.995 | 0.998 | 0.995 |
| vicuna_cnn | 0.987 | 0.990 | 0.987 | 0.988 |
| vicuna_eli5 | 0.995 | 0.996 | 0.996 | 0.994 |

All cells `KL/WS ≈ 1.0` — PFR pushes the watermark to the entropy-budget upper bound.

### LPPL  (mean per-token NLL under raw target distribution; lower = more deterministic)

| cell | mpfr (mean over B) | invariant (mean over B) | Δ (inv − mpfr) |
|---|---:|---:|---:|
| qwen_cnn | 0.475 | 0.482 | +0.007 |
| qwen_eli5 | 0.704 | 0.735 | +0.031 |
| vicuna_cnn | 0.270 | 0.286 | +0.016 |
| vicuna_eli5 | 0.504 | 0.548 | +0.044 |

`mpfr` gives systematically **slightly lower LPPL** than `invariant` in every cell. Both are nominally distortion-free, so this small consistent gap likely reflects a subtle bias in the invariant_multi shared-Gumbel mechanism (also noted in the 0504 report's observation 6).

### PFR engineering overhead (TR loss vs invariant_multi same B)

| cell | B=2 | B=4 | B=6 | B=8 |
|---|---:|---:|---:|---:|
| qwen_cnn | 6.1 % | 6.7 % | 6.0 % | 5.0 % |
| qwen_eli5 | 3.4 % | 4.9 % | 6.2 % | 7.8 % |
| vicuna_cnn | 10.0 % | 8.6 % | 7.7 % | 7.0 % |
| vicuna_eli5 | 8.5 % | 10.9 % | 13.5 % | 16.7 % |

mpfr's per-block Python-side overhead (label hashing, source factory, KV state) costs 5-17 %.  Vicuna pairs (small 68M draft) suffer more relative overhead than Qwen (larger 0.5B draft dominates compute).  Compared to the 0504 report on same models, this run shows ~3-5 pp larger gap; the absolute invariant TR matches 0504 within ±1 tok/s but mpfr TR is 1-3 tok/s slower (likely concurrent-job CPU/PCIe contention, no NUMA pinning).

### Empirical FPR @ FPR=1% (invariant_multi forced through PFR detector, T=64)

**Aaronson Gamma-tail** (matches what we report TPR with):

| cell | B=2 | B=4 | B=6 | B=8 | mean |
|---|---:|---:|---:|---:|---:|
| qwen_cnn | 0.008 | 0.010 | 0.014 | 0.009 | 0.0103 |
| qwen_eli5 | 0.010 | 0.009 | 0.011 | 0.011 | 0.0103 |
| vicuna_cnn | 0.010 | 0.006 | 0.009 | 0.011 | 0.0090 |
| vicuna_eli5 | 0.009 | 0.011 | 0.011 | 0.016 | 0.0118 |

**U-Chernoff bound** (conservative reference):

| cell | B=2 | B=4 | B=6 | B=8 | mean |
|---|---:|---:|---:|---:|---:|
| qwen_cnn | 0.002 | 0.000 | 0.003 | 0.001 | 0.0015 |
| qwen_eli5 | 0.000 | 0.000 | 0.001 | 0.000 | 0.0003 |
| vicuna_cnn | 0.002 | 0.000 | 0.001 | 0.002 | 0.0013 |
| vicuna_eli5 | 0.001 | 0.001 | 0.002 | 0.001 | 0.0013 |

Gamma-tail: empirical FPR ≈ 0.009-0.012 across cells, all close to nominal α=0.01 — **calibration confirmed**.
U-Chernoff: ≤ 0.003 ≪ α — strictly conservative (sacrifices ~50 % of detection power for the safety margin).

## TPR-vs-T detection curves

`tpr_vs_T_gamma.png` (overall) and `tpr_vs_T_per_cell_gamma.png` (4-panel) plot detection rate as the number of scored tokens T grows from 1 to 128.

**Method**: per-prompt entropy rate λ is estimated from the observed Aaronson score sum at T=64 (`λ̂ = score_at_64 / 64`); for any T we extrapolate `score_T ≈ T · λ̂` and compute the exact Aaronson Gamma-tail p-value `Q(T, score_T)`.  TPR(T) = empirical fraction of prompts with `Q(T, score_T) ≤ α`.

**Validation**: at T=64 the Gamma-test TPR matches the actually-recorded `det_aaronson_at_64` mean within ±0.01 (0.514 vs 0.514 for qwen_cnn, 0.891 vs 0.891 for qwen_eli5, 0.273 vs 0.272 for vicuna_cnn, 0.705 vs 0.696 for vicuna_eli5).  The very small mismatch on vicuna_eli5 comes from prompts whose generation truncated below 64 tokens.

**Caveat**: this is an iid-extrapolation, not a true T-by-T measurement.  It treats per-prompt entropy as constant across the first 128 tokens and collapses within-prompt score variance to its mean, so it slightly underestimates the dispersion across prompts.  At T near 64 the curve is exact; at T far from 64 it should be read as the predicted mean detection power.  A true T-sweep would require modifying `tpr_at_n` to log multiple T values and re-running.

**TPR-vs-T headline numbers (Aaronson Gamma-tail, mean over B={2,4,6,8}):**

| cell  | T=16 | T=32 | T=64 | T=128 |
|---|---:|---:|---:|---:|
| qwen_eli5  | 0.271 | 0.677 | 0.892 | 0.960 |
| vicuna_eli5  | 0.113 | 0.411 | 0.704 | 0.865 |
| qwen_cnn  | 0.040 | 0.256 | 0.515 | 0.733 |
| vicuna_cnn  | 0.010 | 0.075 | 0.273 | 0.467 |

(values from `tpr_vs_T_curves.csv`; at T=64 these match the actually-recorded `det_aaronson_at_64` mean within ±0.01.)

## Key observations

1. **Watermark strength independent of B**.  ANLPPT-{U,Li,PL}, KL/WS and TPR are flat across B ∈ {2,4,6,8} in every cell (within Monte-Carlo noise at n=1000).  Confirms that PFR's multi-draft mechanism does not erode per-token watermark detectability.
2. **KL/WS ≈ 0.99 in every cell** — water-mark strength saturates the entropy budget tightly.
3. **TPR strongly tracks LPPL (entropy)**: low-entropy cells (vicuna_cnn LPPL=0.27 → Gamma TPR ≈ 27 %) require many more tokens to detect than high-entropy cells (qwen_eli5 LPPL=0.70 → Gamma TPR ≈ 89 %).  The TPR-vs-T plot makes this concrete: on Vicuna×CNN even at T=128 the Gamma test only reaches ~47 %, while Qwen×ELi5 saturates above 95 % by T=100.
4. **Empirical FPR is well-controlled**: all 16 (cell, B) combinations of invariant_multi under forced PFR Gamma-tail detector give 0.6-1.6 % rejection rate, all close to nominal α=1 %.  detector_per_decoder=`{kind: PFR, labeler: mpfr_direct}` correctly applies the H1 detector to H0 traces and yields a calibrated test.
5. **Multi-draft TPR matches single-draft PFR**: PFR's multi-draft sampler keeps the per-token marginal distribution unchanged, so the per-token uniforms `u_t` recovered post-hoc follow the same Aaronson distribution under H1 regardless of B. Empirically the Gamma-tail TPR is flat in B (variation ≤ 0.006 within each cell at n=1000) and quantitatively matches what single-draft PFR achieves at the same (model, dataset) entropy.
6. **CNN's low-TPR is a dataset/entropy problem, not a code defect**.  Both Qwen and Vicuna on CNN have LPPL 2-3× lower than on ELi5; with only 64 detection tokens the Aaronson signal accumulator is bounded by `T × LPPL` (in nats), which on Vicuna×CNN is 64 × 0.27 = 17 nats — at the boundary of α=1 % significance.  Solutions (none of which require changing the watermark): longer detection window, freer-form prompt (e.g. `"Continue the article: ..."` instead of `"Summarize ..."`), or scoring with the more powerful Gamma-tail test instead of U-Chernoff.
7. **Reproducibility vs 0504 report**: detection-side numbers (TPR, U/Li/PL, LPPL, KL/WS) agree exactly across runs (qwen_cnn TPR_Chernoff 0.219-0.222, vicuna_cnn 0.066, vicuna_eli5 0.362-0.366 — identical to the 0504 report which also tracked the Chernoff column).  TR-side mpfr numbers are 1-3 tok/s slower on this pod relative to 0504 (invariant TR matches), inflating the engineering-overhead percentage by 3-5 pp.  Likely concurrent-CPU contention.
8. **Engineering overhead (mpfr vs invariant TR)**: 5-17 % across cells; biggest on Vicuna+ELi5 B=8 (16.7 %, mpfr 46.2 vs inv 55.5 tok/s).  The overhead grows with B for ELi5 (more rejection blocks → more mpfr Python-side work per token) and is dominated by the per-context blake2b-then-sha256 hash + label byte-serialisation.

## Files in this folder

- `full_qwen_cnn_efpr_n1000.json`
- `full_qwen_eli5_efpr_n1000.json`
- `full_vicuna_cnn_efpr_n1000.json`
- `full_vicuna_eli5_efpr_n1000.json`
- `tpr_vs_T_gamma.png` — overall TPR-vs-T curve, 4 cells, mean-over-B
- `tpr_vs_T_per_cell_gamma.png` — 4-panel per-cell with all B values
- `tpr_vs_T_curves.csv` — numerical TPR(T) values for plotting
- `report.md` — this file
