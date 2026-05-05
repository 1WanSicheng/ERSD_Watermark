# Ablation Experiment Plan (2026-05-05, rev 3)

> Two ablations attached to the main 2×2 result (Qwen2.5 / Vicuna ×
> cnn_dailymail / eli5). Anchor cell for both ablations:
> **Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct on `cnn_dailymail`**.
>
> **Revision history**:
> - rev 1 → rev 2: drop MAUVE; simplify Exp 2 drafters from 7 (3 axes)
>   to 4 variants (1 model swap + 2 temperatures); 1 seed.
> - rev 2 → rev 3: drop output length distribution from Exp 1; drop AUC
>   from Exp 2 (TPR@FPR=1% sufficient); drop combined-timeline section.

## Notation

| Symbol | Meaning |
|---|---|
| Target | Qwen2.5-7B-Instruct (held fixed across all ablation runs) |
| D₀ | Default drafter: Qwen2.5-0.5B-Instruct (matches main result) |
| Dₖ | Drafter variant k for Exp 2 |
| K | Lookahead (= L in our config) |
| B | Number of drafts at root (= num_drafts) |
| τ | Detection threshold |

Sampling parameters held fixed at main-paper defaults (`top_k = 50`,
`top_p = 1.0`, `T_target = 1.0`, `max_new_tokens = 128`, `private_key`
fixed). The watermark scheme operates on the post-`process_logits`
distribution.

---

## Exp 1 — Context Quality Preservation

### Goal
Demonstrate that PFR (single-draft) and MPFR (multi-draft) **preserve
context quality** of the target model: their generated outputs are
statistically indistinguishable from non-watermarked, non-speculative
direct sampling under standard quality metrics.

### Hypothesis
Under unbiased watermarking + correct speculative-decoding implementation,
the marginal output distribution at every token position equals the
target's (post-truncation) distribution. Therefore any quality metric
that depends only on this distribution should match direct sampling.

### Setup
| Item | Value |
|---|---|
| Model pair | Qwen2.5-7B-Instruct (target) / Qwen2.5-0.5B-Instruct (drafter D₀) |
| Dataset | `cnn_dailymail` (cnn `highlights` field used for ROUGE-L gold) |
| Sampling | `top_k = 50`, `top_p = 1.0`, `T = 1.0`, `max_new_tokens = 128` |
| Lookahead | K = 4 (matches main result) |
| Multi-draft B | 4 (for the multi-draft cell) |
| n | 1000 (matches main paper scale) |
| Seeds | **1 seed** (using Hu & Huang style 3σ Bernoulli/empirical CIs from n = 1000) |

### Decoder grid (7 rows)

| # | Decoder | Speedup? | Watermark? | Role |
|---|---|---|---|---|
| 1 | `basic_sample` (no spec, no wm) | — | — | **Quality reference** |
| 2 | `pfr_no_watermark` (B=1) | ✓ (spec only) | — | Spec-only baseline |
| 3 | `basic_uwm` (autoregressive watermark) | — | DG | Watermark-only baseline |
| 4 | `mc_uwm_speed` (Hu & Huang MSE) | ✓ | DG | Trade-off baseline |
| 5 | `mc_uwm_strength` (Hu & Huang MWS) | ✓ | DG | Trade-off baseline |
| 6 | **`pfr` (ours, B=1)** | ✓ | PFR | **Our single-draft** |
| 7 | **`mpfr_torchgen_cached` (ours, B=4)** | ✓✓ | PFR | **Our multi-draft** |

### Metrics

| Metric | Direction | What it measures | Standard reference |
|---|---|---|---|
| **LOGPPL** (log-perplexity of generated output under target) | match row 1 | Token-level naturalness under target's distribution | Hu & Huang 2024 Tab 1 |
| **ROUGE-L vs gold summary** (cnn `highlights`) | match row 1 | Task quality (semantic overlap with gold reference) | Lin 2004 |
| AATPS | — | speculative decoding efficiency (sanity) | — |
| `tok/s` (TR) | — | wall-clock throughput (sanity) | — |
| ANLPPT_U | — | watermark strength reference | Hu & Huang 2024 |

### Headline table layout (paper-ready)

| Decoder | LOGPPL ↓ | ROUGE-L ↑ | AATPS | tok/s |
|---|---|---|---|---|
| `basic_sample` (reference) | a₀ ± σ | b₀ ± σ | 1.00 | … |
| `pfr_no_watermark` | a₀ ± σ | b₀ ± σ | … | … |
| `basic_uwm` | a₀ ± σ | b₀ ± σ | 1.00 | … |
| `mc_uwm_speed` (MSE) | … | … | … | … |
| `mc_uwm_strength` (MWS) | … | … | … | … |
| **`pfr`** | … | … | … | … |
| **`mpfr_torchgen_cached`** | … | … | … | … |

**Pass criterion**: rows 6 & 7 (ours) have LOGPPL and ROUGE-L within 1
std-error of row 1 (`basic_sample`).

### Expected outcome
All 7 rows should have LOGPPL within `±0.02` of each other (mirrors
Hu & Huang Tab 1, where all 8 schemes report LOGPPL ∈ [1.71, 1.78]).
ROUGE-L statistically indistinguishable. AATPS and tok/s of course
differ — that's the point of the spec-decoding rows.

### Reporting
Single LaTeX table mirroring Hu & Huang appendix Table 1 layout.
Caption: **"Output quality preservation across watermarking and
speculative-decoding schemes. All values are mean ± std-error over
n = 1000 prompts. LOGPPL: lower is better, but unbiased schemes match
the no-watermark reference. ROUGE-L: vs gold summary, higher is
better."**

### Code requirements (Exp 1)

| File | Change |
|---|---|
| `experiments/_shared.py` | Register new decoder `basic_sample` (no spec, no watermark; uses `model.generate(...)` with `do_sample=True`) |
| `experiments/run_single_draft.py` + `run_multi_draft.py` | Compute LOGPPL post-generation: re-run target on the generated sequence, compute mean `-log p(y_t \| y_{<t})` |
| `experiments/_shared.py` | Compute ROUGE-L vs `highlights` (use `rouge_score` library); only when dataset has reference field |

---

## Exp 2 — Drafter Invariance via Drafter Substitution

### Goal
Demonstrate that PFR / MPFR's watermark **detection signal is invariant
under drafter substitution**, while Hu & Huang's MWS and MSE are not.

### Hypothesis
PFR's verify mechanism `argmax(log p_target + g)` depends only on the
target's logits + keyed Gumbel noise; the drafter only proposes
candidates. Therefore swapping the drafter cannot change which token is
accepted, and per-token watermark detection signal is preserved.
By contrast, MWS and MSE's verify rules depend on `p_drafter`, so
drafter substitution changes the per-token signal.

### Setup
| Item | Value |
|---|---|
| Target | Qwen2.5-7B-Instruct (**fixed across all variants**) |
| Dataset | `cnn_dailymail` |
| Sampling | `top_k = 50`, `top_p = 1.0`, `T_target = 1.0`, `max_new_tokens = 128` |
| Lookahead | K = 4 |
| n | 1000 |
| Seeds | **1 seed** |
| Detection token-count | T_eval ∈ {32, 64, 128} (TPR vs # tokens curve, He et al. 2026 style) |

### Drafter variants (4 cells)

| Variant | Drafter model | T_drafter | Perturbation axis |
|---|---|---|---|
| **D₀** | Qwen2.5-0.5B-Instruct | 1.0 | default |
| **D₁** | **Qwen2.5-1.5B-Instruct** | 1.0 | **scale (3× larger drafter)** |
| **D₂** | Qwen2.5-0.5B-Instruct | **0.5** | **temperature (sharper)** |
| **D₃** | Qwen2.5-0.5B-Instruct | **1.5** | **temperature (more diffuse)** |

**Rationale**:
- D₀ → D₁: model-scale swap. The most important perturbation because it
  forces the drafter's logit distribution to genuinely change (not just
  re-scale). This is where MWS would be expected to lose invariance most
  cleanly, since its verify rule explicitly uses `p_drafter`.
- D₀ → D₂ / D₃: temperature swap (within same drafter). Cheaper and
  isolates the effect of drafter sampling sharpness from model identity.

> Pre-experiment download required: **Qwen2.5-1.5B-Instruct** (~3 GB)
> from HuggingFace.

### Decoder grid

| # | Decoder | Watermarked? | Role |
|---|---|---|---|
| 1 | `pfr_no_watermark` | — | H₀ distribution (for TPR threshold) |
| 2 | `basic_uwm` | DG | Single-draft no-spec reference (vacuously drafter-invariant since no drafter) |
| 3 | `mc_uwm_speed` (MSE) | DG | Hu & Huang non-invariant baseline |
| 4 | `mc_uwm_strength` (MWS) | DG | Hu & Huang baseline that maintains aggregate strength |
| 5 | **`pfr` (ours, B=1)** | PFR | **Our single-draft** |
| 6 | **`mpfr_torchgen_cached` (ours, B=4)** | PFR | **Our multi-draft** |

### Metrics

| Metric | Direction | Reference |
|---|---|---|
| **TPR @ FPR = 1%** at T_eval | higher better | He et al. 2026; Kirchenbauer et al. 2023 |
| **TPR variance across drafter variants** (range, max−min) | **lower** better | The drafter-invariance signature |
| **Pairwise ROUGE-L** between drafter conditions (per-prompt min, mean across n) | higher = output more similar | Lin 2004; Rowan 2025 |
| AATPS per drafter variant | sanity check | mirrors Rowan 2025 Tab 2 protocol |

### H₀ distribution for TPR computation

H₀ = `pfr_no_watermark` outputs. Since H₀ depends on target's natural
distribution (not on key), drafter substitution affects H₀ only through
the spec-decoding rejection sampling — which is unbiased. We pool H₀
scores across all 4 drafter variants (~4000 prompts) for a tight
1%-quantile estimate.

### Headline table layout (paper-ready, 2 stacked tables)

#### Table 2a — TPR @ FPR=1% (T_eval = 128) per drafter variant

| Decoder | TPR(D₀) | TPR(D₁) | TPR(D₂) | TPR(D₃) | **range (max−min)** |
|---|---|---|---|---|---|
| `basic_uwm` (no drafter, vacuous invariance) | x | x | x | x | 0% (by design) |
| `mc_uwm_speed` (MSE) | … | … | … | … | should be **large** |
| `mc_uwm_strength` (MWS) | … | … | … | … | small-moderate |
| **`pfr` (ours)** | … | … | … | … | **≈ 0%** |
| **`mpfr_torchgen_cached`** | … | … | … | … | **≈ 0%** |

Reported as `mean ± SE` over n = 1000.

#### Table 2b — TPR vs sequence length (T_eval ∈ {32, 64, 128}), at D₀

| Decoder | TPR @ T=32 | TPR @ T=64 | TPR @ T=128 |
|---|---|---|---|
| `basic_uwm` | … | … | … |
| `mc_uwm_speed` | … | … | … |
| `mc_uwm_strength` | … | … | … |
| **`pfr`** | … | … | … |
| **`mpfr_torchgen_cached`** | … | … | … |

Mirrors He et al. 2026 middle panel (TPR vs # tokens). Shows our scheme
not only is invariant but also reaches high TPR with fewer tokens.

### Expected outcome
- `pfr` and `mpfr` TPR range across D₀..D₃ ≤ 2 percentage points (within
  statistical noise of the ±0% predicted by the theorem).
- `mc_uwm_speed` (MSE) TPR range ≥ 8 pp — most dramatic when drafter
  swaps from 0.5B → 1.5B (D₀ → D₁).
- `mc_uwm_strength` (MWS) TPR range ~3-5 pp — somewhat invariant on
  aggregate detection but **noticeably more variable** than ours under
  the model-scale swap (D₀ → D₁).
- `basic_uwm` TPR range = 0 by construction (no drafter).

### Reporting
Two-panel table (Table 2a + 2b) plus optional ROUGE-L pairwise table in
appendix as auxiliary evidence of byte-level invariance.

Caption suggestion:

> **Drafter invariance under drafter substitution (Qwen2.5-7B-Instruct
> target, cnn_dailymail, n = 1000).** The drafter is varied across model
> scale (D₀ = Qwen2.5-0.5B-Instruct; D₁ = Qwen2.5-1.5B-Instruct) and
> sampling temperature (D₂ = 0.5; D₃ = 1.5). Our PFR/MPFR maintains
> TPR @ FPR=1% within ±2 pp across all variants, while Hu & Huang's
> MSE varies by ≥ 8 pp under model-scale swap and MWS by 3–5 pp. The
> drafter-invariance property is therefore not merely theoretical:
> under realistic deployment scenarios where the drafter may be
> substituted (cost optimization, multi-tenant serving, fine-tune
> updates), only the invariant scheme delivers stable detection.

### Code requirements (Exp 2)

| File | Change |
|---|---|
| `experiments/_shared.py` | `draft_temperature` override (already done) |
| `experiments/configs/` | 4 new config files (one per drafter variant) for Qwen+cnn+each Dᵢ |
| `experiments/compute_tpr_at_fpr.py` (already exists) | Add `--T_eval` parameter to evaluate TPR on a fixed prefix length, not the full realized sequence |
| `experiments/compute_pairwise_rouge.py` (already exists) | No change |

### Pre-experiment downloads required
- **Qwen2.5-1.5B-Instruct** (~3 GB) from HuggingFace.
  One-time download, ~5 minutes on remote GPU box's HF cache.

---

## Decision log

- ❌ **MAUVE**: dropped per user (rev 2).
- ❌ **Output length distribution**: dropped per user (rev 3). LOGPPL +
  ROUGE-L sufficient for quality audit.
- ❌ **AUC of ROC**: dropped per user (rev 3). Threshold-free summary,
  but TPR @ FPR=1% gives the watermark-deployment-relevant number
  directly (fixed false-alarm budget); AUC was redundant.
- ✅ **1 seed**: per user (rev 2). With n = 1000, Bernoulli/empirical SE
  on TPR is ~1.5pp (sufficient to claim ±2pp invariance). Hu & Huang
  also use 1 seed × n = 1000.
- ✅ **Drafter variants 7 → 4**: per user (rev 2). Drop axes 2
  (specialized drafters) and trim axis 1 (drop 3B). Keep model swap
  (0.5B → 1.5B) + 2 temperatures.

---

## Appendix: Standard-metric provenance (for paper citation)

| Metric | Used by |
|---|---|
| **LOGPPL** | Hu & Huang 2024 (NeurIPS) Tab 1 |
| **ROUGE-L** | Lin 2004 (ACL Workshop); Rowan et al. 2025 (NeurIPS) §4.3 |
| **TPR @ FPR=1%** | He et al. 2026 (ICLR); Kirchenbauer et al. 2023 (ICML) |
| **TPR vs # tokens curve** | He et al. 2026 (ICLR) — main figure middle panel |
| **AATPS / Block Efficiency** | Hu & Huang 2024; Rowan et al. 2025 |
| **Pairwise output similarity** | Rowan et al. 2025 §4.3 (drafter invariance check) |

No invented metrics. Every column in every table maps to a peer-reviewed
prior work.
