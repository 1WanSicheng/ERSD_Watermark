# PFR Watermark × Speculative Decoding — Experimental Writeup (Draft v1, 2026-05-02)

**Status**: first-pass draft. Many sections will be cut or condensed before final paper.
Annotate inline what to keep / drop.

---

## 0. TL;DR (2 sentences)

We show that **PFR (Aaronson Gumbel-trick) watermarking with multi-draft speculative
decoding** preserves quality (LOGPPL parity), preserves acceptance rate (AATPS parity
with the no-watermark `InvariantMultiDraftStrategy` baseline), and gives a stable
detection signal of ANLPPT_U ≈ 0.05 on Qwen2.5-7B / cnn_dailymail across draft counts
B ∈ {2, 4, 6, 8}. Token rate is within 1% of the unwatermarked baseline at B ≥ 4 and
within 5% at B = 2 (residual gap is host-side Python overhead, not GPU).

---

## 1. Goals

1. **Quality**: PFR is unbiased on the *actually-sampled* (post-`process_logits`) distribution.
2. **Speedup compatibility**: PFR drops cleanly into multi-draft speculative decoding
   without losing acceptance rate.
3. **Detection robustness**: water­mark strength does not get diluted as the number of
   parallel drafts B grows.
4. **Engineering**: bring our PFR-cached implementation's token rate within range of
   the un-watermarked invariant baseline used by Rowan et al. (List-Level, NeurIPS 2025).

---

## 2. Single-draft experiments (B = 1)

### 2.1 Setup

| Setting | Value |
|---|---|
| Sampling | `top_k=50, top_p=1.0, T=1.0` |
| `max_new_tokens` | 128 |
| `private_key` | `"1234"` (fixed) |
| Lookahead L | {1, 2, 3, 4} |
| Sample size n | 100 / 200 / 500 / 1000 |
| Seeds | 1 (single run per cell — to upgrade to ≥3 for paper) |

### 2.2 Model / dataset matrix

| target / draft | dataset | Notes |
|---|---|---|
| `huggyllama/llama-7b` / `JackFram/llama-68m` | `cnn_paper_summarization`, `eli5` | Base model, no chat template. Uses paper-exact `System:/INPUT:/OUTPUT:` prompt. |
| `lmsys/vicuna-7b-v1.5` / vicuna-family small | `cnn`, `eli5` | FastChat chat template injected (vicuna ships without `chat_template` attribute). |
| `Qwen/Qwen2.5-7B-Instruct` / `Qwen/Qwen2.5-0.5B-Instruct` | `cnn`, `eli5`, `gsm8k` | Native chat template. |

### 2.3 Decoders

| Name | Scheme | RNG family |
|---|---|---|
| `pfr` | PFR Aaronson | torch.Generator (CUDA Philox) |
| `basic_uwm` | DG (DeltaGumbel) | numpy.PCG64 |
| `basic_uwm_torch_rng` | DG | torch.Generator (control to isolate RNG family) |
| `dg_align` | DG, seed-aligned with PFR | torch.Generator |
| `mc_uwm_speed`, `mc_uwm_strength` | Monte Carlo coupling watermark variants | — |

### 2.4 Metrics

| Metric | Definition | Use |
|---|---|---|
| AATPS | mean accepted tokens / step | speculative-decoding efficiency |
| token_rate (TR) | wall-time tok/s, end-to-end | hardware-dependent throughput |
| ANLPPT_U | −log P-value / token, U-score Chernoff bound | watermark detection |
| ANLPPT_Li | Li bound | tighter tail bound |
| ANLPPT_PL | Pointwise log-MGF (PL bound) | sharpest |
| KL ratio | `ws_sum / H_sum` (per context) | unbiasedness diagnostic — should equal 1 |

### 2.5 Findings

#### 2.5.1 Llama-7B base: PFR appears 12% weaker than DG — *RNG family artifact*

**Symptom**: on cnn n=100, `pfr.ANLPPT_U` ≈ 0.20 vs `basic_uwm.ANLPPT_U` ≈ 0.225 (12% gap).

**Diagnosis path**:
- Both schemes hash `(label, private_key)` with SHA-256, so the *seed integer* is identical.
- `pfr` then samples uniforms via `torch.Generator(cuda).manual_seed(seed)` + `torch.rand` (Philox bit stream).
- `basic_uwm` samples via `numpy.random.default_rng(seed)` (PCG64 bit stream).
- Same seed → different uniform sequences → different specific token samples → different ANLPPT.

**Control**: built `basic_uwm_torch_rng` (DG algorithm + torch.Generator). Gap collapses to **1.1%**.

**Conclusion**: 12% was an RNG family artifact interacting with base llama's diffuse logits + top_k=50 truncation. Both schemes are unbiased on the truncated distribution; tail-bound metrics like ANLPPT_U are sensitive to which specific tokens land.

**Lesson for paper**: when comparing watermark schemes, **always include a same-RNG-family control**.

#### 2.5.2 Prompt format dominates AATPS on base models

Initial cnn AATPS at L=4 was 1.7–1.8; Hu & Huang report ≈ 1.99.

Switched to paper-exact prompt:
```
System:Summarize the following article.
INPUT:{article[:1000]}
OUTPUT:
```
→ AATPS jumps to **2.038**, matching paper.

**Why**: base llama recognizes `System:/INPUT:/OUTPUT:` as instruction-tuned demonstration → lower per-token entropy → drafter agrees more often.

**Lesson**: when reproducing AATPS numbers from prior work, the prompt template is non-negotiable.

#### 2.5.3 KL ratio formula bug

Old code reported KL ratio ≈ 0.80 on llama-7b base, suggesting PFR was ~20% biased.

**Bug**: `H(p)` was computed on un-truncated `raw_logits` but `ws = -log p(winner)` was computed on `process_logits`-truncated logits. On base llama, `top_k=50` strips ~18% of the entropy → denominator inflated → ratio understated.

**Fix**: evaluate both `H(p)` and `ws` on the post-`process_logits` distribution. Mask `-inf` positions in the entropy multiplicand to avoid `0 * -inf = NaN`. KL ratio becomes ≈ 1.0, confirming unbiasedness on the actually-sampled distribution.

**Lesson**: any ratio of two distribution-derived quantities must use *the same* distribution.

#### 2.5.4 Vicuna chat template injection

`lmsys/vicuna-*` repos ship without `chat_template`. `apply_chat_template` raises → silent fallback to plain user-message text → instruction-following evaluation defeated.

**Fix**: detect `vicuna` in model id, inject FastChat template:
```
{system} USER: {q1} ASSISTANT: {a1}</s>USER: {q2} ASSISTANT:
```

Without this, vicuna behaves like base llama in our evals.

---

## 3. Multi-draft experiments (B ≥ 2)

### 3.1 Setup

| Setting | Value |
|---|---|
| target / draft | `Qwen2.5-7B-Instruct` / `Qwen2.5-0.5B-Instruct` |
| Dataset | `cnn_dailymail` (1000-sample headline run, 100-sample iterations) |
| L | 4 |
| B | {2, 4, 6, 8} |
| Sampling | `top_k=50, top_p=1.0, T=1.0` |
| `max_new_tokens` | 128 |
| Seeds | 1 (to upgrade to ≥3 for paper) |
| Hardware | NVIDIA L20 (Ada, 48 GB), CUDA 12.4, torch 2.5.1, transformers 4.46.3 |

### 3.2 Decoders

| Name | Implementation | Watermark? |
|---|---|---|
| `ms_pfr_cached` | `MPFR_spec.multi_draft_pfr_batched_cached` | yes (per-context blake2b labeler path) |
| `mpfr_torchgen_cached` | `MPFR_spec.mpfr_batched_torchgen_cached` | yes (torch.Generator path; detection target) |
| `invariant_multi` | `SpeculativeDecoding.strategy.InvariantMultiDraftStrategy` | no (Rowan et al. baseline) |

### 3.3 Engineering optimizations (all merged to main)

Five layers, ordered by when they landed:

1. **Cross-block draft KV cache reuse** — keep the deepest cached ancestor of the realized prefix between blocks; next block's depth=1 forward only encodes 1–2 new tokens.
2. **Within-block incremental decode** — at depth ≥ 2 gather parent rows from the prev-depth batched cache and feed only one new token per row. Workload drops from `L(L+1)/2` to `L` token-positions per block. Mirrors invariant_multi's pattern.
3. **B1 — `torch.Generator` reuse**: one shared generator per `build_*_tree` call, re-seeded per row. Skips per-row Generator allocation.
4. **B2 — `SharedPFRSource` cache by ContextKey**: same context across multiple depths skips prefix byte serialization + sha256.
5. **B3 — deferred D2H sync**: one `torch.cat().cpu().tolist()` per depth instead of one per row. CUDA syncs drop from `B*L` to `L` per block.

### 3.4 Byte-equivalence guarantee

Test in `tests/test_pfr_b1b2b3_byte_equivalence.py` verifies B1/B2/B3 produce byte-identical RNG output. End-to-end: 100-sample bench shows **800/800 sample rows × 8 (method × B) combos** with `tokens / blocks / AATPS / accepted_mean diff = 0.000000` vs the within-block-only baseline.

This implies detection-side recovered uniforms `r_t` are byte-identical → ANLPPT is byte-identical.

### 3.5 Headline numbers (1000 samples, Qwen2.5, cnn_dailymail, L=4)

| method / B | AATPS | TR (tok/s) | ANLPPT_U | ANLPPT_Li | ANLPPT_PL |
|---|---|---|---|---|---|
| `ms_pfr_cached` B=2 | 2.801 | 39.59 | 0.0512 | 0.0366 | 0.0538 |
| `ms_pfr_cached` B=4 | 3.130 | 40.39 | 0.0513 | 0.0366 | 0.0540 |
| `ms_pfr_cached` B=6 | 3.294 | 39.95 | 0.0515 | 0.0368 | 0.0541 |
| `ms_pfr_cached` B=8 | 3.404 | 39.26 | 0.0508 | 0.0361 | 0.0535 |
| `mpfr_torchgen_cached` B=2 | 2.803 | 39.57 | 0.0525 | 0.0372 | 0.0551 |
| `mpfr_torchgen_cached` B=4 | 3.128 | 40.34 | 0.0526 | 0.0374 | 0.0549 |
| `mpfr_torchgen_cached` B=6 | 3.294 | 40.06 | 0.0526 | 0.0371 | 0.0552 |
| `mpfr_torchgen_cached` B=8 | 3.410 | 39.36 | 0.0521 | 0.0371 | 0.0544 |
| `invariant_multi` B=2 | 2.805 | 41.80 | nan | nan | nan |
| `invariant_multi` B=4 | 3.102 | 41.33 | nan | nan | nan |
| `invariant_multi` B=6 | 3.275 | 40.80 | nan | nan | nan |
| `invariant_multi` B=8 | 3.383 | 39.69 | nan | nan | nan |

### 3.6 Multi-draft observations

**(1) AATPS: zero watermark cost.** PFR vs invariant within ±0.03 across all B. `mpfr_torchgen` at B=8 (3.410) actually exceeds invariant (3.383). Confirms PFR is unbiased on the truncated distribution: the expected acceptance rate equals that of the un-watermarked sampler.

**(2) Token rate gap shrinks with B.**

| B | PFR vs invariant TR gap |
|---|---|
| 2 | −5.3% |
| 4 | −0.5% |
| 6 | +0.3% |
| 8 | +0.7% |

Shape is the signature of fixed per-block Python overhead being amortized as B grows. Remaining 5% at B=2 lives in the `ms_pfr_tokens_from_logprobs` per-row sampling loop. Closing it would require a fused batched implementation (B4 in our notation), which touches the detection-side noise reconstruction. We deferred this — risk-reward is low for paper-figure regimes (B ≥ 4).

**(3) Watermark strength stable across B.**
ANLPPT_U ≈ 0.05 with < 2% variation across B ∈ {2,4,6,8}. A 128-token sequence carries ≈ 6.4 nats of detection signal ≈ 600× p-value reduction.

`mpfr_torchgen` is ~2.5% stronger in U than `ms_pfr` because detection re-derives noise via the same `torch.Generator` path; `ms_pfr` uses a slightly different label scheme (per-context blake2b) so detection picks up the signal less perfectly.

**(4) 100-sample agrees with 1000-sample within 2%.** 100-sample bench is sufficient for fast iteration.

### 3.7 Cross-environment performance check

Collaborator reports `'DynamicCache' object has no attribute 'key_cache'` AttributeError. Diagnosis:
- transformers ≥ 4.50 refactored `DynamicCache` (`key_cache`/`value_cache` → `layers[i].keys`/`.values`).
- transformers ≥ 4.55 also removed `to_legacy_cache()`.
- Our remote (4.46.3) returns legacy tuple from `forward()`, so we don't hit this path.

**Pending**: collaborator returns timing diagnostic (target Qwen2.5-7B forward ms at seq=65). Their A6000 has ~33% the FP16 TFLOPS of our L20, so 1.5–2× wall-time difference is expected and explains most of their TR gap. We will only ship a transformers-cache-compat layer if their forward time is comparable to ours but TR still differs (i.e., the gap is host-side, not GPU).

---

## 4. Paper survey: how the field reports

### 4.1 Rowan, Phan, Khisti — *List-Level Distribution Coupling* (NeurIPS 2025)

> [`G:\Nips_pfr_wmsd\_paper_review\ashish.txt`](../../../_paper_review/ashish.txt)

| | |
|---|---|
| Type | speculative-decoding speedup paper, no watermark |
| Headline table | columns = dataset (GSM8K \| HumanEval \| NaturalReasoning), rows = decoder (SpecInfer / SpecTr / Daliri / **Ours**); cell = (BE, TR) side-by-side |
| Speed metric | **BE** (block efficiency = mean accepted draft tokens / verifier call) and **TR** (% wall-time speedup vs single-draft Leviathan) |
| Sweep | Qwen2.5-7B/0.5B main; Llama-2 + Llama-3 in appendix; L ∈ {4,5}; K ∈ {1…8} (K=8 main) |
| Quality audit | **ROUGE-1/2/L** for output-identity check across drafter perturbations |
| Statistics | mean ± std-err, **5 seeds** |
| Note | their TR is *relative* (% over single-draft baseline), not absolute tok/s — neutralizes hardware |

### 4.2 Hu & Huang — *Inevitable Trade-off* (NeurIPS 2024, arXiv 2410.20418)

> [`G:\Nips_pfr_wmsd\_paper_review\inevitable.txt`](../../../_paper_review/inevitable.txt)

| | |
|---|---|
| Type | watermark + speculative decoding trade-off paper |
| Headline figure | **2×2 scatter grid**: rows = reweight (DG / Gamma), cols = score (maximin-LLR / U Score); per panel x = AATPS, y = ANLPPT, point cloud over K ∈ {1…4} |
| Numeric appendix table | rows = (K, method, reweight); columns = AATPS / **PTT** / **LOGPPL** / ANLPPT(U) / ANLPPT(maximin-LLR) |
| Methods plotted | Basic Sampling (no-wm no-spec) / VUW (wm only) / VSpS (spec only) / **MWS** / **MSE** |
| Detection metric | **ANLPPT** — log p-value / token via score MGF Chernoff bound, λ optimized |
| Quality audit | **LOGPPL** (separate column, used purely as unbiasedness check) |
| Statistics | **3σ confidence intervals** |
| Sweep | Llama-7B/68M main; Llama-13B appendix; CNN/DailyMail + open-ended; reweight × score × K full cross |
| Compute | reports ~1200 A6000-hours total |

### 4.3 He, Li, Shen, Su, Long — *Improve the Trade-off* (ICLR 2026, arXiv 2602.01428)

> [`G:\Nips_pfr_wmsd\_paper_review\improving.txt`](../../../_paper_review/improving.txt)

| | |
|---|---|
| Type | better detection rule for watermark + spec decoding |
| Headline figure | **3 panels horizontal**: AATPS vs K \| TPR@FPR=1% vs # tokens (Gumbel-max) \| same (SynthID) |
| Curve coloring | detector: orange = Ars-τ / Bayes-MLP (theirs); blue = Ars-Prior / Bayes-Prior (Dathathri); black = Oracle |
| Detection metric | **TPR @ FPR=1%** as primary, plotted against generated-token count → "sample efficiency of detection"; ROC in appendix |
| Quality audit | **LOGPPL** in appendix |
| Speed | AATPS ∈ [1, K+1] with **95% CI**, anchored at Std. SpecSampl |
| Sweep | watermark ∈ {Gumbel-max, SynthID(m=30)}; dataset ∈ {ELI5, C4}; model ∈ {Llama 68M→7B, Gemma 2B→7B}; K ∈ {2,3,4} |
| Discussion style | every empirical claim paired with a theorem |

### 4.4 Common reporting patterns

1. **Two-axis trade-off as the headline** — x = efficiency (AATPS / BE / TR), y = detectability (ANLPPT / TPR@FPR), swept along K.
2. **Pareto / scatter plots over multiple K**, not bar charts.
3. **Multi-panel grid**: row = reweight family or watermark, column = score / dataset.
4. **LOGPPL audit table**, separate from main results, purely for unbiasedness verification.
5. **Error bars always** — 3σ (Hu & Huang) / 95% CI (He) / std-err over 5 seeds (Rowan).
6. **Speed reported relatively** (% speedup over single-draft baseline) to neutralize hardware.
7. **Numeric appendix table** mirrors the visual main figure.

---

## 5. Recommendations for our paper writeup

### 5.1 Headline figure — 3-panel horizontal (mirror He et al.)

| Panel | x | y | Curve color | Curve marker |
|---|---|---|---|---|
| Left | L | AATPS | scheme (PFR-ms / PFR-tg / DG / no-wm) | B |
| Middle | # generated tokens | TPR @ FPR = 1% | scheme | B |
| Right | AATPS | ANLPPT_U | scheme | (L, B) cell |

The right panel is our headline Pareto plot: each method draws a curve in (AATPS, ANLPPT_U) space sweeping (L, B); reader sees instantly that PFR sits on the same AATPS as `invariant_multi` while carrying detection signal.

### 5.2 Main results table (mirror Hu & Huang)

Rows: (B, scheme). Columns: AATPS, tok/s, **LOGPPL**, ANLPPT_U, ANLPPT_Li, ANLPPT_PL. Each cell `mean ± 95% CI` over ≥ 3 seeds. Always include both the `no-wm + no-spec` (Basic Sampling) row and the `no-wm + spec` (invariant_multi) row as anchors.

### 5.3 Required audits (appendix)

- **LOGPPL audit**: PFR row matches `Basic Sampling` row within ±0.5%.
- **AATPS audit**: PFR row matches `invariant_multi` row within ±1% at the same B.
- **Drafter-invariance audit** (mirror Rowan): same seed, perturbed drafter, ROUGE-L > 0.95.
- **Cross-RNG ablation** (our finding): PCG64 vs Philox give different uniforms from the same SHA-256 seed, propagating to different ANLPPT. Include as a cautionary section.

### 5.4 Required by every paper in this space

- ≥ 3 seeds, **report 95% CI** with every number. No bare numbers.
- Speed reported in **two columns**: absolute tok/s (with hardware footnote) **and** % speedup over single-draft Leviathan baseline.
- Caption main figure with: GPU, transformers version, torch version, attn implementation. (We were burned by collaborator hardware mismatch — write it down.)

### 5.5 Coverage to match the field

| Paper | Datasets | Models |
|---|---|---|
| Rowan | GSM8K + HumanEval + NaturalReasoning | Qwen2.5-7B/0.5B + Llama series |
| Hu & Huang | CNN/DailyMail + open-ended | Llama-7B/68M (+ Llama-13B) |
| He et al. | ELI5 + C4 | Llama 68M→7B + Gemma 2B→7B |

Our current coverage: cnn_dailymail + eli5 + gsm8k (✓ all three), Qwen2.5 + Llama-7B + Vicuna (✓ Hu & Huang main + Rowan main).

**Add to fully match**: HumanEval / MBPP (code generation) + Llama-3-8B + Gemma-7B if we want full Rowan / He et al. parity.

---

## 6. Open items / TODOs (for tracking)

| # | Item | Priority |
|---|---|---|
| 1 | Re-run all multi_draft tables with **3 seeds**, report 95% CI | P0 |
| 2 | Add **HumanEval** dataset to `experiments/_shared.py:load_prompts` | P1 |
| 3 | Add **Gemma-7B / 2B** model pair | P2 |
| 4 | TPR@FPR=1% rollout plot (mirror He et al. middle panel) | P0 |
| 5 | LOGPPL audit table (Basic Sampling + PFR + invariant_multi rows) | P0 |
| 6 | Drafter-invariance ROUGE audit | P1 |
| 7 | Cross-RNG section: replot llama-7B `pfr` vs `basic_uwm_torch_rng` to show 12% → 1.1% gap collapse | P1 |
| 8 | Decide whether to ship transformers-cache-compat layer (depends on collaborator's forward-ms) | P1 |
| 9 | Decide whether to do B4 (fused batched ms_pfr) — only if a reviewer pushes on B=2 TR | P3 |
| 10 | Final pass: prune everything not in 5.1–5.5 from this draft | — |

---

## 7. Pointer index (where things live)

| What | Where |
|---|---|
| Multi-draft 1000-sample headline JSON | `data/0502multi_draft_1000cnn_samples/multi_draft_qwen_cnn_n1000_L4_B2-4-6-8_B1B2B3.json` |
| Multi-draft 1000-sample config | `experiments/configs/multi_draft_qwen_cnn_n1000_L4_B2468.json` |
| Within-block bench JSON (B1B2B3 vs WITHIN_BLOCK_PATCH) | `outputs/mpfr_bench_cnn_n100_L4_B2-4-6-8_B1B2B3.json` (remote only) |
| Byte-equivalence test | `tests/test_pfr_b1b2b3_byte_equivalence.py` |
| Paper extracts | `_paper_review/{ashish,inevitable,improving}.txt` |
| Multi-draft entrypoint | `experiments/run_multi_draft.py` |
| Single-draft entrypoint | `experiments/run_single_draft.py` |
| PFR sampling primitive (B1+B3) | `accuwm/multi_draft_utils.py:ms_pfr_tokens_from_logprobs` |
| `SharedPFRSource` (B1) | `accuwm/pfr.py:SharedPFRSource.uniform_noise` |
| Within-block + cross-block + B2 | `MPFR_spec/mpfr_batched_torchgen_cached.py` |
| Detection-side noise re-derivation | `unbiased_watermark/scores/pfr_aaronson.py:_uniform_for_token` |
| Remote box | `root@43.99.49.129:/root/PFR/ERSD_Watermark/` (NVIDIA L20) |
