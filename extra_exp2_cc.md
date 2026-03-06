# EXTRA EXPERIMENTS

These additional experiments aim to understand why **watermarked ERSD can achieve competitive or improved speculative decoding efficiency**, and how it compares against both the original **prefix-coupled ERSD** and the stronger **context-coded ERSD (ERSD-CC)**.

In this revision, we focus on three coupled variants:

| Method | Noise coupling | Seed source | Watermark |
|------|------|------|------|
| ERSD | coupled | prefix | ✗ |
| ERSD-CC | coupled | context code | ✗ |
| ERSD-WM | coupled | prefix | ✓ |

The main purpose is to compare:

1. **Prefix coupling vs context-code coupling**
2. **Vanilla ERSD vs watermarked ERSD**
3. Whether watermark changes speculative behavior at the race, acceptance, and system levels

We no longer center the analysis on the independent-noise ablation; that setting is treated as a separate ablation and is not the focus of this section.

---

# Experiment 1 — Race Gap Analysis

## Goal

Measure how the winner–runner-up gap behaves under the three coupled variants:

- ERSD
- ERSD-CC
- ERSD-WM

This experiment tests whether the race statistics differ across coupling strategies and watermarking.

---

## Variables to Log (per generated token i)

During each race, record the comparison values already produced by the algorithm:

t_min[i]  = winner race time  
t_2nd[i]  = runner-up race time  
gap[i]    = t_2nd[i] - t_min[i]

If using higher-is-better scores, define instead:

gap[i] = score_winner - score_runnerup

Optional:

topk_scores[i]

---

## Stratification Factors

Group results by:

- Draft count **K**
- Method:
  - **ERSD**
  - **ERSD-CC**
  - **ERSD-WM**
- Same temperature and sampling settings

---

## Statistics

For each configuration compute:

E[gap]  
median(gap)  
P(gap > τ)

for several thresholds τ.

Since the gap distribution may be heavy-tailed, emphasis should be placed on:

- `median(gap)`
- tail probabilities
- CDF / CCDF plots

rather than only the mean.

---

## Visualizations

### 1. Gap CDF

Compare:

- ERSD
- ERSD-CC
- ERSD-WM

Key question:

> Does context-coded coupling or watermarking systematically shift the gap distribution?

---

### 2. Box / Violin plots

Grouped by:

K × method

---

### 3. Optional

gap vs token position

---

## Key Hypothesis

This experiment is not only about whether watermark sharpens races, but more generally:

> Do different coupling strategies produce different race-gap statistics, and are those statistics predictive of downstream acceptance?

---

# Experiment 2 — Acceptance Dynamics

## Goal

Understand how speculative acceptance changes across:

- ERSD
- ERSD-CC
- ERSD-WM

This experiment directly studies the block-level behavior that determines efficiency.

---

## Metrics to Record (per speculative block)

ERSD proposes a block of size **K**.

For each block log:

accepted_len ∈ [0, K]  
rejected_pos  
num_retries

Where:

rejected_pos = first reject position

---

## Stratification

Group by:

- K
- method:
  - ERSD
  - ERSD-CC
  - ERSD-WM
- prompt set

---

## Core Statistics

Acceptance profile:

P(accepted_len ≥ r)

for:

r = 1..K

Also compute:

E[accepted_len]  
median(accepted_len)  
E[num_retries]

---

## Visualizations

### 1. Acceptance curves

P(accepted_len ≥ r)

Compare:

- ERSD
- ERSD-CC
- ERSD-WM

Key questions:

> Does context coding increase acceptance depth compared with prefix coupling?  
> Does watermark preserve or improve acceptance depth relative to ERSD?

---

### 2. Reject position histogram

If a method is more stable, we expect:

reject positions shift right

---

### 3. Box plots of accepted_len

Grouped by:

K × method

---

## Interpretation Goal

This experiment should answer:

- whether **ERSD-CC** improves acceptance through better stochastic alignment
- whether **ERSD-WM** preserves speculative acceptability under watermark bias

---

# Experiment 3 — Decomposing AATPS Gain

## Goal

Determine whether differences in AATPS come from real compute reduction.

We compare:

- ERSD
- ERSD-CC
- ERSD-WM

and examine whether improvements come from:

- fewer target evaluations
- better speculative acceptance
- actual wall-clock improvements

---

## Counters to Record (per sequence)

N_out  
N_target_fwd  
N_draft_fwd  
N_verify_ops  
wall_time

Optional per-token logs:

target_fwd_per_token[i]  
verify_ops_per_token[i]

---

## Derived Metrics

Compute:

cost_target = N_target_fwd / N_out  
AATPS_proxy = 1 / cost_target

Report both:

- proxy AATPS
- real wall-clock AATPS

---

## Tables

Grouped by **K × method**

| Metric | ERSD | ERSD-CC | ERSD-WM |
|------|------|------|------|
| N_out | | | |
| cost_target | ⭐ | ⭐ | ⭐ |
| accepted_len | ⭐ | ⭐ | ⭐ |
| draft_cost | | | |
| wall_time/token | | | |

---

## Plots

### 1. cost_target vs K

Key questions:

> Does ERSD-CC reduce target cost relative to ERSD?  
> Does ERSD-WM preserve or improve target cost relative to ERSD?

---

### 2. AATPS vs K

Compare:

- ERSD
- ERSD-CC
- ERSD-WM

---

### 3. wall_time/token vs K

This helps verify that proxy improvements are reflected in real runtime.

---

## Interpretation Goal

This experiment should establish whether:

- **ERSD-CC** improves efficiency through stronger coupling
- **ERSD-WM** remains compute-efficient under watermarking

---

# Experiment 4 — Gap–Acceptance Relationship

## Goal

Directly test whether the winner–runner-up gap predicts speculative acceptance, and whether that relationship differs across:

- ERSD
- ERSD-CC
- ERSD-WM

This experiment links race-level statistics to block-level speculative efficiency.

---

## Procedure

For each generated token record:

gap[i]  
accept_indicator[i]

Where:

accept_indicator[i] = 1 if the draft token is accepted  
accept_indicator[i] = 0 otherwise

---

## Gap Binning

Partition tokens into bins such as:

gap ∈ [0,1)  
gap ∈ [1,2)  
gap ∈ [2,4)  
gap ∈ [4,8)  
gap ∈ [8,+∞)

---

## Statistics

For each bin compute:

accept_rate(gap_bin)

Also compute:

E[accept | gap]

and optionally the correlation:

corr(gap, accept)

---

## Visualizations

### 1. Acceptance rate vs gap bin

Plot:

accept_rate(gap_bin)

for:

- ERSD
- ERSD-CC
- ERSD-WM

Key question:

> Is speculative acceptance monotone in the gap size?

---

### 2. Method comparison at fixed gap

Compare the three methods within each gap bin.

Key question:

> Does ERSD-CC achieve higher acceptance than ERSD even at the same gap level?  
> Does ERSD-WM preserve the same monotone gap–acceptance structure?

---

### 3. Gap distribution overlay

Overlay histograms / CDFs of gap for:

- ERSD
- ERSD-CC
- ERSD-WM

---

## Interpretation Goal

This experiment should clarify whether:

- the gap is a robust predictor of speculative acceptance
- context coding improves acceptance only by changing the gap distribution, or also by improving alignment at fixed gap
- watermark preserves compatibility with this mechanism

---

# Expected Outcome

These four experiments triangulate the mechanism from complementary levels.

---

## Race-Level Explanation

Different coupling strategies may induce different race statistics, summarized by the winner–runner-up gap.

---

## Acceptance-Level Explanation

Better stochastic alignment should lead to:

- longer accepted prefixes
- fewer rejects
- fewer retries

---

## Systems-Level Explanation

Longer accepted prefixes reduce:

target evaluation cost

which improves AATPS.

---

# Main Mechanistic Questions

The revised experiments are designed to answer three questions:

1. **Does context-coded coupling outperform prefix-based coupling?**
2. **Does watermark preserve speculative efficiency relative to vanilla ERSD?**
3. **How are race gap, acceptance dynamics, and AATPS connected?**

---

# Intended Narrative

If confirmed, the empirical story becomes:

context-coded coupling  
→ better stochastic alignment  
→ longer accepted prefixes  
→ fewer target evaluations  
→ higher AATPS

while watermarking:

adds structured bias  
but preserves speculative compatibility  
and may slightly improve or maintain acceptance and efficiency

---

# Implication

The main takeaway is not merely whether watermark changes the race gap, but whether **watermark remains compatible with coupled speculative decoding**, and how its behavior compares with stronger coupling strategies such as **context-coded ERSD**.