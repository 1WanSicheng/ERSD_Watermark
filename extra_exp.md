# EXTRA EXPERIMENTS

These additional experiments aim to understand why **watermarked ERSD achieves higher AATPS when K becomes large**.

A key concern is whether the observed phenomenon comes from **prefix-token artifacts** or from the **race coupling mechanism itself**.  
Therefore, all experiments are repeated under a **CC ablation setting**.

We compare the following variants:

| Method | Prefix reuse | Race Coupling (CC) | Watermark |
|------|------|------|------|
| ERSD | ✓ | ✓ | ✗ |
| ERSD + WM | ✓ | ✓ | ✓ |
| ERSD − CC | ✓ | ✗ | ✗ |
| ERSD − CC + WM | ✓ | ✗ | ✓ |

This allows us to isolate the role of **race coupling** in the watermark–speculative interaction.

---

# Experiment 1 — Race Sharpness Analysis

## Goal

Test whether watermarking makes each speculative race **sharper**, i.e., enlarges the gap between the winner and the runner-up.

If the hypothesis holds, sharper races may enable:

- earlier decisions
- fewer verification steps
- lower target evaluation cost.

---

## Variables to Log (per generated token i)

During each race the algorithm already produces scalar values used for comparison (e.g., exponential race times or equivalent scores).

Record:

t_min[i]  = winner race time  
t_2nd[i]  = runner-up race time  
gap[i]    = t_2nd[i] - t_min[i]

If using higher-is-better scores:

gap[i] = score_winner - score_runnerup

Optional:

topk_scores[i]

---

## Stratification Factors

Group logs by:

- Draft count **K**
- **Watermark on/off**
- **CC on/off**
- identical temperature and sampling settings

---

## Statistics

For each configuration compute:

E[gap]  
median(gap)  
P(gap > τ)

for several thresholds τ.

---

## Visualizations

### Gap CDF

Compare distributions:

ERSD  
ERSD + WM  
ERSD − CC  
ERSD − CC + WM

Key question:

Does watermark shift the gap distribution right only when CC is enabled?

---

### Box / Violin plots

Grouped by:

K × watermark × CC

---

### Optional

gap vs token position

---

## Correlation Analysis

Compute correlation between:

gap[i]

and

reject indicator  
target evaluation count

Hypothesis chain:

larger gap  
→ more decisive race  
→ fewer rejects  
→ higher AATPS

---

# Experiment 2 — Acceptance Dynamics

## Goal

Understand whether watermark changes the **structure of speculative acceptance**.

Specifically test whether watermark leads to:

- fewer rejects
- later rejects
- longer accepted prefixes.

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

K  
watermark on/off  
CC on/off  
prompt set

---

## Core Statistics

Acceptance profile:

P(accepted_len ≥ r)

for

r = 1..K

Also compute:

E[accepted_len]  
E[num_retries]

---

## Visualizations

### Acceptance curves

P(accepted_len ≥ r)

Compare:

ERSD  
ERSD + WM  
ERSD − CC  
ERSD − CC + WM

Key question:

Does watermark increase acceptance depth only when CC is enabled?

---

### Reject position histogram

If watermark sharpens races we expect:

reject positions shift right

---

### Box plots

accepted_len

grouped by:

K × watermark × CC

---

# Experiment 3 — Decomposing AATPS Gain

## Goal

Identify the true source of AATPS improvement.

Possible explanations:

1. Algorithmic effect

fewer target evaluations

2. Implementation artifacts

cache  
branching  
kernel scheduling

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

proxy AATPS  
wall-clock AATPS

---

## Tables

Grouped by **K × CC × watermark**

| Metric | ERSD | ERSD+WM | ERSD−CC | ERSD−CC+WM |
|------|------|------|------|------|
| N_out | | | | |
| cost_target | ⭐ | ⭐ | | |
| draft_cost | | | | |
| wall_time/token | | | | |

---

## Plots

cost_target vs K

Key question:

Does watermark reduce target cost only when CC is active?

If yes, this supports the mechanism:

race sharpening → higher agreement → fewer target evaluations

---

# Experiment 4 — Gap–Acceptance Coupling

## Goal

Directly test whether **race sharpness (gap size) predicts speculative acceptance**.

This experiment links the stochastic race behavior to the final speculative efficiency.

---

## Procedure

For each generated token record:

gap[i]  
accept_indicator[i]

Where:

accept_indicator[i] = 1 if draft token accepted  
accept_indicator[i] = 0 otherwise

---

## Gap Binning

Partition tokens into gap intervals:

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

---

## Visualizations

### Acceptance vs gap curve

Plot:

accept_rate(gap_bin)

Expected pattern:

gap ↑ → acceptance probability ↑

---

### Separate curves for each method

Plot curves for:

ERSD  
ERSD + WM  
ERSD − CC  
ERSD − CC + WM

Key question:

Does watermark shift tokens into higher-gap regions?

---

### Gap distribution overlay

Overlay histograms:

gap distribution (ERSD)  
gap distribution (ERSD + WM)

Expected effect:

watermark shifts distribution toward larger gaps

---

# Expected Outcome

These four experiments triangulate the mechanism from complementary levels.

---

## Race-Level Explanation

Watermark **sharpens the stochastic race** by enlarging the winner–runner-up gap.

---

## Algorithm-Level Explanation

Sharper races produce:

higher agreement probability

between draft and target models.

---

## Speculative Dynamics Explanation

Higher agreement leads to:

longer accepted prefixes  
fewer rejects

---

## Systems-Level Explanation

Longer accepted prefixes reduce:

target evaluation cost

which increases **AATPS**.

---

# Mechanistic Hypothesis

The overall mechanism can be summarized as:

watermark bias  
→ sharper Poisson race  
→ larger winner–runner-up gap  
→ higher draft–target agreement  
→ longer speculative acceptance  
→ fewer target evaluations  
→ higher AATPS

---

# Implication

If confirmed, this result suggests that watermarking does **not necessarily conflict with speculative decoding**.

Instead, watermarking can act as a form of **structured stochastic coupling**, which may **improve speculative efficiency** under race-based decoding algorithms.