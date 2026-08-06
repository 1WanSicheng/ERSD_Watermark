# Rebuttal record: single-draft efficiency and runtime

This document records the intended claim, current evidence, draft rebuttal
language, and the protocol for the larger formal run.

## 1. Claim hierarchy

The single-draft comparisons answer different questions:

| Comparison | Question answered |
|---|---|
| PFR vs PFR-NOWM | Does adding a recoverable key/watermark to the PFR coupling reduce acceptance or runtime? |
| PFR vs VSPS | What is the acceptance cost of no-communication, drafter-invariant coupling relative to maximal coupling? |
| PFR vs optimized AR | Does the current reference implementation realize end-to-end wall-clock acceleration? |
| PFR vs MSE/MWS | Where does PFR lie on the joint watermark-strength/acceptance frontier? |

The fair ablation for the paper's **efficiency-preserving watermarking** claim
is PFR versus PFR-NOWM. VSPS is not the unwatermarked version of PFR: it uses
recursive maximal coupling, whereas PFR uses no-communication exponential-race
coupling.

## 2. Existing 1000-prompt evidence: PFR vs PFR-NOWM

Values are from paper Tables 1 and 2.

| Dataset | L | NOWM AATPS | PFR AATPS | Δ AATPS | NOWM TR | PFR TR | Δ TR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN/DM | 1 | 1.623 | 1.625 | +0.12% | 32.147 | 33.321 | +3.65% |
| CNN/DM | 2 | 2.014 | 2.016 | +0.10% | 30.119 | 29.746 | -1.24% |
| CNN/DM | 3 | 2.270 | 2.272 | +0.09% | 26.709 | 25.999 | -2.66% |
| CNN/DM | 4 | 2.439 | 2.446 | +0.29% | 23.285 | 23.075 | -0.90% |
| ELI5 | 1 | 1.567 | 1.570 | +0.19% | 33.046 | 33.213 | +0.51% |
| ELI5 | 2 | 1.890 | 1.895 | +0.26% | 28.719 | 28.851 | +0.46% |
| ELI5 | 3 | 2.074 | 2.084 | +0.48% | 24.315 | 24.675 | +1.48% |
| ELI5 | 4 | 2.187 | 2.196 | +0.41% | 21.512 | 21.319 | -0.90% |

Observations:

- The AATPS differences are only 0.09--0.48% and have no practically
  meaningful watermark penalty.
- PFR TR differences range from -2.66% to +3.65%, with mixed signs.
- The arithmetic mean of the eight PFR/NOWM TR ratios is 1.0005
  (geometric mean 1.0003). This is descriptive, not a confidence interval,
  but it shows no systematic runtime loss from adding the key.

In expectation, PFR and PFR-NOWM use the same coupling law; finite-sample
trajectory differences can change block counts in either direction.

## 3. Full Tables 1/2 context: VSPS, MSE, PFR-NOWM, and PFR

These values are retained so the final rebuttal can choose between a compact
PFR/PFR-NOWM ablation and a broader contextual table.

### Table 1: Qwen/CNN-DailyMail

Each cell is `AATPS / token rate`.

| L | VSPS | MSE | PFR-NOWM | PFR |
|---:|---:|---:|---:|---:|
| 1 | 1.641 / 33.229 | 1.642 / 20.578 | 1.623 / 32.147 | 1.625 / 33.321 |
| 2 | 2.055 / 29.399 | 2.056 / 16.954 | 2.014 / 30.119 | 2.016 / 29.746 |
| 3 | 2.327 / 26.272 | 2.330 / 13.453 | 2.270 / 26.709 | 2.272 / 25.999 |
| 4 | 2.513 / 23.229 | 2.525 / 12.623 | 2.439 / 23.285 | 2.446 / 23.075 |

### Table 2: Qwen/ELI5

| L | VSPS | MSE | PFR-NOWM | PFR |
|---:|---:|---:|---:|---:|
| 1 | 1.593 / 32.090 | 1.595 / 18.218 | 1.567 / 33.046 | 1.570 / 33.213 |
| 2 | 1.948 / 27.350 | 1.956 / 14.867 | 1.890 / 28.719 | 1.895 / 28.851 |
| 3 | 2.159 / 23.619 | 2.164 / 13.359 | 2.074 / 24.315 | 2.084 / 24.675 |
| 4 | 2.294 / 17.142 | 2.292 / 13.200 | 2.187 / 21.512 | 2.196 / 21.319 |

### L=4 decomposition implied by the paper means

`blocks/128 = 128/AATPS`; `ms/block ≈ 1000 × AATPS/TR`. The latter is a
ratio of separately averaged quantities and is diagnostic rather than an
exact aggregate.

| Dataset | Method | AATPS | TR | blocks/128 | implied ms/block |
|---|---|---:|---:|---:|---:|
| CNN/DM | VSPS | 2.513 | 23.229 | 50.94 | 108.18 |
| CNN/DM | MSE | 2.525 | 12.623 | 50.69 | 200.03 |
| CNN/DM | PFR-NOWM | 2.439 | 23.285 | 52.48 | 104.75 |
| CNN/DM | PFR | 2.446 | 23.075 | 52.33 | 106.00 |
| ELI5 | VSPS | 2.294 | 17.142 | 55.80 | 133.82 |
| ELI5 | MSE | 2.292 | 13.200 | 55.85 | 173.64 |
| ELI5 | PFR-NOWM | 2.187 | 21.512 | 58.53 | 101.66 |
| ELI5 | PFR | 2.196 | 21.319 | 58.29 | 103.01 |

Interpretation:

- VSPS and MSE retain maximal-coupling-like acceptance and therefore have
  slightly higher AATPS than PFR.
- MSE's reported TR is much lower despite comparable AATPS, so its gap is
  per-block implementation cost rather than block count.
- PFR and PFR-NOWM have matching AATPS and TR; this is the direct support for
  the efficiency-preserving-watermark claim.
- PFR versus VSPS separates the small algorithmic acceptance cost of
  no-communication coupling from watermark overhead.

## 4. Runtime component diagnostic

Current A100 result on 10 real CNN/DailyMail prompts, `L=4`, top-k 50. These
numbers localize the bottleneck but must be replaced by the formal larger run
before submission.

| Component per block | PFR-NOWM | PFR |
|---|---:|---:|
| Target verification | 26.09 ms | 26.04 ms |
| Four draft forwards | 78.95 ms | 78.70 ms |
| Logits processing | 1.68 ms | 1.67 ms |
| All PFR arrival sampling | 2.11 ms | 2.06 ms |
| RNG subset | 0.32 ms fresh | 0.27 ms keyed |
| Cache/context/Python remainder | 3.49 ms | 3.47 ms |
| **Total** | **112.32 ms** | **111.94 ms** |

The keyed RNG is a subset of arrival sampling and must not be added twice.
It occupies approximately 0.24% of PFR block time.

This supports the narrow conclusion that attaching the key does not add a
measurable per-block compute cost. About 94% of block time is model forward
work shared by PFR and PFR-NOWM.

### GPU memory diagnostic

The A100 L sweep records incremental peak allocated memory above the loaded
model baseline. Values below are means across 10 prompts; parentheses contain
the maximum prompt.

| Dataset | L | VSPS | PFR-NOWM | PFR |
|---|---:|---:|---:|---:|
| CNN/DM | 1 | 134.0 (146.2) MiB | 134.3 (146.4) MiB | 133.7 (145.9) MiB |
| CNN/DM | 2 | 134.8 (146.5) MiB | 135.4 (147.1) MiB | 134.2 (146.0) MiB |
| CNN/DM | 3 | 135.4 (147.2) MiB | 136.3 (148.0) MiB | 134.6 (146.3) MiB |
| CNN/DM | 4 | 136.9 (148.2) MiB | 137.7 (149.4) MiB | 135.4 (147.1) MiB |
| ELI5 | 1 | 20.2 (28.6) MiB | 35.3 (40.0) MiB | 35.6 (40.9) MiB |
| ELI5 | 2 | 21.3 (29.3) MiB | 36.9 (41.9) MiB | 37.2 (41.6) MiB |
| ELI5 | 3 | 22.2 (30.2) MiB | 38.9 (44.3) MiB | 37.8 (42.2) MiB |
| ELI5 | 4 | 24.3 (32.2) MiB | 40.9 (45.6) MiB | 38.4 (43.5) MiB |

- PFR and PFR-NOWM differ by only 0.3--2.6 MiB, with mixed signs. There is no
  systematic GPU-memory cost from attaching the watermark key.
- PFR grows slowly with lookahead: only +1.7 MiB on CNN/DM and +2.7 MiB on
  ELI5 from L=1 to L=4.
- PFR and VSPS are effectively equal on CNN/DM. On ELI5, the PFR coupling
  path uses about 14--16 MiB more incremental memory than VSPS, but PFR-NOWM
  has the same behavior, so this is not watermark memory.
- Absolute peak allocation is about 15.2--15.3 GiB for every method because
  model weights dominate. Incremental peak allocation is the more informative
  decoder comparison.

## 5. Acceptance cost relative to VSPS

PFR should not be claimed to match maximal-coupling VSPS acceptance exactly.
Its no-communication coupling has a lower matching probability in general.

At L=4:

| Dataset | VSPS AATPS | PFR AATPS | PFR extra blocks per 128 tokens |
|---|---:|---:|---:|
| CNN/DM | 2.513 | 2.446 | +1.40 (+2.7%) |
| ELI5 | 2.294 | 2.196 | +2.49 (+4.5%) |

This is an algorithmic cost paid for no-communication coupling, stopping-time
drafter invariance, and coupling-level watermarkability. It is distinct from
the cost of adding the watermark key: PFR-NOWM has the same behavior.

## 6. What “AATPS is the primary metric” means

AATPS is

```text
output tokens / speculative blocks
```

and each speculative block contains one target verification invocation.
Equivalently, `1/AATPS` is approximately the number of target invocations per
output token. It therefore measures the communication- or
target-invocation-centric efficiency optimized by the theory.

For example, PFR on CNN/DM improves from AATPS 1.625 at L=1 to 2.446 at L=4:

- target invocations/token decrease from `1/1.625 = 0.615` to
  `1/2.446 = 0.409`;
- this is a 33.5% reduction in target invocations/token.

AATPS deliberately does **not** include:

- the number and latency of drafter forwards;
- target verification batch shape and kernel efficiency;
- Poisson or residual sampling;
- cache conversion/truncation;
- Python scheduling and synchronization;
- hardware and framework effects.

Therefore it is appropriate evidence for:

> Adding the watermark key does not degrade the target-invocation efficiency
> of the underlying PFR coupling.

It is not sufficient evidence for:

> The current implementation accelerates end-to-end generation.

The paper and rebuttal should call it “target-invocation efficiency measured
by AATPS,” rather than using unqualified “inference efficiency” where a reader
could reasonably expect wall-clock speedup.

## 7. Simple measurement standard for “almost no TR loss”

No specialized equivalence test is necessary for the rebuttal. We should
follow the reporting style already used in the paper and standard speculative
decoding experiments.

On the same prompt set, report for PFR and PFR-NOWM:

- AATPS, mean ± std across prompts;
- token rate in tokens/s, mean ± std across prompts;
- total blocks and blocks per 128 output tokens;
- wall-clock time per block;
- peak GPU memory;
- relative PFR-vs-NOWM difference for AATPS and TR.

For completeness, also report aggregate throughput
`sum(tokens)/sum(time)`, but it does not need a separate statistical analysis.

This is sufficient to support:

- “nearly identical AATPS”;
- “no systematic token-rate loss across the reported settings”;
- “no measurable per-block overhead in our runtime profile.”

Avoid only overly absolute wording such as “zero overhead” or “identical in
all executions,” since the two methods sample different finite trajectories.

## 8. Draft rebuttal language

### Compact version

> We thank the reviewer for raising the practical-overhead question. We
> clarify that our efficiency-preserving watermark claim is relative to the
> same unwatermarked PFR coupling (PFR-NOWM), rather than to maximal-coupling
> VSPS. Across the eight Qwen cells in Tables 1--2 (two datasets and four
> lookaheads), PFR and PFR-NOWM differ by only 0.09--0.48% in AATPS. Their
> token-rate differences have mixed signs, and the mean PFR/NOWM throughput
> ratio is 1.0005, indicating no systematic loss from adding the key.
>
> We additionally profiled runtime components on an A100. At L=4, PFR and
> PFR-NOWM require 111.94 and 112.32 ms per speculative block, respectively.
> Model forwards account for approximately 94% of runtime; all Poisson
> arrival sampling accounts for 1.8%, and keyed RNG itself for only 0.24%.
> Thus, the keyed watermark neither reduces speculative acceptance nor adds
> a measurable per-block runtime cost.

### Follow-on clarification about VSPS

> VSPS answers a different question. It uses token-level maximal coupling,
> while PFR uses no-communication exponential-race coupling to obtain
> drafter invariance and recoverable target-side randomness. The latter is
> not maximal in general, and PFR therefore uses 2.7% and 4.5% more blocks
> than VSPS at L=4 on CNN/DailyMail and ELI5, respectively. We will clarify
> this small algorithmic acceptance cost rather than attributing it to
> implementation overhead. The watermark itself introduces no further loss,
> as shown by PFR-NOWM.

### Wall-clock limitation

> We also agree that the current reference implementation does not
> consistently accelerate optimized autoregressive generation. The component
> profile shows that each additional lookahead adds approximately one 20 ms
> drafter forward, while the marginal AATPS gain diminishes. We will revise
> the paper to distinguish target-invocation efficiency (AATPS) from realized
> wall-clock acceleration and identify optimized drafter execution as future
> systems work.

## 9. Formal larger-run protocol

### End-to-end run

- Model pair: Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct.
- Datasets: CNN/DailyMail and ELI5 with the paper-exact prompt templates.
- Methods: PFR and PFR-NOWM as the primary paired comparison; VSPS retained
  as a maximal-coupling reference.
- `L ∈ {1,2,3,4}`, top-k 50, temperature 1, strict maximum 128 output tokens.
- Preferably use all 1000 paper prompts; at minimum use 100 for an initial
  rebuttal run.
- Use the same seed/key convention as the paper.
- Keep method order and warm-up protocol fixed and report them.
- Run at least 10 excluded warm-up prompts before timing.

Report:

- AATPS mean ± std;
- token rate mean ± std;
- relative PFR/NOWM AATPS and TR difference;
- aggregate throughput: `sum(output tokens) / sum(wall time)`;
- total blocks and blocks per 128 emitted tokens;
- mean time/block;
- exact emitted-token counts and EOS frequency;
- peak incremental allocated and reserved GPU memory.

The current paper tables use mean ± std of per-prompt values. Retain that
format for continuity and add aggregate throughput as a simple supplementary
system metric.

### Component run

- Use a representative subset of at least 100 prompts.
- Profile PFR and PFR-NOWM at L=4; optionally add L=1 to show scaling.
- Use CUDA events or `torch.profiler`, not per-call synchronized Python
  wrappers, for final timing attribution.
- Report absolute ms/block and ms/output-token, not percentages alone.
- Components: target forward, draft forward, logits processing, PFR arrival
  sampling, keyed/fresh RNG subset, cache/context/Python.

## 10. Claim and wording polish

Recommended:

- “preserves the acceptance efficiency of the underlying PFR coupling”;
- “no additional acceptance loss from watermarking”;
- “target-invocation efficiency, measured by AATPS”;
- “no systematic PFR-vs-PFR-NOWM token-rate loss”;
- “current reference implementation is not yet wall-clock optimized.”

Avoid:

- “PFR has no efficiency loss relative to VSPS”;
- “PFR matches maximal coupling”;
- “watermarking is free” without specifying the PFR-NOWM reference;
- “AATPS proves end-to-end acceleration”;
- “the lack of wall-clock speedup is entirely an engineering issue.”

The most defensible contribution statement is:

> No-communication PFR pays a small acceptance cost relative to maximal
> coupling, but once this coupling is chosen for drafter invariance, adding
> the recoverable watermark key causes no further acceptance or runtime loss.
