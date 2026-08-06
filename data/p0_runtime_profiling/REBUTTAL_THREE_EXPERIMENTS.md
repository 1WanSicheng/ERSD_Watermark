# Rebuttal evidence: three additional experiments

This document keeps the three experiments separate and records the metric
definitions, measured data, and short conclusions that are safe to use in the
rebuttal.

## Reporting conventions

- **AATPS** and **TR** follow the definitions used in the paper. TR is
  reported in token/s over the complete generation procedure.
- **End-to-end latency/block** is the total generation latency divided by the
  number of speculative decoding blocks.
- **Component latency** is the average time per block spent in the target
  model, draft model, sampling procedure, or other decoding operations.
- **Incremental peak GPU memory** is the additional peak allocation during
  decoding after the models have been loaded. It is reported as an independent
  resource metric, not as an explanation for TR.
- **ANLPPT-U/Li/PL** measure watermark evidence; higher is stronger.
- **LPPL** is the mean negative log-probability under the target decoding
  distribution; lower is better.

Runtime profiling and watermark/quality evaluation use the same prompts and
decoding configuration but are run separately, so metric evaluation is not
included in TR.

---

## Experiment 1: detailed MPFR overhead analysis

### Question

> A detailed overhead analysis (e.g., compute, memory, runtime,
> implementation complexity) is helpful.

### Setup

- Models: Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct.
- Hardware: one NVIDIA A100-SXM4-40GB per experimental cell.
- Data: the same 100 CNN/DailyMail prompts.
- Decoding: \(L=4\), top-\(k=50\), temperature 1, at most 128 new tokens.
- Comparison: MPFR versus INVARIANT at \(B\in\{4,8\}\).

### End-to-end runtime

| \(B\) | Method | AATPS ↑ | TR ↑ | End-to-end latency/block ↓ |
|---:|---|---:|---:|---:|
| 4 | INVARIANT | 3.064 | 24.347 | 125.83 ms |
| 4 | MPFR | 3.076 | 24.008 | 128.12 ms |
| 8 | INVARIANT | 3.337 | 25.422 | 131.25 ms |
| 8 | MPFR | 3.334 | 24.602 | 135.52 ms |

MPFR and INVARIANT have matched AATPS, so the TR difference is not explained
by lower acceptance or more target-verification blocks. At \(B=4/8\), MPFR
is 1.4%/3.2% lower in TR and 2.29/4.27 ms higher in end-to-end latency per
block.

### GPU memory

| \(B\) | Method | Incremental peak GPU memory ↓ |
|---:|---|---:|
| 4 | INVARIANT | 910.2 MiB |
| 4 | MPFR | 609.3 MiB |
| 8 | INVARIANT | 1816.0 MiB |
| 8 | MPFR | 1182.1 MiB |

In the current implementations, MPFR uses 33.1% less incremental peak memory
at \(B=4\) and 34.9% less at \(B=8\). This is an implementation-level result:
INVARIANT eagerly stores future float32 Gumbel randomness, whereas MPFR
generates arrival randomness on demand. Streaming INVARIANT's randomness
could reduce this difference. We report this as a separate resource result;
the profile does not establish that the memory difference causes the observed
TR difference.

### Compute and runtime components

The table reports each component's share of the separately profiled block
latency. End-to-end latency is reported only in the preceding table.

| \(B\) | Method | Target inference | Draft inference | Sampling | Other decoding |
|---:|---|---:|---:|---:|---:|
| 4 | INVARIANT | 23.8% | 69.7% | 1.5% | 5.0% |
| 4 | MPFR | 21.6% | 63.1% | 5.3% | 10.0% |
| 8 | INVARIANT | 25.8% | 67.5% | 1.6% | 5.1% |
| 8 | MPFR | 21.7% | 59.9% | 7.2% | 11.2% |

`Sampling` includes logits preparation and the method-specific sampling
operation (Poisson arrival generation/selection for MPFR and Gumbel sampling
for INVARIANT). `Other decoding` includes proposal-tree construction,
acceptance, and KV-cache/control operations.

Model inference accounts for 84.7% and 81.6% of MPFR's profiled block
latency at \(B=4\) and \(B=8\). The additional non-model time is distributed
between MPFR sampling and other decoding operations rather than arising from
additional model calls.

### Implementation-complexity interpretation

| Part | INVARIANT | MPFR | Empirical implication |
|---|---|---|---|
| Model execution | Target tree verification plus \(L\) draft steps | Same high-level target/draft schedule | Model latency is not higher for MPFR in this profile |
| Sampling primitive | Logits preparation, Gumbel sampling, and selection | Logits preparation, Poisson arrivals, and first-arrival selection across drafts | MPFR sampling accounts for 5.3%/7.2% at \(B=4/8\) |
| Multi-draft control | Proposal tree, verification, cache update | Proposal tree, verification, context-conditioned PFR sources, cache update | Their combined cost is included in tree/cache/control and is not individually attributed |
| Randomness storage | Current code eagerly stores future Gumbel tensors | Current code generates arrivals on demand | MPFR has lower incremental peak memory in this implementation |

We do not use source-code line count as a complexity metric: it is sensitive
to refactoring and does not measure runtime work. The component timers and
peak allocation give a more reproducible account of implementation overhead.

### Concise conclusion

> MPFR preserves essentially the same target-invocation efficiency as
> INVARIANT. Its current implementation pays a small end-to-end latency cost
> from sampling and other decoding operations. As a separate resource result,
> the current MPFR implementation uses roughly one-third less incremental
> peak memory at \(B=4\) and \(B=8\).

---

## Experiment 2: Qwen2.5-72B scale experiment

### Setup

- Target/drafter: Qwen2.5-72B-Instruct / Qwen2.5-0.5B-Instruct.
- Hardware: 8 NVIDIA A100 40GB GPUs with identical balanced sharding for all
  methods.
- Data: the same 100 filtered CNN/DailyMail prompts.
- Decoding: \(L=4\), top-\(k=50\), temperature 1, at most 128 new tokens.
- TR and AATPS come from the runtime evaluation. ANLPPT and LPPL come from a
  separate metric evaluation.
- No-watermark methods are scored with the corresponding detector to provide
  the null reference; they do not embed a watermark.

### Results

| Regime | Method | \(B\) | AATPS ↑ | TR ↑ | ANLPPT-U ↑ | ANLPPT-Li ↑ | ANLPPT-PL ↑ | LPPL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Single | VSPS | 1 | 2.380 | 10.025 | 0.0031 | 0.0027 | 0.0031 | 0.2429 |
| Single | PFR-NOWM | 1 | 2.339 | 9.838 | 0.0024 | 0.0023 | 0.0023 | 0.2364 |
| Single | PFR | 1 | 2.321 | 9.771 | 0.0178 | 0.0121 | 0.0190 | 0.2357 |
| Multi | INVARIANT | 4 | 3.057 | 10.328 | 0.0018 | 0.0019 | 0.0016 | 0.2645 |
| Multi | MPFR | 4 | 2.988 | 10.218 | 0.0156 | 0.0134 | 0.0154 | 0.2428 |
| Multi | INVARIANT | 8 | 3.307 | 9.633 | 0.0019 | 0.0024 | 0.0016 | 0.2609 |
| Multi | MPFR | 8 | 3.274 | 9.922 | 0.0160 | 0.0125 | 0.0164 | 0.2465 |

### Concise conclusion

> The method remains practical at 72B under identical 8-GPU sharding. MPFR
> is within 1.1% of INVARIANT in TR at \(B=4\) and is 3.0% faster at \(B=8\),
> while their AATPS values remain close. PFR/MPFR retain a clear watermark
> signal under all three score families, and LPPL remains comparable to the
> no-watermark references.

---

## Experiment 3: top-k scaling and the proposed latency bottleneck

### Reviewer hypothesis

> Token rate shows MPFR is slower than basic autoregressive. The overhead of
> generating and sorting \(|\mathrm{top}\text{-}k|\times B\) exponentials
> plus watermark bookkeeping outweighs the AATPS gains. The overhead is
> dominated by target inference, draft inference, and watermark embedding.

### Setup

- Models, hardware, prompts, output length, and \(L\) match Experiment 1.
- MPFR is evaluated at \(B\in\{4,8\}\) and
  top-\(k\in\{50,100,500,\mathrm{None}\}\).
- Each runtime cell uses one uncontended GPU.
- Watermark scoring is executed in a separate pass.
- To follow the reviewer's proposed decomposition conservatively, the
  **watermark/PFR sampling** bucket includes the complete exponential-arrival
  generation and first-arrival selection, not only key construction.
- The three categories proposed by the reviewer do not cover proposal-tree
  construction, acceptance, cache operations, tensor movement, and control
  flow. We therefore retain a **tree/cache/control** category for operations
  outside model inference and sampling/coupling.

This sweep diagnoses MPFR's top-\(k\) sensitivity and latency attribution. It
does not contain a newly timed autoregressive cell under the identical
profiling harness, so we do not use it to manufacture a new MPFR-versus-AR
number; the end-to-end comparison cited by the reviewer remains the submitted
result.

### End-to-end efficiency and watermark signal

| \(B\) | top-\(k\) | AATPS ↑ | TR ↑ | End-to-end latency/block ↓ | ANLPPT-U ↑ |
|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 3.076 | 24.008 | 128.12 | 0.0489 |
| 4 | 100 | 3.062 | 24.124 | 126.93 | 0.0507 |
| 4 | 500 | 3.055 | 23.935 | 127.64 | 0.0512 |
| 4 | None | 3.057 | 23.821 | 128.35 | 0.0512 |
| 8 | 50 | 3.334 | 24.602 | 135.52 | 0.0490 |
| 8 | 100 | 3.322 | 24.893 | 133.46 | 0.0504 |
| 8 | 500 | 3.311 | 25.114 | 131.85 | 0.0507 |
| 8 | None | 3.306 | 24.538 | 134.74 | 0.0511 |

Removing top-\(k\) changes MPFR TR by \(-0.8\%\) at \(B=4\) and
\(-0.3\%\) at \(B=8\) relative to top-\(k=50\). AATPS changes by 0.6%/0.8%,
and ANLPPT-U remains stable or slightly increases. There is no observed
end-to-end latency jump when \(k\) is increased to 500 or removed.

### TR component decomposition following the reviewer

The table reports each component's share of the separately profiled block
latency. End-to-end latency remains the value in the preceding table.

| \(B\) | top-\(k\) | Target inference | Draft inference | MPFR sampling | Other decoding |
|---:|---:|---:|---:|---:|---:|
| 4 | 50 | 21.6% | 63.1% | 5.3% | 10.0% |
| 4 | 100 | 21.6% | 62.9% | 5.5% | 10.1% |
| 4 | 500 | 21.7% | 63.1% | 5.4% | 9.9% |
| 4 | None | 21.9% | 64.0% | 4.0% | 10.1% |
| 8 | 50 | 21.7% | 59.8% | 7.2% | 11.2% |
| 8 | 100 | 21.8% | 59.6% | 7.3% | 11.3% |
| 8 | 500 | 21.8% | 59.6% | 7.2% | 11.4% |
| 8 | None | 22.3% | 60.8% | 5.4% | 11.6% |

`MPFR sampling` includes logits preparation, exponential-arrival generation,
and first-arrival selection. `Other decoding` includes proposal-tree
construction, acceptance, and KV-cache/control operations.

The reviewer's decomposition is directionally correct that model inference
dominates: target plus draft model latency accounts for 81.6%--85.9% of
profiled block latency. However:

1. MPFR sampling does not increase as top-\(k\) grows. Its share is stable
   from \(k=50\) to \(k=500\) and becomes slightly smaller without top-\(k\)
   filtering.
2. Other decoding operations also contribute non-negligible latency, so the
   runtime cannot be described solely by target inference, draft inference,
   and watermark embedding.
3. These non-model costs can still outweigh a modest AATPS gain. Therefore,
   the evidence supports a qualified target-invocation-efficiency claim, not
   a universal wall-clock speedup claim.

### Concise conclusion

> Across top-\(k=50,100,500\), and no truncation, MPFR's AATPS, TR, and
> watermark signal remain stable. Target and draft inference dominate total
> latency. MPFR sampling shows no measured growth with \(k\), while the
> remaining non-model latency is distributed across sampling and
> tree/cache/control operations.

---

## Raw artifacts

- Experiment 1 runtime and components:
  `rebuttal_runtime_n100_cnn_paired/multi_cnn_dailymail_B{4,8}_k50.json`
- Experiment 2 runtime:
  `rebuttal_qwen72_scale/n100/{single_n100,multi_n100}.json`
- Experiment 2 watermark/quality:
  `rebuttal_qwen72_scale/metrics_n100/{single_metrics_n100,multi_metrics_n100}.json`
- Experiment 3 runtime and components:
  `rebuttal_runtime_n100_cnn_paired/multi_cnn_dailymail_B{4,8}_k{50,100,500,none}.json`
- Experiment 3 watermark signal:
  `rebuttal_topk_signal_n100/multi_k{50,100,500,none}.json`
