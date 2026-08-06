# Multi-draft runtime: MPFR vs INVARIANT

## Comparison logic

For the multi-draft setting, INVARIANT is a more informative comparator than
VSPS because both MPFR and INVARIANT target drafter invariance through coupling
without communication. The comparison therefore controls for the key
algorithmic requirement that motivates MPFR.

It is still not an exact watermark-cost ablation: MPFR and INVARIANT use
different coupling primitives and different decoder implementations. We should
use this comparison to discuss end-to-end effectiveness and locate engineering
overhead, not claim that every runtime difference is caused by watermarking.

The safest novelty wording is:

> To our knowledge, MPFR is the first watermarkable multi-draft speculative
> sampling method that preserves drafter invariance.

Avoid the unqualified statement "the first method to combine watermarking and
speculative decoding," because prior single-draft methods already combine the
two.

### Natural narrative from single-draft PFR to MPFR

The single-draft result gives a clean motivation rather than a weakness to
hide:

1. PFR obtains watermarkability and drafter invariance through coupling
   without communication.
2. In the single-draft case, this coupling can have lower acceptance than
   maximal coupling, although PFR and PFR-NOWM show that watermarking itself
   adds almost no further throughput loss.
3. Multi-draft sampling supplies multiple candidate continuations and raises
   AATPS while retaining the no-communication property.
4. MPFR is the resulting watermarkable multi-draft construction. INVARIANT is
   then the natural algorithm-level baseline because it shares the
   no-communication/drafter-invariance goal, while MPFR additionally provides
   a recoverable watermark.

This narrative separates two questions cleanly: AATPS tests whether MPFR uses
multiple drafts effectively under the invariance constraint; component timing
tests the cost of the current implementation.

## What the existing paper tables tell us

Tables 5--6 already show that acceptance efficiency is not the main problem:

- On CNN/DailyMail, MPFR and INVARIANT have almost the same AATPS
  (MPFR differs by -0.18% to +0.70%).
- On ELI5, MPFR has 1.75%--2.50% higher AATPS.

Therefore, the lower TR cannot be explained by MPFR consistently producing
fewer tokens per speculative block. The direct timing experiment below tests
where the additional block latency is spent.

## A100 component profiling

Setup: Qwen2.5-7B-Instruct target, Qwen2.5-0.5B-Instruct drafter, top-k 50,
lookahead 4, 128 requested new tokens, 10 prompts per dataset, one warmup
prompt, NVIDIA A100-SXM4-40GB. End-to-end measurements have no profiling
hooks. Component attribution is measured in a separate CUDA-synchronized pass.

The 10-prompt runs are intended for bottleneck diagnosis, not to replace the
paper-scale AATPS estimates. In particular, their AATPS ordering varies with
the finite prompt/sample set, while the per-block component pattern is stable
across CNN/DailyMail and ELI5.

### Large-scale run protocol

For every prompt, record token rate, AATPS, directly measured milliseconds per
block, peak GPU allocated memory, and incremental peak allocated memory above
the loaded-model baseline. Report mean/std and the maximum across prompts.

Reset CUDA peak-memory statistics immediately before generation and synchronize
CUDA around the measured region. Use allocated memory as the primary metric,
since allocator reservation can persist across methods. Keep the model pair,
prompt order, L, B, output length, dtype, and decoding parameters identical
for MPFR and INVARIANT.

### Direct end-to-end timing

`ms/block` below is directly measured as total wall-clock generation time
divided by the actual number of speculative blocks. It is not inferred from
the paper table.

#### CNN/DailyMail

| B | Method | TR | AATPS | ms/block | Incremental peak memory |
|---:|---|---:|---:|---:|---:|
| 2 | MPFR | 21.48 | 2.672 | 124.36 | 317 MiB |
| 2 | INVARIANT | 23.78 | 2.777 | 116.82 | 453 MiB |
| 4 | MPFR | 23.10 | 2.969 | 128.52 | 608 MiB |
| 4 | INVARIANT | 24.62 | 2.902 | 117.86 | 904 MiB |
| 6 | MPFR | 23.73 | 3.133 | 132.06 | 830 MiB |
| 6 | INVARIANT | 25.48 | 3.066 | 120.37 | 1,350 MiB |
| 8 | MPFR | 23.95 | 3.241 | 135.37 | 1,103 MiB |
| 8 | INVARIANT | 27.39 | 3.330 | 121.58 | 1,802 MiB |

#### ELI5

| B | Method | TR | AATPS | ms/block | Incremental peak memory |
|---:|---|---:|---:|---:|---:|
| 2 | MPFR | 20.03 | 2.444 | 121.99 | 90 MiB |
| 2 | INVARIANT | 21.54 | 2.488 | 115.51 | 221 MiB |
| 4 | MPFR | 22.09 | 2.773 | 125.54 | 170 MiB |
| 4 | INVARIANT | 25.22 | 2.906 | 115.22 | 437 MiB |
| 6 | MPFR | 23.76 | 3.045 | 128.14 | 271 MiB |
| 6 | INVARIANT | 24.79 | 2.869 | 115.72 | 655 MiB |
| 8 | MPFR | 23.93 | 3.132 | 130.89 | 346 MiB |
| 8 | INVARIANT | 26.66 | 3.090 | 115.91 | 874 MiB |

The directly measured MPFR block is 6.47--14.98 ms slower across the eight
configurations. We next split this measured difference into runtime
components.

### Where the additional block time goes

The component pass synchronizes CUDA around each component. The table reports
`MPFR time - INVARIANT time`; a positive value means that component is slower
in MPFR.

| Dataset | B | Direct block gap | Target + draft | Logits | Sampler | Other non-model |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 2 | +7.54 ms | -0.93 ms | +1.39 ms | +2.05 ms | +5.54 ms |
| CNN | 4 | +10.66 ms | -0.99 ms | +2.18 ms | +3.17 ms | +6.90 ms |
| CNN | 6 | +11.70 ms | -2.38 ms | +2.80 ms | +4.03 ms | +8.00 ms |
| CNN | 8 | +13.79 ms | -2.25 ms | +3.41 ms | +4.85 ms | +9.11 ms |
| ELI5 | 2 | +6.47 ms | -1.27 ms | +1.44 ms | +2.05 ms | +5.03 ms |
| ELI5 | 4 | +10.32 ms | -0.72 ms | +2.43 ms | +3.41 ms | +6.62 ms |
| ELI5 | 6 | +12.43 ms | -1.01 ms | +3.25 ms | +4.55 ms | +7.74 ms |
| ELI5 | 8 | +14.98 ms | -1.17 ms | +4.02 ms | +5.56 ms | +8.87 ms |

This is the direct data support for why MPFR's TR is lower:

- Target and draft model computation is not slower.
- MPFR spends more time in logits processing and PFR arrival sampling.
- The largest difference is the current other non-model bookkeeping.
- These costs grow with B, matching the observed growth in the block-time gap.

The component times come from a separate synchronized diagnostic pass, so
their row sum need not equal the uninstrumented direct block gap exactly. Their
role is to localize the gap, while TR and direct ms/block come from the
uninstrumented pass.

### Is the gap caused by watermarking?

The current MPFR--INVARIANT comparison is not an exact watermark ablation,
because the two methods use different coupling primitives. We should therefore
not claim that it proves zero watermark overhead.

It does, however, directly show that keyed watermark randomness is not the
main bottleneck:

- The complete keyed uniform RNG operation takes only 0.36--0.77 ms/block.
- This is 0.3%--0.6% of MPFR's total block time.
- It is approximately 5% of the observed MPFR--INVARIANT block-time gap.
- This timer includes ordinary random-tensor generation as well as key-based
  seeding, so the incremental cost attributable specifically to keying is no
  larger than this measurement.

Two representative configurations make the distinction clear:

| Dataset, B | Direct block gap | Target + draft | Logits | PFR-vs-Gumbel sampler | Keyed RNG subset | Other non-model |
|---|---:|---:|---:|---:|---:|---:|
| CNN, B=4 | +10.66 ms | -0.99 ms | +2.18 ms | +3.17 ms | 0.51 ms | +6.90 ms |
| ELI5, B=8 | +14.98 ms | -1.17 ms | +4.02 ms | +5.56 ms | 0.77 ms | +8.87 ms |

`Keyed RNG subset` is already included in the PFR sampler time and must not be
added again.

The supported conclusion is therefore:

> The lower TR is mainly due to the current non-model implementation path,
> rather than watermark key generation. In particular, MPFR performs more
> per-context logits/sampling work and Python-side bookkeeping than the dense
> INVARIANT implementation. These costs grow with B, while keyed RNG remains
> below 0.8 ms/block.

Part of the PFR-versus-Gumbel sampler difference comes from the different
coupling primitive itself, so it is safer to call the overall gap "current
implementation and coupling cost," not purely watermark overhead and not
purely timing noise.

### Full component measurements

#### CNN/DailyMail synchronized component time (ms/block)

| B | Method | Target | Draft | Logits | Sampler | Other non-model |
|---:|---|---:|---:|---:|---:|---:|
| 2 | MPFR | 27.46 | 82.29 | 2.21 | 2.96 | 11.38 |
| 2 | INVARIANT | 27.55 | 83.13 | 0.83 | 0.91 | 5.84 |
| 4 | MPFR | 28.41 | 82.67 | 3.02 | 4.12 | 12.84 |
| 4 | INVARIANT | 28.76 | 83.31 | 0.84 | 0.95 | 5.94 |
| 6 | MPFR | 29.04 | 82.92 | 3.69 | 5.00 | 14.08 |
| 6 | INVARIANT | 30.75 | 83.59 | 0.89 | 0.98 | 6.08 |
| 8 | MPFR | 30.05 | 82.99 | 4.34 | 5.85 | 15.29 |
| 8 | INVARIANT | 32.43 | 82.87 | 0.94 | 1.00 | 6.18 |

#### ELI5 synchronized component time (ms/block)

| B | Method | Target | Draft | Logits | Sampler | Other non-model |
|---:|---|---:|---:|---:|---:|---:|
| 2 | MPFR | 27.15 | 81.01 | 2.27 | 2.95 | 10.81 |
| 2 | INVARIANT | 26.85 | 82.58 | 0.82 | 0.89 | 5.78 |
| 4 | MPFR | 26.56 | 81.57 | 3.26 | 4.35 | 12.58 |
| 4 | INVARIANT | 26.40 | 82.46 | 0.83 | 0.94 | 5.95 |
| 6 | MPFR | 26.61 | 81.72 | 4.13 | 5.51 | 13.82 |
| 6 | INVARIANT | 26.57 | 82.77 | 0.88 | 0.96 | 6.08 |
| 8 | MPFR | 26.79 | 81.42 | 4.94 | 6.53 | 15.04 |
| 8 | INVARIANT | 26.70 | 82.68 | 0.93 | 0.97 | 6.17 |

Across all eight profiled configurations:

- MPFR end-to-end block latency is 6.5--15.0 ms higher, with an average
  difference of 11.0 ms.
- Target and draft forward time do not explain the gap. Their average
  MPFR-minus-INVARIANT differences are -0.5 and -0.9 ms/block.
- The stable diagnostic differences are logits processing
  (+1.4--4.0 ms/block), PFR arrival sampling versus Gumbel sampling
  (+2.0--5.6 ms/block), and other non-model bookkeeping
  (+5.0--9.1 ms/block).
- Keyed uniform generation is only 0.36--0.77 ms/block and is already included
  inside PFR arrival sampling. This is only about 5% of the observed block
  gap, so the watermark key itself is not the principal bottleneck.
- MPFR's incremental peak allocation is 39%--70% of INVARIANT's across these
  configurations. The current MPFR path trades some latency for substantially
  lower memory growth as B increases.

## What the result supports

1. The paper-scale AATPS results show that MPFR preserves competitive
   multi-draft acceptance efficiency under the same no-communication/drafter-
   invariant goal as INVARIANT.
2. The observed TR difference is primarily a per-block implementation
   difference, not evidence that MPFR needs systematically more blocks.
3. Model-forward costs are already comparable. The remaining optimization
   targets are non-model components: batched/reused logits transforms,
   vectorized PFR arrivals, and reduced Python/tree/cache bookkeeping.
4. The keyed watermark randomness is a small part of the measured latency.
5. MPFR has a useful memory advantage at larger B, which can be reported
   alongside TR if space permits.

## What not to claim

- Do not say that MPFR has "zero runtime overhead" relative to INVARIANT.
- Do not present INVARIANT as an exact no-watermark version of MPFR.
- Do not claim that the entire difference is merely timing noise: the
  6--15 ms/block difference is stable and measurable.
- Do not use the 10-prompt profiling run to override the paper-scale AATPS
  estimates.
- Do not make the unqualified novelty claim that MPFR is the first method
  combining watermarking and speculative decoding.

## Candidate rebuttal paragraph

> We thank the reviewer for raising the runtime question. Tables 5--6 show that
> MPFR's multi-draft acceptance efficiency is comparable to INVARIANT, the
> closest baseline sharing our no-communication/drafter-invariance objective:
> on CNN/DailyMail, MPFR's AATPS differs by only -0.18% to +0.70%; on ELI5 it
> is 1.75%--2.50% higher. We additionally measured block latency directly on
> an A100. Across B=2,4,6,8 and both datasets, MPFR takes 6.47--14.98 ms more
> per block. Component profiling shows comparable target and drafter forward
> times and localizes the difference to the current PFR arrival/logits
> processing and other non-model bookkeeping. The complete keyed RNG operation
> is only 0.36--0.77 ms per block (0.3%--0.6% of MPFR's block time and about
> 5% of the observed gap), so watermark key generation is not the main
> bottleneck. Thus, the TR gap is not caused by systematically lower
> multi-draft acceptance or by keying, but primarily by identifiable
> non-model implementation and coupling costs.

## Files

- Profiler: `profile_multi_two_way.py`
- CNN/DailyMail raw result: `p0_multi_cnn_B_sweep_n10.json`
- ELI5 raw result: `p0_multi_eli5_B_sweep_n10.json`
