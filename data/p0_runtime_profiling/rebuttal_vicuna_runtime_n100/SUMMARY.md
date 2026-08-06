# Vicuna runtime diagnosis (100 prompts)

## Setup

- Target/drafter: `lmsys/vicuna-7b-v1.5` / `double7/vicuna-68m`
- Dataset: the same first 100 filtered CNN/DailyMail prompts
- Decoding: \(L=4\), top-\(k=50\), at most 128 new tokens, FP16
- Seven cells ran simultaneously on seven A100-40GB GPUs. Each process had
  one exclusive GPU and a disjoint set of CPU cores local to that GPU's NUMA
  node. OMP, MKL, and OpenBLAS were restricted to one thread.
- Each cell used 100 uninstrumented prompts for headline TR and another 100
  instrumented prompts for component attribution.

## Headline results

| Regime | Method | \(B\) | TR (global) | TR (per-prompt mean) | AATPS | Measured ms/block | Incremental peak memory |
|---|---|---:|---:|---:|---:|---:|---:|
| Single | VSPS | 1 | 56.63 | 57.61 | 2.184 | 38.56 | 303.5 MiB |
| Single | PFR-NOWM | 1 | 53.55 | 54.30 | 2.125 | 39.68 | 884.1 MiB |
| Single | PFR | 1 | 54.47 | 55.44 | 2.159 | 39.65 | 880.7 MiB |
| Multi | INVARIANT | 4 | 64.43 | 65.09 | 2.673 | 41.48 | 1269.9 MiB |
| Multi | MPFR | 4 | 54.97 | 55.65 | 2.695 | 49.02 | 3510.8 MiB |
| Multi | INVARIANT | 8 | 60.37 | 61.05 | 2.925 | 48.46 | 2531.8 MiB |
| Multi | MPFR | 8 | 50.61 | 51.21 | 2.926 | 57.81 | 6636.2 MiB |

## Component breakdown

All entries below are directly measured instrumented milliseconds per block.
`Sampling` is MC accept/residual plus basic sampling for VSPS, PFR arrival
sampling for PFR/MPFR, and Gumbel sampling for INVARIANT. Keyed RNG is a
subset of PFR arrival sampling and is therefore shown separately rather than
added to the total.

| Method | \(B\) | Model forwards | Logits | Sampling | Remainder | Instrumented total | Keyed RNG (subset) |
|---|---:|---:|---:|---:|---:|---:|---:|
| VSPS | 1 | 34.28 | 1.58 | 1.26 | 2.15 | 39.27 | -- |
| PFR-NOWM | 1 | 34.29 | 1.56 | 1.82 | 2.84 | 40.52 | -- |
| PFR | 1 | 34.21 | 1.58 | 1.87 | 2.85 | 40.52 | 0.23 |
| INVARIANT | 4 | 36.35 | 0.73 | 0.84 | 4.27 | 42.18 | -- |
| MPFR | 4 | 36.26 | 2.69 | 3.59 | 7.65 | 50.20 | 0.46 |
| INVARIANT | 8 | 42.53 | 0.74 | 0.84 | 5.54 | 49.65 | -- |
| MPFR | 8 | 40.97 | 3.98 | 5.27 | 9.53 | 59.76 | 0.68 |

## What explains the TR gaps?

### Single draft

- PFR and PFR-NOWM have essentially identical measured time per block:
  39.65 versus 39.68 ms. PFR has slightly higher AATPS in this run, so it is
  also slightly faster in TR. The watermark key is not the source of a
  slowdown.
- Relative to VSPS, PFR has 1.1% lower AATPS and 2.8% higher time per block;
  together these produce a 3.8% TR gap.
- Target and draft forward times are identical across the three methods.
  PFR's approximately 1.25 ms/block instrumented difference from VSPS is
  localized to arrival sampling (about +0.61 ms) and Python/bookkeeping
  remainder (about +0.70 ms). Keyed RNG itself is only 0.23 ms/block, or
  0.58% of total time.

### Multi draft

- AATPS is not the explanation. MPFR is slightly better than INVARIANT at
  \(B=4\) (2.695 versus 2.673) and identical at \(B=8\).
- At \(B=4\), MPFR takes 7.55 ms more per block end to end. In the
  instrumented pass, model-forward time is identical. The difference is
  explained by logits processing (+1.97 ms), arrival versus Gumbel sampling
  (+2.76 ms), and tree/cache/Python remainder (+3.39 ms).
- At \(B=8\), MPFR takes 9.35 ms more per block end to end. MPFR actually
  saves 1.56 ms in model forwards, but this is outweighed by logits processing
  (+3.25 ms), arrival sampling (+4.43 ms), and remainder (+4.00 ms).
- Keyed RNG is only 0.46--0.68 ms/block (0.92--1.13% of MPFR time) and is a
  subset of arrival sampling. The multi-draft gap is therefore not a heavy
  watermark-embedding cost. It comes from the current dense
  logits/Poisson-arrival path and Python/tree/cache implementation.

## Why the model-pair gap is larger than on Qwen

The fixed non-model MPFR overhead is not larger in absolute terms on Vicuna.
For example, the \(B=4\) MPFR--INVARIANT non-model difference is about
8.1 ms/block on Vicuna versus about 11.8 ms/block in the matched Qwen
component profile. It matters more because:

1. Vicuna model forwards are much cheaper. At \(B=4\), target plus draft
   forwards take about 36 ms/block on Vicuna versus 112--120 ms/block on
   Qwen. The same order of fixed overhead is therefore a much larger fraction
   of total latency.
2. On Qwen, MPFR's model-forward path was about 8--10 ms/block faster than
   INVARIANT, offsetting much of its non-model overhead. On Vicuna the two
   model-forward paths are equal at \(B=4\), and MPFR saves only 1.6 ms at
   \(B=8\). The non-model difference is therefore exposed almost directly.

This supports an amortization/implementation explanation, not an acceptance
or watermark-key explanation.

## Relation to the submitted tables

The submitted Vicuna/CNN single-draft Table 3 reports per-prompt mean TR
53.94 for VSPS, 40.08 for PFR-NOWM, and 41.80 for PFR at \(L=4\). The new
controlled profile gives 57.61, 54.30, and 55.44, respectively. Thus the
large submitted single-draft gap does not reproduce with the current shared
cached PFR path.

Repository history provides a likely reason: the Vicuna single-draft data
commit (`f690fd3`, May 1) predates the B1/B2/B3 sampling-path optimization
commit (`8103ea6`, May 2). This is an internal provenance diagnosis; rebuttal
wording should describe the new run as a controlled implementation update
without claiming that the submitted measurements were invalid.

For multi draft, the controlled run still shows a genuine per-block gap:
14.7% in TR at \(B=4\) and 16.2% at \(B=8\). This is larger than the
7--9% gap in submitted Table 7, but the component profile now explains it
directly and shows that it is unrelated to AATPS or keyed watermark
generation.

## Memory caveat

Unlike the Qwen runs, the current Vicuna implementation gives MPFR a much
higher incremental peak than INVARIANT. The runtime profile alone localizes
neither allocation nor lifetime, so this should not yet be attributed to a
specific tensor. A CUDA allocation trace would be required before making a
mechanistic memory claim for this model pair.
