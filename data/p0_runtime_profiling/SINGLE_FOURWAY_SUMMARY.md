# Single-draft four-way runtime breakdown

Diagnostic first pass for rebuttal planning. This is not a paper-ready
benchmark.

## Setup

- NVIDIA A100-SXM4 40GB
- Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct
- `L=4`, effective `top_k=50`, temperature 1
- 5 fixed prompts, 64 output tokens, 1 warm-up prompt
- MSE is the paper's `mc_uwm_speed` path (`reweight_in_mc=False`)
- End-to-end throughput is uninstrumented.
- Phase attribution uses synchronized boundaries and is diagnostic only.

## End-to-end result

| Method | tokens/s | blocks | tokens/block | incremental peak allocated |
|---|---:|---:|---:|---:|
| VSPS | 24.46 | 117 | 2.74 | 18.6 MiB |
| MSE | 24.60 | 106 | 3.02 | 28.7 MiB |
| PFR-NOWM | 26.34 | 112 | 2.86 | 27.2 MiB |
| PFR | 25.75 | 114 | 2.81 | 24.5 MiB |

The prompt count is too small to interpret the raw acceptance differences.
PFR and PFR-NOWM use different random trajectories, and the instrumented pass
uses a separate seed range from the end-to-end pass.

## Cost per output token

Values come from the synchronized instrumented pass.

| Method | target | four draft depths | logits | PFR sampling | MSE watermark | accept/residual + basic sampling | remainder | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VSPS | 8.55 | 26.40 | 0.56 | — | — | 0.49 | 0.99 | 37.00 ms |
| MSE | 9.34 | 28.62 | 0.63 | — | 3.63 | 0.56 | 1.34 | 44.13 ms |
| PFR-NOWM | 9.11 | 27.27 | 0.60 | 0.81 | — | — | 1.21 | 39.00 ms |
| PFR | 9.26 | 27.80 | 0.61 | 0.80 | — | — | 1.23 | 39.69 ms |

## Cost per speculative block

This normalization is useful here only as a decoder-invocation micro-cost.
It is not the primary efficiency metric.

| Method | total/block | target/block | draft/block | method-specific sampling/block | other/block |
|---|---:|---:|---:|---:|---:|
| VSPS | 112.77 ms | 26.07 ms | 80.47 ms | — | 6.24 ms |
| MSE | 124.97 ms | 26.46 ms | 81.06 ms | 10.29 ms watermark | 7.16 ms |
| PFR-NOWM | 111.42 ms | 26.04 ms | 77.90 ms | 2.32 ms Poisson | 5.16 ms |
| PFR | 111.40 ms | 25.99 ms | 78.03 ms | 2.24 ms Poisson | 5.14 ms |

PFR and PFR-NOWM have indistinguishable cost per block in this run. PFR also
has approximately the same per-block cost as VSPS. The PFR watermark is
therefore not the source of the single-draft wall-clock gap.

## MSE watermark sub-phases

An additional MSE-only pass split its 2.35 ms average watermark step:

| Sub-phase | ms/call | share of watermark step |
|---|---:|---:|
| CPU NumPy full-vocabulary Gumbel construction | 1.68 | 71% |
| CPU-to-GPU code transfer | 0.24 | 10% |
| GPU reweight/argmax | 0.15 | 6% |
| Context extraction, hashing, and other | 0.28 | 12% |

There are about 4.3 watermark-step calls per MSE speculative block, yielding
roughly 10 ms of watermark work per block. Most of this measured cost is an
engineering artifact of constructing a full-vocabulary Gumbel vector in
NumPy and then copying it to the GPU. The algorithmic part is that MSE invokes
this construction at every draft depth and sometimes for a bonus token.

## Algorithmic versus engineering diagnosis

Using the nearby optimized target-only measurement of about 40.17 tokens/s,
one target autoregressive token costs roughly 24.9 ms. The current PFR block
costs about 111.4 ms, so it would need approximately 4.48 emitted tokens per
block to break even. With `L=4`, the absolute maximum is only five tokens.
The observed value is about 2.8.

Removing all PFR arrival-sampling time would reduce the block from about
111.4 to 109.2 ms and would still require about 4.39 tokens/block. Optimizing
the Poisson sampler alone therefore cannot produce end-to-end speedup.

The diagnosis has two parts:

1. **Structural:** a single-draft block requires `L` sequential drafter
   forwards. Acceptance determines whether their cost is amortized over
   enough emitted tokens.
2. **Engineering:** in this implementation, each 0.5B drafter step costs
   roughly 19.5--20 ms, so four steps cost about 78--80 ms. This is the
   dominant optimization target: cache layout, CUDA graphs/compile, kernel
   launch overhead, and a faster drafter path matter far more than sparse
   Poisson clocks.

For MSE, the additional full-vocabulary CPU Gumbel construction is primarily
an engineering issue, although its repeated invocation across draft depths is
an algorithmic requirement of that decoder.

## Required next validation

- Repeat on at least 100 real CNN/DailyMail prompts and three seeds.
- Report global throughput (`sum(tokens) / sum(time)`) with confidence
  intervals.
- Keep exact output lengths and effective logits processing identical.
- Record blocks/token and phase ms/token together; neither raw throughput nor
  time/block is sufficient alone.
- Use CUDA events or `torch.profiler` for the final phase table rather than
  synchronized Python wrappers.
