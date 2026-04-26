# MPFR `wmin` Stopping Analysis

This note records the current diagnosis for slow finite MPFR calls in
`pfr_mpfr_firstarrival.py`, especially when the proposal distribution inside
MPFR is the draft model.

## Where `wmin` Appears

The finite MPFR implementation follows the standard stopping rule:

```text
continue while |H| < B or max_score(H) > t * wmin
```

For one sample, this is effectively:

```text
stop when S_star <= t * wmin
```

where `t` is the proposal Poisson-process time and `S_star` is the best mapped
score seen so far.

In the code, for target distribution `P` and proposal distribution `Q`:

```python
ratio = exp(target_logprobs - proposal_logprobs)      # r = P / Q
log_inv_ratio = proposal_logprobs - target_logprobs   # log(Q / P)
wmin = exp(min(log_inv_ratio))                        # inf_u Q(u) / P(u)
```

So the implemented quantity is:

```text
wmin = inf_u 1 / r(u) = inf_u Q(u) / P(u)
```

This direction is correct for the MPFR algorithm.

## Why It Becomes Slow

The stopping threshold is `t * wmin`. If `wmin` is very small, the threshold
grows very slowly. Then MPFR needs a very large proposal time `t` before the
current best candidate can be certified as final.

Roughly, if `S_star` is order 1, stopping needs:

```text
t >= S_star / wmin
```

So:

```text
wmin = 1e-3  -> around 1e3 proposal-time scale
wmin = 1e-6  -> around 1e6 proposal-time scale
```

This is why a few bad contexts can require hundreds of thousands or even one
million proposals.

## Full-Vocabulary Minimum Is the Problem

For language models, `wmin` is computed over the full vocabulary:

```text
wmin = min over all tokens u of Q(u) / P(u)
```

This is a worst-case global bound. It can be tiny even when `P` and `Q` are
close on high-probability tokens. A single token with:

```text
P(u) >> Q(u)
```

makes:

```text
Q(u) / P(u)
```

very small, and that one token tightens the stopping threshold for the entire
MPFR call.

In recent GSM8K profiling with Qwen target/draft and `proposal=draft_model`,
target-side MPFR had examples like:

```text
log10(wmin) = -6.86  -> hit 1,000,000 proposals
log10(wmin) = -6.74  -> hit 1,000,000 proposals
log10(wmin) = -6.24  -> about 606,208 proposals
log10(wmin) = -6.01  -> about 661,504 proposals
```

Across one short run:

```text
verify MPFR mean log10(wmin): about -3.34
verify MPFR min  log10(wmin): about -6.86
verify MPFR mean proposals:  about 102,887
```

The draft-side MPFR was fast in the same setting because `P = Q` there, so the
same-distribution fast path stops after one proposal. The slow calls are mainly
target-side verification and bonus MPFR, where:

```text
P = target model
Q = draft model proposal
```

## Implication

Using an optimizer to compute the exact infimum does not by itself fix the
speed issue. For a discrete vocabulary, the exact full-range optimizer is
equivalent to:

```text
min_u Q(u) / P(u)
```

and that exact value is often too small.

The bottleneck is therefore not an incorrect `wmin` direction. The bottleneck is
that the exact full-vocabulary worst-case `wmin` produces a very conservative
stopping threshold.

## Possible Directions

To make this faster, one of the following has to change:

- Use a restricted support or top-k/top-p proposal support, with the resulting
  distributional change handled explicitly.
- Use a tighter bound that is local to a partition or support subset instead of
  the whole vocabulary.
- Batch/GPU-vectorize the proposal loop to reduce implementation overhead, though
  this will not remove the fundamental `1 / wmin` scaling.
- Use `proposal=model` for same-distribution MPFR fast paths where appropriate.
- Accept an approximate or truncated MPFR variant and measure the bias/quality
  tradeoff.

The current exact full-vocabulary finite MPFR implementation is correct, but
when `inf_u Q(u)/P(u)` is around `1e-6`, the stopping condition is intrinsically
too strict for fast decoding.
