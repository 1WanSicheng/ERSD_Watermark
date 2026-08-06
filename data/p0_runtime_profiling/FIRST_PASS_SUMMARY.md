# P0 first-pass runtime result

This is a diagnostic run, not yet a paper-ready benchmark.

## Setup

- GPU: NVIDIA A100-SXM4 40GB
- Target: Qwen2.5-7B-Instruct
- Drafter: Qwen2.5-0.5B-Instruct
- `top_k=50`, `temperature=1`, `L=4`
- 5 fixed prompts, up to 64 output tokens per prompt
- 1 warm-up prompt
- End-to-end throughput is measured without phase synchronization.
- Phase attribution uses CUDA synchronization before and after each measured
  region.

## End-to-end throughput

| Method | B | tokens/s | vs target-only | vs INVARIANT | peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| Target-only | — | 39.21 | — | — | 15.19 |
| MPFR | 2 | 27.97 | -28.7% | -2.4% | 15.22 |
| INVARIANT | 2 | 28.66 | -26.9% | — | 15.29 |
| MPFR | 4 | 28.98 | -26.1% | -7.5% | 15.26 |
| INVARIANT | 4 | 31.32 | -20.1% | — | 15.41 |
| MPFR | 8 | 31.05 | -20.8% | -0.8% | 15.33 |
| INVARIANT | 8 | 31.29 | -20.2% | — | 15.64 |

## MPFR phase attribution

Percentages are fractions of synchronized, instrumented end-to-end time.
The RNG column is a subset of arrival sampling and must not be added again.

| B | target forward | draft forward | arrival sampling | keyed RNG subset | tree/cache/Python remainder |
|---:|---:|---:|---:|---:|---:|
| 2 | 21.95% | 65.97% | 2.47% | 0.30% | 9.61% |
| 4 | 21.33% | 65.07% | 3.13% | 0.37% | 10.47% |
| 8 | 20.77% | 63.56% | 4.02% | 0.47% | 11.65% |

## Approximate time per speculative block

| B | target forward | four draft depths | arrival sampling | remainder |
|---:|---:|---:|---:|---:|
| 2 | 26.50 ms | 79.66 ms | 2.98 ms | 11.61 ms |
| 4 | 26.22 ms | 80.00 ms | 3.85 ms | 12.88 ms |
| 8 | 26.17 ms | 80.07 ms | 5.06 ms | 14.68 ms |

One arrival-sampling call takes approximately 0.34 ms in all three cells.

## First conclusion

The dense `B × vocabulary` Poisson implementation is inefficient in
principle, but it is not the dominant runtime cost in this implementation at
`top_k=50`. Even at `B=8`, all arrival sampling accounts for about 4% of
end-to-end time, and keyed uniform generation itself accounts for less than
0.5%.

The dominant cost is the drafter: the decoder performs four sequential draft
depths per speculative block. Together these take about 80 ms per block,
roughly three times the target verification forward.

The next P0 step should therefore split the 80 ms draft component by tree
depth, batch width, and processed token positions, and compare the MPFR and
INVARIANT draft paths. Sparse keyed clocks may still be worthwhile, especially
for large `B`, but they cannot by themselves turn the current implementation
into an end-to-end speedup.

## Caveats

- Five prompts are sufficient for bottleneck localization, not final claims.
- The server uses an A100 rather than the paper's H100.
- Target-only uses Hugging Face's optimized generation loop, while MPFR and
  INVARIANT are custom Python decoders.
- INVARIANT can overshoot the requested output length by a few tokens at the
  final block; throughput is normalized by its actual emitted token count.
- A paper-ready table should use a real dataset, at least 100 prompts, repeated
  runs, and identical output-length handling.
