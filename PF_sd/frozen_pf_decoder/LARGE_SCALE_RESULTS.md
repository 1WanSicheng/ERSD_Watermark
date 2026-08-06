# Independent Latin-PF B/L sweep

## Status and protocol

All 18 jobs completed successfully with no OOM, traceback, or failed result:

- `B in {2, 4, 6}` and lookahead `L in {2, 4, 6}`;
- 1000 CNN/DailyMail prompts for each of seeds 7 and 107;
- Vicuna-7B-v1.5 target and Vicuna-68M drafter;
- up to 128 generated tokens, top-k 50, FP16;
- Latin PF reference, fused Latin PF, and INVARIANT;
- one complete excluded generation warm-up per method and setting;
- method order rotated across measured prompts.

Pooled AATPS is total generated tokens divided by total verification blocks.
Pooled TR is total generated tokens divided by total measured generation time.
The two seeds are pooled by summing their numerators and denominators, rather
than averaging per-prompt rates. Confidence intervals use 20,000 paired,
prompt-stratified bootstrap replicates, resampling independently within each
seed.

## Main results

`Latin PF` below means the fused implementation.

| B | L | INVARIANT AATPS | Latin PF AATPS | Delta AATPS | INVARIANT TR | Latin PF TR | Delta TR | Fusion gain over reference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 2.0111 | **2.0466** | **+1.76%** | 58.52 | **60.28** | **+3.01%** | +4.11% |
| 2 | 4 | 2.4000 | **2.4621** | **+2.59%** | 60.52 | 60.78 | +0.42% | +6.21% |
| 2 | 6 | 2.5879 | **2.6444** | **+2.18%** | **58.66** | 57.56 | -1.87% | +7.74% |
| 4 | 2 | 2.1530 | **2.1672** | **+0.66%** | 58.80 | **59.63** | **+1.41%** | +4.63% |
| 4 | 4 | 2.6471 | **2.6720** | **+0.94%** | **64.67** | 62.70 | -3.04% | +8.18% |
| 4 | 6 | 2.8752 | **2.9011** | **+0.90%** | **63.01** | 58.85 | -6.61% | +11.16% |
| 6 | 2 | 2.2264 | 2.2285 | +0.09% | 59.62 | **60.07** | **+0.75%** | +5.04% |
| 6 | 4 | 2.7828 | 2.7830 | +0.01% | **64.97** | 62.32 | -4.08% | +9.54% |
| 6 | 6 | **3.0435** | 3.0387 | -0.16% | **62.77** | 57.09 | -9.05% | +13.60% |

TR is in generated tokens per second. Bold deltas are statistically resolved
in the favorable direction; the complete intervals are below.

| B | L | Delta AATPS, 95% CI | Delta TR, 95% CI | Fusion gain, 95% CI |
|---:|---:|---:|---:|---:|
| 2 | 2 | +1.76% `[+1.34,+2.19]` | +3.01% `[+2.58,+3.44]` | +4.11% `[+4.09,+4.13]` |
| 2 | 4 | +2.59% `[+1.91,+3.27]` | +0.42% `[-0.24,+1.08]` | +6.21% `[+6.15,+6.28]` |
| 2 | 6 | +2.18% `[+1.37,+2.99]` | -1.87% `[-2.64,-1.11]` | +7.74% `[+7.68,+7.80]` |
| 4 | 2 | +0.66% `[+0.27,+1.06]` | +1.41% `[+1.02,+1.81]` | +4.63% `[+4.59,+4.68]` |
| 4 | 4 | +0.94% `[+0.29,+1.58]` | -3.04% `[-3.65,-2.45]` | +8.18% `[+8.11,+8.25]` |
| 4 | 6 | +0.90% `[+0.11,+1.70]` | -6.61% `[-7.32,-5.89]` | +11.16% `[+11.08,+11.25]` |
| 6 | 2 | +0.09% `[-0.28,+0.46]` | +0.75% `[+0.37,+1.13]` | +5.04% `[+4.96,+5.12]` |
| 6 | 4 | +0.01% `[-0.62,+0.64]` | -4.08% `[-4.66,-3.50]` | +9.54% `[+9.49,+9.60]` |
| 6 | 6 | -0.16% `[-0.94,+0.63]` | -9.05% `[-9.73,-8.36]` | +13.60% `[+13.51,+13.70]` |

## Runtime and memory diagnosis

| B | L | INVARIANT ms/block | Latin PF ms/block | Block-time delta | INVARIANT GiB | Latin PF GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 34.37 | **33.95** | -1.21% | 13.48 | 14.77 |
| 2 | 4 | **39.65** | 40.51 | +2.16% | 13.49 | 14.71 |
| 2 | 6 | **44.12** | 45.94 | +4.13% | 13.46 | 14.75 |
| 4 | 2 | 36.61 | **36.34** | -0.74% | 14.26 | 16.61 |
| 4 | 4 | **40.93** | 42.61 | +4.11% | 14.25 | 16.51 |
| 4 | 6 | **45.63** | 49.30 | +8.04% | 14.25 | 16.48 |
| 6 | 2 | 37.34 | **37.10** | -0.65% | 15.01 | 17.92 |
| 6 | 4 | **42.83** | 44.66 | +4.26% | 15.02 | 17.97 |
| 6 | 6 | **48.48** | 53.22 | +9.78% | 15.03 | 18.13 |

At `L=2`, fused Latin PF has slightly lower time per block than INVARIANT. As
lookahead grows, the number of visited PF target and draft contexts grows and
Latin-PF block time becomes larger. The AATPS improvement is not large enough
to compensate at the deeper settings. Peak memory is approximately 9--10%
higher at B=2, 16% higher at B=4, and 19--21% higher at B=6.

## Consistency audit

Across all 18,000 paired prompt/settings, reference and fused Latin PF have
zero mismatches in:

- generated token count and verification blocks;
- AATPS;
- PF ANLPPT, Li, PL, and mean-pivot statistics;
- target and draft context counts.

Thus the fusion gains are implementation gains and do not change the coupling
or watermark. The two seeds also agree on every qualitative comparison. For
example, fused-vs-reference TR gains differ by at most 0.24 percentage points
between seeds in all nine settings.

## Interpretation

1. **The clearest joint win is shallow lookahead.** At B=2,L=2 and B=4,L=2,
   Latin PF significantly improves both AATPS and end-to-end TR over INVARIANT.
2. **The largest relative AATPS gain is B=2,L=4.** It improves AATPS by 2.59%
   while TR is statistically tied with INVARIANT. This is the cleanest setting
   when acceptance efficiency is the primary objective.
3. **The Latin diversity benefit saturates with B.** At B=6, AATPS is
   indistinguishable from INVARIANT for every L. Increasing the number of Latin
   strata is therefore not producing additional coupling benefit in this model
   pair.
4. **Large L is a runtime trade-off, not a coupling failure.** AATPS rises with
   L, but verification work per block rises faster. Fused Latin-PF TR falls
   below INVARIANT by 1.87--9.05% at L=6.
5. **Fusion is increasingly valuable at harder settings.** It improves Latin
   PF TR by 4.11--13.60%, with larger gains at larger B and L, while preserving
   exact outputs and watermark statistics.

Within this grid, fused Latin PF has its highest absolute TR at `B=4,L=4`
(62.70 tok/s). Its highest AATPS is at `B=6,L=6` (3.0387), but that setting is
not attractive for wall-clock throughput. For a claim relative to INVARIANT,
`B=2,L=2` is the strongest joint AATPS/TR result, while `B=2,L=4` is the
strongest AATPS-focused result.

## Artifacts

- `results_n1000/`: the 18 raw JSON files and the completion manifest;
- `sweep_n1000_raw.tar.gz`: compressed raw-result bundle;
- `config_vicuna_sweep.json`: benchmark configuration;
- `run_sweep.py`: four-GPU dynamic scheduler.
