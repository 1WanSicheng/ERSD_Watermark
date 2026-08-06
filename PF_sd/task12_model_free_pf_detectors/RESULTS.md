# Model-free PF detector results

## Setup

- Saved Vicuna PF generations; no new generation and no model forward.
- 100 prompts per seed, seeds 7 and 107.
- Latin PF, lookahead 4, top-k 50, temperature 1.0.
- 200,000 i.i.d.-uniform null sequences for every detector and scored length.
- TPR is measured at theoretical keyed-null FPR 1% and 0.1%.
- Values below pool the two seeds by taking the mean of their prompt-level
  aggregate results (200 watermarked sequences per B).

## B = 2

| Detector | Mean ANLPPT | TPR @ 1% FPR | TPR @ 0.1% FPR |
|---|---:|---:|---:|
| Original | **0.0423** | 32.5% | **16.0%** |
| Li, rho=0.05 | 0.0351 | 24.5% | 11.5% |
| Li, rho=0.10 | 0.0375 | 29.5% | 12.5% |
| Li, rho=0.20 | 0.0373 | **33.0%** | 12.5% |
| Li, rho=0.50 | 0.0337 | 26.5% | 7.5% |
| Power-law, eps=0.01 | 0.0392 | 30.0% | 14.5% |
| Power-law, eps=0.02 | 0.0393 | 30.5% | 14.0% |
| Power-law, eps=0.05 | 0.0379 | **33.0%** | 13.0% |
| Power-law, eps=0.10 | 0.0353 | 29.0% | 12.0% |

The apparent 0.5 percentage-point gain at 1% FPR is one additional detected
sequence out of 200 and is not a meaningful improvement. Original is better
on mean ANLPPT and at the stricter 0.1% FPR.

## B = 4

| Detector | Mean ANLPPT | TPR @ 1% FPR | TPR @ 0.1% FPR |
|---|---:|---:|---:|
| Original | **0.0344** | **25.0%** | **9.5%** |
| Li, rho=0.05 | 0.0289 | 19.0% | 7.0% |
| Li, rho=0.10 | 0.0321 | 21.0% | 7.5% |
| Li, rho=0.20 | 0.0318 | 23.0% | 8.0% |
| Li, rho=0.50 | 0.0277 | 17.0% | 5.5% |
| Power-law, eps=0.01 | 0.0329 | 23.5% | 8.0% |
| Power-law, eps=0.02 | 0.0331 | 21.5% | 8.5% |
| Power-law, eps=0.05 | 0.0322 | 23.5% | 7.0% |
| Power-law, eps=0.10 | 0.0297 | 19.5% | 5.5% |

## Conclusion

The PDF methods are implemented correctly and the recovered raw scores match
the frozen benchmark records. On this real multiclass, temperature-1.0
setting, however, neither fixed-rho Li nor fixed-epsilon power-law gives a
stable improvement over the original detector. The power-law theorem concerns
a binary low-signal regime and does not claim uniform dominance.

The most useful next model-free check is a text-length sweep and a genuinely
pre-specified adaptive detector with joint null calibration. Choosing the best
rho or epsilon after observing the watermarked set would invalidate the stated
FPR and must not be reported as a detector improvement.
