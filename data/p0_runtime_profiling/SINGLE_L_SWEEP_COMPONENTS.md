# Single-draft runtime components across lookahead

Diagnostic reproduction of the Qwen model pair used in paper Tables 1 and 2.
This is a 10-prompt bottleneck study, not a replacement for the paper's
1000-prompt acceptance estimates.

## Setup

- A100-SXM4 40GB
- Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct
- `top_k=50`, temperature 1, maximum 128 output tokens
- Real CNN/DailyMail and ELI5 prompts using the experiment harness templates
- Methods: VSPS, PFR-NOWM, PFR
- `L ∈ {1,2,3,4}`

## CNN/DailyMail: PFR

The tokens/block and TR columns are from the uninstrumented pass. Component
times are from a separate synchronized pass.

| L | tokens/block | TR | time/block | target | draft | PFR sampling | logits + cache/Python |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.61 | 33.38 | 49.20 ms | 25.49 | 20.42 | 0.86 | 2.44 |
| 2 | 1.99 | 28.97 | 70.08 ms | 25.53 | 39.85 | 1.30 | 3.40 |
| 3 | 2.22 | 24.77 | 91.25 ms | 26.01 | 59.27 | 1.69 | 4.28 |
| 4 | 2.36 | 21.56 | 111.94 ms | 26.04 | 78.70 | 2.06 | 5.14 |

The 10-prompt TR values reproduce the direction and approximate scale of
Table 1. The paper's 1000-prompt PFR AATPS values are
`1.625, 2.016, 2.272, 2.446`, showing the same diminishing marginal gain.

## ELI5: PFR

| L | tokens/block | TR | time/block | target | draft | PFR sampling | logits + cache/Python |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.53 | 31.72 | 49.12 ms | 25.48 | 20.37 | 0.84 | 2.43 |
| 2 | 1.82 | 26.67 | 69.64 ms | 25.52 | 39.47 | 1.26 | 3.39 |
| 3 | 1.99 | 22.51 | 90.43 ms | 25.96 | 58.57 | 1.64 | 4.26 |
| 4 | 2.09 | 19.24 | 110.71 ms | 25.98 | 77.61 | 2.00 | 5.12 |

The measured block costs are almost dataset invariant. The current 10-prompt
acceptance estimates are noisy and should not be compared directly with the
paper's 1000-prompt AATPS.

## PFR versus VSPS and PFR-NOWM

CNN synchronized time/block:

| L | VSPS | PFR-NOWM | PFR |
|---:|---:|---:|---:|
| 1 | 49.53 ms | 49.22 ms | 49.20 ms |
| 2 | 70.38 ms | 70.08 ms | 70.08 ms |
| 3 | 91.15 ms | 91.50 ms | 91.25 ms |
| 4 | 111.18 ms | 112.32 ms | 111.94 ms |

All three decoders have essentially the same per-block runtime at every L.
PFR therefore does not introduce a meaningful per-step wall-clock penalty
relative to VSPS, and adding the key does not increase the PFR-NOWM cost.

## Why AATPS rises while TR falls

For PFR on CNN, increasing L from 1 to 4:

- raises the paper's AATPS from 1.625 to 2.446 (+50.5%);
- lowers target invocations/token from 0.615 to 0.409 (-33.5%);
- raises measured block time from 49.2 to 111.9 ms (+127.5%).

The block-time increase is approximately 62.7 ms:

- draft forward increases by 58.3 ms (93% of the increase);
- PFR sampling increases by only 1.2 ms (2%);
- logits/cache/Python account for the remaining increase.

Each additional lookahead position costs roughly one 19.5--20 ms drafter
forward, while the marginal AATPS gains diminish. Under the current runtime
ratio, reducing target invocations is therefore insufficient to improve
wall-clock throughput.

## Interpretation

The data support a precise, limited conclusion:

1. AATPS is the appropriate primary metric for the paper's theoretical claim
   about target-model invocation/communication efficiency.
2. PFR and PFR-NOWM have matching AATPS in the paper, so watermarking does not
   degrade that efficiency.
3. PFR, PFR-NOWM, and VSPS have matching time/block in the profile, so the PFR
   coupling and key are not responsible for the low Qwen token rate.
4. The AATPS-to-TR gap comes from the current cost ratio: a single 0.5B draft
   step costs about 20 ms versus about 26 ms for one batched 7B verification.

The logical requirement for L sequential draft steps is structural and shared
by standard single-draft speculative decoding. Their unusually high measured
cost relative to target verification is an implementation/model-hardware
issue with clear optimization space, but an actual optimized implementation
would be needed to claim a realized future speedup.

## Table 2 note

The current ELI5 VSPS `L=4` result (22.22 tok/s, 112.4 ms/block) does not
reproduce the paper's unusually low 17.142 tok/s cell. The paper values imply
approximately 133.8 ms/block for only that VSPS cell, whereas the other
lookaheads and the current implementation follow the approximately 20 ms per
added-depth trend. Full raw 1000-prompt timing files are not present in this
checkout, so this isolated discrepancy cannot yet be attributed to prompt
outliers, aggregation, or a prior code version.

The PFR-side conclusion is unaffected: its measured block-time scaling is
nearly identical on CNN and ELI5 and is dominated by draft forwards.
