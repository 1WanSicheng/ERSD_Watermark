# Why PFR token rate differs from VSPS

The exact decomposition is

```text
token rate = tokens/block ÷ time/block
           = AATPS × block rate.
```

PFR can therefore be slower because it emits fewer tokens per block, because
each block is slower, or both. The previous component profile established
only the second term. This note closes the decomposition.

## Algorithmic acceptance difference

VSPS uses recursive token-level maximal coupling, whose one-token matching
probability is `1 - TV(P,Q)`. PFR uses shared exponential races, a
no-communication coupling needed for its drafter-invariant keyed
construction. This coupling is not maximal in general. The paper itself notes
the lower no-communication matching bound in Eq. (4).

Consequently, PFR has slightly lower AATPS than VSPS and MSE in Tables 1 and
2. This is an algorithmic coupling difference, not watermark runtime
overhead.

## Paper Table 1: Qwen/CNN

| L | VSPS AATPS | PFR AATPS | PFR AATPS gap | VSPS blocks/128 | PFR blocks/128 | extra PFR blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.641 | 1.625 | -1.0% | 78.00 | 78.77 | +0.77 |
| 2 | 2.055 | 2.016 | -1.9% | 62.29 | 63.49 | +1.20 |
| 3 | 2.327 | 2.272 | -2.4% | 55.01 | 56.34 | +1.33 |
| 4 | 2.513 | 2.446 | -2.7% | 50.94 | 52.33 | +1.40 |

At L=4, the paper means imply:

- PFR emits 2.7% fewer tokens/block;
- PFR executes blocks about 2.1% faster;
- the two effects nearly cancel, giving 23.075 versus 23.229 tokens/s
  (`-0.7%`).

Thus the small Table 1 PFR–VSPS TR difference is acceptance-driven, not
per-block overhead.

## Paper Table 2: Qwen/ELI5

| L | VSPS AATPS | PFR AATPS | PFR AATPS gap | VSPS blocks/128 | PFR blocks/128 | extra PFR blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.593 | 1.570 | -1.4% | 80.35 | 81.53 | +1.18 |
| 2 | 1.948 | 1.895 | -2.7% | 65.71 | 67.55 | +1.84 |
| 3 | 2.159 | 2.084 | -3.5% | 59.29 | 61.42 | +2.13 |
| 4 | 2.294 | 2.196 | -4.3% | 55.80 | 58.29 | +2.49 |

PFR again needs more blocks. It nevertheless has higher reported TR in Table
2 because its reported time/block is substantially lower than VSPS. The
unusually low Table 2 VSPS L=4 rate is not reproduced by the current checkout
and requires the original raw timing files to explain fully.

## Same-run CNN diagnostic

The 10-prompt uninstrumented L=4 pass provides a direct same-run identity:

| Method | output tokens | blocks | blocks/token | ms/block | tokens/s |
|---|---:|---:|---:|---:|---:|
| VSPS | 1074 | 423 | 0.394 | 109.93 | 23.10 |
| PFR | 1066 | 451 | 0.423 | 109.63 | 21.56 |

PFR's block is 0.3% faster, but it uses 7.4% more blocks/token in this small
sample. Its 6.6% lower TR is therefore entirely explained by the sampled
acceptance/block-count difference. The 1000-prompt paper estimate is more
reliable for the magnitude and puts the L=4 difference at 2.7%.

## PFR versus MSE

PFR is not slower than MSE in token rate in Tables 1 or 2. At L=4:

- Table 1: PFR 23.075 versus MSE 12.623 tokens/s;
- Table 2: PFR 21.319 versus MSE 13.200 tokens/s.

MSE has slightly higher AATPS because its speed endpoint retains maximal
coupling, but its current watermark implementation has much higher
per-block runtime. This is separate from the PFR–VSPS explanation.

## Correct causal conclusion

There are two distinct effects:

1. **PFR versus VSPS:** PFR's no-communication coupling has slightly lower
   acceptance, so it runs approximately 2.7--4.5% more blocks in the paper's
   L=4 Qwen cells. This is an algorithmic cost paid for drafter invariance and
   coupling-level watermarkability.
2. **Speculative decoding versus optimized AR:** both PFR and VSPS have high
   block cost because each extra lookahead adds an approximately 20 ms
   drafter forward. This is the dominant reason AATPS improvements do not
   become wall-clock speedups in the current Qwen implementation.

Therefore AATPS can support the claim that adding the watermark does not
degrade PFR (PFR versus PFR-NOWM), but it cannot support a claim that PFR has
the same acceptance as maximal-coupling VSPS. The rebuttal should acknowledge
the small block-count penalty and separately show that PFR adds no per-block
runtime penalty.
