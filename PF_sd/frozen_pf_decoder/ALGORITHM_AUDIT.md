# Algorithm 2 implementation audit

## Equation-by-equation check

The implementation was checked against `ICLR_27_Speculative_PF_water.pdf`.

| Paper definition | Implementation | Status |
|---|---|---|
| `U_c^(b)(y) = PRF_k(c,b,y)` | context- and field-separated keyed torch RNG stream | matched |
| `X^(b)(c) = argmax(v_c(y)/T - log U_c^(b)(y))` | `argmin U/exp(v/T-max)` | algebraically identical |
| merge fields with the same draft prefix | `active_fields` and merged context levels | matched |
| `R_c(y)=min_b U_c^(b)(y)` | fieldwise minimum | matched |
| `U_c,B(y)=1-(1-R_c(y))^B` | `aggregate_min_uniform` | matched |
| target uses all B fields, independent of active drafts | target regenerates fields `range(B)` | matched |
| continue iff target token is in `D(c)` | membership in `tree.draft_tokens[c]` | matched |
| emit a bonus token after full coverage | covered-path bonus branch | matched |

An additional slow oracle now evaluates Equations (3)--(6) literally using
`argmax(logit/T - log(U))`, without tree batching or KV-cache reuse. The cached
decoder produces the same block tokens and accepted count.

Ten deterministic tests pass, including:

- literal Algorithm-2 block versus cached decoder;
- speculative output versus the fixed-B target decoder for two very different
  drafters;
- exact detector-pivot recovery.

## AATPS discrepancy audit

The original two-prompt result used only the first two rows, for which
Max-order PF happened to produce 64 tokens in 25 blocks: AATPS = 2.56. In the
five-prompt run, prompts 2 and 3 each had AATPS 1.882, reducing the aggregate to
2.254. The implementation and seed for the first two prompts did not change.

There was also a comparison bug in the diagnostic runner: legacy INVARIANT can
emit its entire final block and exceeded the 32-token limit. It was credited
with 35 tokens on three prompts, whereas Max-order PF and MPFR were capped at
32. The runner now crops every method to the same output budget before AATPS
and token-rate aggregation.

## Corrected matched result

Vicuna-7B/Vicuna-68M, CNN/DM, 20 prompts, 32 tokens per prompt, `B=2`, `L=4`,
temperature 1, top-k 50:

| Method | Tokens | Blocks | AATPS | Token rate | Runtime/block |
|---|---:|---:|---:|---:|---:|
| Max-order PF | 640 | 258 | 2.4806 | 56.32 tok/s | 44.0 ms |
| INVARIANT | 640 | 257 | 2.4903 | 59.55 tok/s | 41.8 ms |

The AATPS difference is 0.39%, not a large degradation. Max-order PF remains
about 5.4% lower in end-to-end token rate; this is explained by its roughly
5.3% higher runtime per block rather than by a loss of acceptance.

A literal full-vocabulary (`top-k=0`) five-prompt run gives AATPS 2.3188 for
Max-order PF and 2.3881 for INVARIANT. Thus top-k=50 is not the source of the
earlier apparent drop.

## Artifacts

- `results/audit_matched_n20.json`
- `results/audit_matched_n5.json`
- `results/literal_algorithm2_n5.json`
