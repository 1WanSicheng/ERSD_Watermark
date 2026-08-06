# P0 single-draft first-pass runtime result

This is a diagnostic run for rebuttal planning, not yet a paper-ready
benchmark. The numbers below come from `p0_single_topk50_corrected.json`.
The earlier `p0_single_first_pass.json` is retained for provenance but must
not be used: its scalar `top_k` setting was not consumed by the single-draft
decoder, so only its target-only baseline was actually truncated.

## Setup

- GPU: NVIDIA A100-SXM4 40GB
- Target: Qwen2.5-7B-Instruct
- Drafter: Qwen2.5-0.5B-Instruct
- `top_k=50`, `temperature=1`, `L=4`
- 5 fixed prompts, 64 output tokens per prompt
- 1 warm-up prompt
- End-to-end throughput is measured without phase synchronization.
- Phase attribution uses CUDA synchronization around each measured region.

## End-to-end throughput

| Method | tokens/s | vs target-only | vs PFR-NOWM | peak allocated GiB |
|---|---:|---:|---:|---:|
| Target-only | 40.17 | — | — | 15.193 |
| VSPS | 26.71 | -33.5% | — | 15.191 |
| PFR-NOWM | 26.87 | -33.1% | — | 15.199 |
| PFR | 26.01 | -35.3% | -3.2% | 15.197 |

The uninstrumented PFR and PFR-NOWM passes emitted the same number of output
tokens, but their sampled trajectories need not contain the same number of
speculative blocks. Therefore the observed 3.2% token-rate difference is not
by itself an estimate of the computational cost of watermarking.

## Instrumented phase attribution

The RNG columns are subsets of arrival sampling and must not be added to the
total again.

| Method | target forward | draft forward | arrival sampling | RNG subset | cache/Python remainder |
|---|---:|---:|---:|---:|---:|
| PFR-NOWM | 23.64% | 70.06% | 2.08% | 0.32% fresh | 4.22% |
| PFR | 23.45% | 70.32% | 2.00% | 0.26% keyed | 4.23% |

## Normalized costs

| Method | speculative blocks | time/block | target forward/call | draft forward/call | arrival sample/call | RNG/call |
|---|---:|---:|---:|---:|---:|---:|
| PFR-NOWM | 111 | 108.71 ms | 25.70 ms | 19.62 ms | 0.335 ms | 0.052 ms fresh |
| PFR | 114 | 109.78 ms | 25.74 ms | 19.60 ms | 0.325 ms | 0.043 ms keyed |

The synchronized PFR pass happened to require three more target verification
blocks than PFR-NOWM. After normalizing by target-forward calls, its total
time per speculative block was only 1.0% higher. The individual forward and
sampling costs are nearly identical.

## Rebuttal interpretation

This single-draft split supports a narrow runtime claim:

> In a first-pass A100 profile, replacing fresh randomness in PFR-NOWM with
> keyed watermark randomness did not create a material compute or memory
> overhead. Keyed RNG occupied 0.26% of instrumented runtime, and peak
> allocated memory changed by less than 0.01 GiB.

It does **not** support claiming that the current decoder is faster than
standard target-only generation. Both single-draft custom decoders achieved
about 25--27 tokens/s versus 39 tokens/s for Hugging Face target-only
generation. As in the multi-draft profile, model forwards—not the
vocabulary-wide Poisson/RNG operation—dominated runtime.

For a final rebuttal number, repeat this comparison on at least 100 dataset
prompts and report confidence intervals. Also report both raw token throughput
and costs normalized by verification blocks, since PFR and PFR-NOWM sample
different trajectories.

## Caveats

- Five prompts localize the bottleneck but are too few for a final statistical
  claim.
- The server uses an A100 rather than the paper's H100.
- Target-only uses Hugging Face's optimized generation loop; PFR, PFR-NOWM,
  and VSPS use custom Python decoders.
- Peak memory is PyTorch peak allocated memory, not total process or reserved
  GPU memory.
