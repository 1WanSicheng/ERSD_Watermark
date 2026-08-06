# Qwen3 model-pair speed sanity check

## Setup

- Target/drafter: `Qwen/Qwen3-14B` / `Qwen/Qwen3-1.7B`
- Their tokenizer mappings are exactly identical; model vocab size is 151,936.
- One A100-SXM4-40GB, FP16, CNN/DailyMail, the same 10 prompts per cell.
- 128 generated tokens, top-k 50, temperature 1.
- Basic-UWM, VSPS (`mc`), and PFR; lookahead in {1, 2, 4}.
- Only generation time is measured. Detection and quality metrics are disabled.

This is a small diagnostic, not a rebuttal-scale estimate. Basic-UWM and VSPS
completed before the combined process encountered a Qwen3 cache-API mismatch
when entering PFR. All 10 rows of every Basic-UWM/VSPS cell were complete and
are retained in `qwen3_cnn_n10.log`. PFR was then run separately after adding
new-Transformers compatibility for DynamicCache; its complete JSON and log are
retained alongside this file.

## Results

TR and AATPS are arithmetic means across the 10 prompt rows, matching the paper
aggregation. The final column is the diagnostic identity
`1000 * AATPS / TR`; it is not an independently instrumented component timer.

| Method | L | AATPS | TR (tok/s) | Diagnostic ms/block |
|---|---:|---:|---:|---:|
| Basic-UWM | -- | 1.000 | 22.317 | 44.81 |
| VSPS | 1 | 1.804 | 21.724 | 83.06 |
| VSPS | 2 | 2.375 | 21.139 | 112.37 |
| VSPS | 4 | 3.262 | 19.147 | 170.37 |
| PFR | 1 | 1.804 | 22.140 | 81.48 |
| PFR | 2 | 2.397 | 21.920 | 109.35 |
| PFR | 4 | 3.363 | 20.160 | 166.82 |

## Break-even interpretation

Basic-UWM takes approximately 44.81 ms/output token in this diagnostic. At a
given AATPS, an SD block must therefore finish within `AATPS * 44.81 ms` to
beat Basic-UWM.

| Method | L | Break-even block time | Observed diagnostic block time | Outcome |
|---|---:|---:|---:|---:|
| VSPS | 1 | 80.84 ms | 83.06 ms | 2.7% above break-even |
| VSPS | 2 | 106.44 ms | 112.37 ms | 5.6% above break-even |
| VSPS | 4 | 146.18 ms | 170.37 ms | 16.6% above break-even |
| PFR | 1 | 80.84 ms | 81.48 ms | 0.8% above break-even |
| PFR | 2 | 107.41 ms | 109.35 ms | 1.8% above break-even |
| PFR | 4 | 150.70 ms | 166.82 ms | 10.7% above break-even |

PFR at L=1 and L=2 is very close to break-even but does not cross it in this
reference implementation. Increasing L improves AATPS strongly, but each
additional autoregressive drafter depth adds latency while the marginal AATPS
gain shrinks. This reproduces the same mechanism seen with Qwen2.5 and shows
that Basic-UWM's reported TR is not explained by an obvious token-rate
calculation error.

The experiment also reinforces the matched-baseline result: PFR is faster than
VSPS at every tested lookahead (1.9%, 3.7%, and 5.3% at L=1,2,4 in this small
run). The remaining absolute AR gap is a system-level SD break-even issue, not
a PFR watermark penalty.

## Compatibility-only changes used for Qwen3

- Normalize the Transformers 4/5 `apply_chat_template` return type to the same
  input-id tensor.
- Use the repository's cache-length helper in Basic-UWM so both legacy tuple
  caches and DynamicCache are accepted.
- For new DynamicCache objects that expose `crop()` but not legacy public cache
  lists, crop the B=1 PFR cache in place.

These changes adapt APIs; they do not alter token distributions, coupling, or
the TR definition.
