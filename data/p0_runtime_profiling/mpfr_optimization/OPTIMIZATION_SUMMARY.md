# MPFR token-rate optimization record

## Constraints

- Model pair: Qwen2.5-7B-Instruct / Qwen2.5-0.5B-Instruct
- Dataset: CNN/DailyMail
- Lookahead: 4; generation length: 128
- Widths: B = 4 and 8
- Correctness gate: exact generated token ids, per-block token ids and metadata,
  block count, and AATPS must match the frozen baseline.

## Retained implementation changes

- Lazily process target logits only along the realized verification path.
- Request only the logits positions used by draft and target verification.
- Thread one batched draft KV cache across tree depths; select one realized row
  only after verification.
- Construct only uncached suffix input ids and keep the Python context
  incrementally instead of copying the full sequence each block.
- Reuse per-context keyed sources, their SHA-256-derived seeds, and one CUDA
  generator.
- Use the equivalent log-domain winner for a single Poisson arrival.
- Reuse the already-computed top-k support for multi-arrival races while
  retaining the original full-vocabulary keyed noise tensor.
- Materialize repeated one-row KV caches with `expand(...).contiguous()` and
  skip materialization entirely for legacy batch-size-one caches.
- Skip KV `index_select` when child rows are already an identity mapping.
- Avoid retaining full-vocabulary output log-probability rows when callers do
  not request them. The public default remains unchanged.

## Exact regression

The final 10-prompt regression contains 20 cells (10 prompts × B={4,8}). All
20 cells have identical output tokens, per-block records, block count, and
AATPS relative to the frozen baseline.

| Version | B=4 TR | B=8 TR | Exact regression |
|---|---:|---:|---|
| Frozen baseline | 24.07 | 25.52 | reference |
| First stable optimization set | 24.89 | 27.13 | pass |
| Final candidate | 26.25 | 28.65 | pass |

The small regression set is used only as a correctness and optimization gate.
The final MPFR-versus-INVARIANT claim must use the paired 100-prompt run below.

## Final paired 100-prompt result

Same GPU, CNN/DailyMail, identical prompt list and seed protocol:

| B | Method | AATPS | TR (tok/s) | Latency (ms/block) | Incremental peak memory (MiB) |
|---:|---|---:|---:|---:|---:|
| 4 | MPFR | 3.076 | **27.07** | **113.65** | **449** |
| 4 | INVARIANT | 3.064 | 26.60 | 115.17 | 906 |
| 8 | MPFR | 3.334 | **28.43** | **117.25** | **839** |
| 8 | INVARIANT | 3.337 | 28.32 | 117.83 | 1812 |

MPFR is 1.75% faster at B=4 and 0.41% faster at B=8. The B=8 result should
be described as runtime parity or slightly faster, not as a significant
speedup. Relative to the pre-final-optimization paired run, MPFR TR increased
from 25.44 to 27.07 at B=4 and from 26.86 to 28.43 at B=8.

Raw result: `results/final_paired_B4_B8_n100.json`.

## Serial draft-depth diagnosis

The cached MPFR drafter performs four autoregressive forwards for lookahead 4.
Synchronized 10-prompt profiling gives:

| B | Depth | Mean batch rows | Draft forward (ms) |
|---:|---:|---:|---:|
| 4 | 1 / 2 / 3 / 4 | 1.00 / 2.02 / 2.62 / 3.05 | 20.36 / 20.22 / 20.30 / 20.34 |
| 8 | 1 / 2 / 3 / 4 | 1.00 / 2.86 / 4.25 / 5.24 | 20.44 / 20.27 / 20.31 / 20.34 |

The four drafter forwards total 81.23 ms/block at B=4 and 81.35 ms/block at
B=8. Work between draft depths plus the final pre-target interval is only
4.59 and 5.82 ms/block, respectively. Thus the remaining serial cost is the
strict autoregressive dependency: depth d's token must be sampled before the
input to depth d+1 exists. INVARIANT has the same four sequential drafter
forwards.

Two possible shortcuts were rejected rather than merged:

- Keeping fixed B cache rows removes gathers but changes the model batch shape;
  it changed AATPS on the sixth B=4 regression prompt.
- `torch.compile` on the dynamic draft/cache path did not finish one warmup
  prompt within five minutes because of dynamic-shape compilation/recompilation.

Raw diagnosis: `results/serial_depth_n10.json`. Eliminating the four-forward
dependency would require a different drafter architecture or tree-attention
implementation and is not an output-preserving engineering cleanup.
