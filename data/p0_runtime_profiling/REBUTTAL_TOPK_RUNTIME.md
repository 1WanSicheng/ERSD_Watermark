# Rebuttal record: top-k scaling and no-top-k runtime

## 1. Clean serial MSE/MWS comparison

Status: completed successfully on 2026-07-27. These are the MSE/MWS numbers to
use in analysis and rebuttal. Do not use the earlier concurrently executed
MSE/MWS token rates.

### Protocol

- Hardware: one NVIDIA A100-SXM4-40GB GPU.
- Models: Qwen2.5-7B-Instruct target and Qwen2.5-0.5B-Instruct drafter.
- Data: the same 100 CNN/DailyMail prompts for every method and pass.
- Generation: `L=4`, at most 128 new tokens, temperature 1, top-p 1.
- Conditions: top-k 50 and no top-k truncation.
- Methods: MSE and MWS.
- Seed protocol: one fixed seed assignment per prompt, shared across methods
  and across the end-to-end and instrumented passes.
- Execution: all four method/condition cells ran serially on the same GPU.
  There were no other experiment workers on the server.
- Measurement: an uninstrumented 100-prompt pass for token rate and memory,
  followed by an instrumented pass over the same 100 prompts for component
  attribution.

### End-to-end results

| top-k | Method | TR (tok/s) | AATPS | Direct ms/block | Incremental peak memory |
|---:|---|---:|---:|---:|---:|
| 50 | MSE | 20.15 | 2.491 | 123.62 | 157.94 MiB |
| 50 | MWS | 19.36 | 2.393 | 123.57 | 162.71 MiB |
| None | MSE | 20.31 | 2.477 | 121.94 | 157.94 MiB |
| None | MWS | 19.31 | 2.375 | 123.00 | 162.71 MiB |

Removing top-k changes MSE token rate by `+0.8%` and MWS token rate by `-0.3%`.
AATPS changes by less than `0.8%`, direct block time changes by less than
`1.4%`, and incremental peak memory is unchanged.

### Instrumented component time

All values below are directly measured milliseconds per decoding block.
`Other` combines acceptance/residual sampling, basic sampling, and the
unattributed control-flow remainder.

| top-k | Method | Target forward | Draft forward | Logits processing | Gumbel watermark step | Other |
|---:|---|---:|---:|---:|---:|---:|
| 50 | MSE | 27.13 | 82.02 | 1.80 | 9.38 | 5.50 |
| 50 | MWS | 27.17 | 82.30 | 1.80 | 9.56 | 5.81 |
| None | MSE | 27.06 | 81.97 | 0.08 | 9.17 | 5.42 |
| None | MWS | 27.08 | 81.97 | 0.08 | 9.72 | 5.72 |

The Gumbel watermark step is stable at `9.17--9.72 ms/block`, or
`7.4--7.8%` of instrumented block time. Removing top-k eliminates most of the
top-k logits-processing cost but does not introduce additional Gumbel
watermark latency.

### Replacement rule for the earlier parallel run

The first top-k sweep launched eight independent GPU jobs concurrently. MSE
and MWS include CPU/NumPy work, so those jobs competed for shared CPU
resources. The resulting watermark-step measurements varied from about 9 to
54 ms/block and caused non-algorithmic token-rate jumps. Those concurrent
MSE/MWS token-rate cells are diagnostic only and must not be reported.

The serial results above remove that confound: the target, draft, Gumbel
watermark, block-time, and token-rate measurements are stable between top-k
50 and no top-k.

### Raw artifacts

- `rebuttal_runtime_n100_cnn_paired/mse_mws_serial/single_cnn_L4_k50_mse_mws_serial.json`
- `rebuttal_runtime_n100_cnn_paired/mse_mws_serial/single_cnn_L4_knone_mse_mws_serial.json`
- Matching execution logs are stored beside the JSON files as `k50.log` and
  `knone.log`.

## 2. Rebuttal-ready statement for the Gumbel-max question

> We additionally evaluated the Gumbel-max baselines without top-k
> truncation, using the same 100 prompts and a serial single-GPU protocol.
> Removing top-k changes MSE throughput from 20.15 to 20.31 tok/s and MWS
> throughput from 19.36 to 19.31 tok/s. Their measured Gumbel watermark step
> remains stable at 9.17--9.72 ms per decoding block, while incremental peak
> memory is unchanged. Thus, as the reviewer anticipated, MSE/MWS do not
> require top-k truncation to control their watermarking latency.
