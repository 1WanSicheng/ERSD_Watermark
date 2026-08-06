# Inevitable model-pair validation

## Purpose

This small diagnostic tests whether the absence of an end-to-end speedup on the
Qwen2.5-7B/0.5B pair is primarily caused by the model-pair latency ratio. It
uses the target/drafter pair from *An Inevitable Trade-off between Watermark
Strength and Speculative Sampling Efficiency*: Llama-7B and Llama-68M.

## Setup

- Target: `huggyllama/llama-7b`.
- Drafter: `JackFram/llama-68m`.
- Precision: FP16; one NVIDIA A100-SXM4-40GB.
- Data: 20 CNN/DailyMail prompts; maximum 128 generated tokens; one common seed.
- Top-k: disabled, matching the Inevitable setup.
- TR: total generated tokens divided by full generation wall time, including
  prefill. This is a stricter end-to-end measure than Inevitable's steady-state
  PTT, which starts timing after the first output block.
- AATPS in this diagnostic is the measured mean number of output tokens per
  target-verification block (`tokens / blocks`).

This is a directional diagnostic, not the final rebuttal-scale experiment.

## End-to-end results

| Lookahead | Method | AATPS | TR (tok/s) | TR vs Basic-UWM |
|---:|---|---:|---:|---:|
| - | Basic AR | 1.000 | 39.84 | +3.6% |
| - | Basic-UWM | 1.000 | 38.46 | reference |
| 1 | VSPS | 1.541 | 50.72 | +31.9% |
| 1 | PFR-NOWM | 1.459 | 48.51 | +26.1% |
| 1 | PFR | 1.441 | 48.05 | +24.9% |
| 2 | VSPS | 1.734 | 52.91 | +37.6% |
| 2 | PFR-NOWM | 1.626 | 50.10 | +30.3% |
| 2 | PFR | 1.647 | 50.82 | +32.1% |
| 3 | VSPS | 1.965 | 55.90 | +45.3% |
| 3 | PFR-NOWM | 1.784 | 51.16 | +33.0% |
| 3 | PFR | 1.749 | 50.16 | +30.4% |
| 4 | VSPS | 2.105 | 56.08 | +45.8% |
| 4 | PFR-NOWM | 1.755 | 47.07 | +22.4% |
| 4 | PFR | 1.786 | 48.01 | +24.8% |

Additional watermark-SD baselines were run serially under the same setup:

| Lookahead | Method | AATPS | TR (tok/s) | TR vs Basic-UWM |
|---:|---|---:|---:|---:|
| 1 | MSE | 1.507 | 47.49 | +23.5% |
| 1 | MWS | 1.433 | 44.14 | +14.8% |
| 2 | MSE | 1.796 | 45.25 | +17.6% |
| 2 | MWS | 1.669 | 44.78 | +16.4% |
| 3 | MSE | 2.032 | 49.33 | +28.3% |
| 3 | MWS | 1.741 | 37.92 | -1.4% |
| 4 | MSE | 2.021 | 47.89 | +24.5% |
| 4 | MWS | 1.785 | 36.54 | -5.0% |

MWS at the longer lookaheads is the exception: its method-specific watermark
processing and lower AATPS erase the model-pair benefit. This does not weaken
the PFR diagnosis; it shows why both the model-pair ratio and method-specific
overhead must be measured.

### Why PFR remains slower than VSPS at L=4

| Method | AATPS | TR (tok/s) | Measured time per block |
|---|---:|---:|---:|
| VSPS | 2.105 | 56.08 | 37.53 ms |
| PFR-NOWM | 1.755 | 47.07 | 37.29 ms |
| PFR | 1.786 | 48.01 | 37.20 ms |

The three methods have essentially identical block execution time; PFR is not
slower per block. The L=4 TR gap is caused by its lower AATPS relative to the
maximal coupling used by VSPS. Eliminating PFR arrival sampling alone (2.1% of
runtime) could only provide a small improvement and cannot close this AATPS
gap without changing the coupling itself. At L=2 the AATPS gap is smaller, and
PFR already reaches 50.82 versus 52.91 tok/s for VSPS.

## Runtime decomposition

The component shares below come from a separate five-prompt instrumented pass.
They explain the model-pair effect; they are not added to the end-to-end TR
measure.

| Lookahead | Method | Target forward | Draft forwards | PFR sampling | Other decoding |
|---:|---|---:|---:|---:|---:|
| 1 | PFR | 86.4% | 7.1% | 1.2% | 5.3% |
| 2 | PFR | 79.6% | 12.5% | 1.6% | 6.3% |
| 3 | PFR | 73.9% | 17.1% | 1.9% | 7.0% |
| 4 | PFR | 69.1% | 21.2% | 2.1% | 7.4% |

For comparison, in the earlier Qwen2.5-7B/0.5B L=4 audit, four draft
forwards alone took 79.69 ms of a 110.02 ms PFR block (72.4%). On the
Llama-7B/68M pair they occupy only 21.2% of instrumented runtime. Thus the
dominant change is the drafter/target latency ratio, not removal of watermark
work or a different TR formula.

## Conclusions

1. The hypothesis is confirmed directionally: on the exact model pair used by
   Inevitable, every tested PFR configuration is faster than both Basic-UWM and
   Basic AR under full end-to-end timing.
2. PFR and PFR-NOWM remain close. The watermark does not explain the Qwen
   speed gap; PFR arrival sampling is only 1.2--2.1% here.
3. AATPS alone cannot guarantee end-to-end acceleration. The drafter must be
   sufficiently cheap relative to the target. Llama-68M provides that regime;
   Qwen2.5-0.5B does not on the tested hardware and implementation.
4. L=4 is not optimal for PFR on this sample. Its AATPS rises only slightly
   from 1.749 at L=3 to 1.786 at L=4, which is insufficient to offset the
   fourth draft forward; consequently TR drops. The best observed PFR point is
   L=2 (50.82 tok/s), while L=3 is nearly identical in TR (50.16 tok/s) and
   has higher AATPS.

The appropriate interpretation is not that the Qwen pair was invalid: it is a
published list-level-coupling configuration and remains suitable for testing
draft invariance. However, that prior work did not establish speedup over a
non-SD watermark baseline. The new diagnostic shows that absolute end-to-end
speedup is model-pair dependent and that a substantially smaller drafter restores
the expected SD advantage.

## Artifacts

- `results/baselines.json`: Basic AR and Basic-UWM raw records.
- `results/vsps_pfr.json`: VSPS, PFR-NOWM, and PFR raw records.
- `results/mse_mws.json`: MSE and MWS raw records.
- `run_inevitable_llama7b_n20.sh`: reproducible server command.
