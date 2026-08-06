# Vicuna Basic-UWM token-rate recheck

## What the submitted paper reports

The submitted PDF does not show one uniform relationship between PFR and
Basic-UWM on Vicuna. The result is dataset dependent.

| Dataset | L | Basic-UWM TR | PFR TR | PFR vs Basic-UWM |
|---|---:|---:|---:|---:|
| CNN/DailyMail | 1 | 37.058 | 38.256 | +3.2% |
| CNN/DailyMail | 2 | 37.058 | 42.639 | +15.1% |
| CNN/DailyMail | 3 | 37.058 | 41.813 | +12.8% |
| CNN/DailyMail | 4 | 37.058 | 41.800 | +12.8% |
| ELI5 | 1 | 38.536 | 35.231 | -8.6% |
| ELI5 | 2 | 38.536 | 34.425 | -10.7% |
| ELI5 | 3 | 38.536 | 35.694 | -7.4% |
| ELI5 | 4 | 38.536 | 32.316 | -16.1% |

Thus, the concern is correct for the Vicuna/ELI5 cells, but not for the
Vicuna/CNN-DailyMail cells.

## New matched 100-prompt check

All three methods below use exactly the same current runner and configuration:
Vicuna-7B-v1.5/Vicuna-68M, CNN/DailyMail paper prompt, FP16, A100-40GB,
top-k 50, temperature 1, maximum 128 output tokens, and the same 100-prompt
selection. Timing is full end-to-end generation timing.

| Method | AATPS | TR (tok/s) | TR vs Basic-UWM |
|---|---:|---:|---:|
| Basic AR | 1.000 | 40.05 | +2.9% |
| Basic-UWM | 1.000 | 38.93 | reference |
| VSPS, L=4 | 2.205 | 55.66 | +43.0% |
| PFR, L=4 | 2.142 | 55.11 | +41.5% |

The current PFR runtime decomposition is 66.0% target forward, 20.4% four
draft forwards, 2.2% PFR arrival sampling, and 7.2% remaining decoding work.
This confirms an actual end-to-end speedup on this model/dataset pair under the
current implementation and unified timing convention.

Under the submitted execution path, CNN/DailyMail L=4 reported nearly equal
AATPS (2.194 VSPS versus 2.177 PFR) but a 22.5% PFR TR deficit (53.94 versus
41.80 tok/s). Under the current matched run, the AATPS gap is 2.9% and the TR
gap is only 1.0%. Their measured wall time per verification block is also close
(39.61 ms VSPS versus 38.86 ms PFR). Therefore the large submitted VSPS/PFR TR
gap is not caused by the coupling or watermark acceptance rate.

Absolute TR should not be substituted directly into the submitted H100 table:
the hardware and current execution path differ. The robust conclusion from the
new matched run is the within-run comparison: PFR is faster than both Basic-UWM
and Basic AR on Vicuna/CNN-DailyMail. The smaller matched ELI5 check is reported
separately below.

## ELI5 directional check

We also ran a smaller matched 20-prompt ELI5 diagnostic under the same current
runner:

| Method | AATPS | TR (tok/s) | TR vs Basic-UWM |
|---|---:|---:|---:|
| Basic-UWM | 1.000 | 38.62 | reference |
| PFR, L=4 | 1.757 | 44.79 | +16.0% |

This small check reverses the submitted ELI5 result, but it should be treated
as directional until repeated on 100 prompts. It indicates that the current
PFR execution path can convert even the lower ELI5 AATPS into an end-to-end
speedup; it does not erase the workload dependence of AATPS itself.

## Raw records

- `basic_baselines_n100.json`: Basic AR and Basic-UWM.
- `pfr_paperprompt_n100.json`: PFR at L=4.
- `vsps_paperprompt_n100.json`: VSPS at L=4.
- `eli5_basic_pfr_n20.json`: small matched ELI5 diagnostic.
