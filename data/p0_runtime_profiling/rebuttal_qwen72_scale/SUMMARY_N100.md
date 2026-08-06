# Qwen2.5-72B rebuttal experiment (100 prompts)

## Setup

- Target/drafter: Qwen2.5-72B-Instruct / Qwen2.5-0.5B-Instruct
- Hardware: 8 x NVIDIA A100 40GB, identical balanced sharding for all methods
- Dataset: first 100 filtered CNN/DailyMail test prompts after seed-42 shuffle
- Decoding: 128 maximum new tokens, lookahead \(L=4\), top-\(k=50\),
  temperature 1.0
- Runtime values below come from the two-pass component profiler. Watermark
  and quality values come from the paper-style one-pass runner, whose timing
  window ends before detector and quality metrics are computed.

## Runtime and memory

| Regime | Method | \(B\) | TR | AATPS | Max incremental peak GPU memory |
|---|---|---:|---:|---:|---:|
| Single | VSPS | 1 | 10.025 | 2.380 | 141.2 MiB |
| Single | PFR-NOWM | 1 | 9.838 | 2.339 | 141.6 MiB |
| Single | PFR | 1 | 9.771 | 2.321 | 139.3 MiB |
| Multi | MPFR | 4 | 10.218 | 2.988 | 563.3 MiB |
| Multi | INVARIANT | 4 | 10.328 | 3.057 | 870.2 MiB |
| Multi | MPFR | 8 | 9.922 | 3.274 | 1103.8 MiB |
| Multi | INVARIANT | 8 | 9.633 | 3.307 | 1735.7 MiB |

## Watermark and quality

TPR is measured at 1% FPR using up to the first 128 generated tokens. For the
no-watermark methods, the corresponding detector is applied to provide the
null reference; these methods do not embed a watermark.

| Method | \(B\) | ANLPPT-U | ANLPPT-Li | ANLPPT-PL | TPR-U | TPR-Li | TPR-PL | LPPL | ROUGE-L | KL/WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VSPS | 1 | 0.0031 | 0.0027 | 0.0031 | 0.01 | 0.01 | 0.02 | 0.2429 | 0.2334 | -- |
| PFR-NOWM | 1 | 0.0024 | 0.0023 | 0.0023 | 0.00 | 0.00 | 0.00 | 0.2364 | 0.2435 | -- |
| PFR | 1 | 0.0178 | 0.0121 | 0.0190 | 0.12 | 0.03 | 0.15 | 0.2357 | 0.2354 | 0.9664 |
| MPFR | 4 | 0.0156 | 0.0134 | 0.0154 | 0.07 | 0.04 | 0.07 | 0.2428 | 0.2378 | 1.0126 |
| INVARIANT | 4 | 0.0018 | 0.0019 | 0.0016 | 0.00 | 0.00 | 0.00 | 0.2645 | 0.2354 | -- |
| MPFR | 8 | 0.0160 | 0.0125 | 0.0164 | 0.08 | 0.01 | 0.10 | 0.2465 | 0.2376 | 1.0317 |
| INVARIANT | 8 | 0.0019 | 0.0024 | 0.0016 | 0.00 | 0.00 | 0.00 | 0.2609 | 0.2353 | -- |

## Immediate observations

- Adding the key to single-draft PFR changes TR by only \(-0.7\%\) relative
  to PFR-NOWM (9.771 versus 9.838 token/s), with essentially identical
  incremental peak memory.
- MPFR is within \(-1.1\%\) of INVARIANT in TR at \(B=4\), and is \(3.0\%\)
  faster at \(B=8\). Their AATPS values remain close.
- MPFR uses 35.3% less incremental peak memory at \(B=4\) and 36.4% less at
  \(B=8\) than the current INVARIANT implementation.
- PFR/MPFR show a clear watermark signal over their matched no-watermark
  references, while LPPL and ROUGE-L remain comparable.
- KL/WS is close to 1 for PFR and MPFR, consistent with the intended
  target-side watermark-strength behavior.
