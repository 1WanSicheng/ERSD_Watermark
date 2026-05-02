# KL watermark-strength analysis

Analyzed 2 JSON file(s).

Files:
- `qwen_summarization450_len200_K2_pfr.json`
- `qwen_summarization450_len200_K3_pfr.json`

## Summary table

| K | method | samples | AATPS | token rate | KL mean | KL ratio |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `basic_uwm` | 450 | 1.000 | 3.355 | 0.388 | 0.983 |
| 2 | `mc_uwm_speed` | 450 | 2.021 | 14.125 | 0.333 | 0.429 |
| 2 | `mc_uwm_strength` | 450 | 1.974 | 13.830 | 0.768 | 0.997 |
| 2 | `pfr_uwm` | 450 | 1.986 | 12.230 | 0.721 | 0.970 |
| 3 | `basic_uwm` | 450 | 1.000 | 3.239 | 0.393 | 0.986 |
| 3 | `mc_uwm_speed` | 450 | 2.275 | 12.849 | 0.334 | 0.431 |
| 3 | `mc_uwm_strength` | 450 | 2.203 | 12.405 | 0.777 | 1.002 |
| 3 | `pfr_uwm` | 450 | 2.219 | 10.935 | 0.720 | 0.971 |

## Main takeaways

- For K=2, strongest by KL ratio: `mc_uwm_strength` with KL ratio 0.997.
- For K=2, best AATPS: `mc_uwm_speed` with AATPS 2.021.
- For K=2, `pfr_uwm` gives 1.99x AATPS of `basic_uwm` and retains 98.6% of its KL ratio.
- For K=3, strongest by KL ratio: `mc_uwm_strength` with KL ratio 1.002.
- For K=3, best AATPS: `mc_uwm_speed` with AATPS 2.275.
- For K=3, `pfr_uwm` gives 2.22x AATPS of `basic_uwm` and retains 98.4% of its KL ratio.

Interpretation: higher KL mean/ratio means a stronger white-box watermark signal, while higher AATPS/token rate means better sampling efficiency.