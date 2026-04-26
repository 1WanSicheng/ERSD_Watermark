# Unified Metric Convention

## Primary Metrics

The repository now treats the following two metrics as the primary comparison
axes for watermarkable speculative decoding experiments:

- `AATPS`
- `ANLPPT-U`

## Definitions

- `AATPS`
  - Average accepted tokens per decoding step.
  - This is the primary acceleration metric.

- `ANLPPT-U`
  - Negative log p-value per token computed from a U-score style detector.
  - This is the primary unified watermark-strength metric.

## Why Only These Two

Cross-method comparisons become ambiguous when each method family uses a
different matched detector.

- For PFR, `Aaronson-Gamma` is a matched detector.
- For UWM/MC-UWM, `RobustLLR` is a matched detector.

Those are useful method-specific diagnostics, but they are not directly
comparable across method families.

`ANLPPT-U` is retained as the common comparison metric because both method
families expose a tokenwise uniform-style statistic and support a U-score based
test.

## Method-Specific Detector Mapping

- `pfr_prefix` / `pfr_cc`
  - primary watermark metric: `PFR_Aaronson_U`

- `basic_uwm` / `mc_uwm_speed` / `mc_uwm_strength`
  - primary watermark metric: `DeltaGumbel_U`

- `ersd_wm`
  - primary watermark metric: `ERSD_Aaronson_U`

## Practical Rule

For main tables, plots, and trade-off discussions:

- keep `AATPS`
- keep `ANLPPT-U`
- drop `ANLPPT-Aux`
- treat Aaronson-Gamma / RobustLLR as optional secondary diagnostics only
