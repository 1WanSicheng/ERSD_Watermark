# Configured Experiments

This directory now has two JSON-driven experiment entry points:

```bash
python experiments/run_configured_strength.py --config experiments/config_strength_example.json
python experiments/run_configured_quality.py --config experiments/config_quality_example.json
```

`run_configured_strength.py` reports:

- `AATPS`
- `TR`
- `ANLPPT_U`

`run_configured_quality.py` reports:

- `AATPS`
- `TR`
- `log_perplexity`
- `rouge1_f1`
- `rouge2_f1`
- `rougeL_f1`
- `bleu`

## JSON Fields

Common top-level fields:

```json
{
  "dataset": "gsm8k",
  "samples": 20,
  "lookahead": 4,
  "max_length": 128,
  "seed": 1,
  "private_key": "1234",
  "target_model": "model/Qwen2.5-7B-Instruct",
  "draft_model": "model/Qwen2.5-0.5B-Instruct",
  "target_device": "cuda:0",
  "draft_device": "cuda:0",
  "progress_every": 5,
  "output": "outputs/example.json",
  "rows_output": "outputs/example.rows.jsonl",
  "algorithms": []
}
```

Supported algorithm specs:

```json
{"name": "pfr_nowatermark", "type": "pfr_nowatermark"}
{"name": "pfr", "type": "pfr", "labeler_mode": "prefix"}
{"name": "mspfr_B4", "type": "mspfr", "num_drafts": 4}
{"name": "uwm_strength", "type": "uwm_strength", "reweight": "deltagumbel"}
{"name": "uwm_speed", "type": "uwm_speed", "reweight": "deltagumbel"}
```

Aliases are accepted for compatibility:

- `pfr_no_watermark` -> `pfr_nowatermark`
- `multi_draft_pfr` -> `mspfr`
- `mc_uwm_strength` -> `uwm_strength`
- `mc_uwm_speed` -> `uwm_speed`

## Notes

- `mspfr` currently uses prefix labels for its PFR detector.
- `pfr_nowatermark` has no embedded watermark metadata; strength runs score it with the same prefix PFR U detector as a null baseline.
- UWM methods use `DeltaGumbel_U` by default.
