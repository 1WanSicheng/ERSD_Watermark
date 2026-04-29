# Experiments

Two JSON-config-driven experiment runners.  Both share helpers in
`_shared.py` (model/dataset loading, decoder dispatch, metric computation).

## Single-draft comparison

Decoders (B=1):

- `mc` — basic speculative decoding, no watermark
- `basic_uwm` — autoregressive UWM (DeltaGumbel reweight; AATPS is
  trivially 1, included as the watermark-strength upper-bound reference)
- `mc_uwm_speed` — multi-candidate UWM with draft-side reweight only
- `mc_uwm_strength` — multi-candidate UWM with target-side reweight
- `pfr` — single-draft PFR (B=1 cached pipeline)
- `pfr_no_watermark` — fresh-noise PFR (H0 control)

Metrics: AATPS, token_rate, ANLPPT-{U, Li, PL}; KL/WS-ratio reported only
for `pfr` (delta-conditional empirical estimator under Definition 3.1 of
arXiv:2602.01428, equivalent to `pfr_watermark_strength_from_sequence`).

`basic_uwm` does not depend on the lookahead; it is run once per prompt
and duplicated into every requested `lookaheads` row in the output table.

```bash
python -m experiments.run_single_draft \
    --config experiments/configs/single_draft_default.json
```

## Multi-draft comparison

Decoders (B>=1):

- `ms_pfr_cached` — `MPFR_spec.multi_draft_pfr_batched_cached`
- `mpfr_torchgen_cached` — `MPFR_spec.mpfr_batched_torchgen_cached`
- `invariant_multi` — `SpeculativeDecoding.strategy.InvariantMultiDraftStrategy`
- `strong_multi` — `SpeculativeDecoding.strategy.StrongMultiDraftStrategy`

Metrics: AATPS, token_rate.  ANLPPT-{U, Li, PL} reported only for the
PFR-family decoders listed in `metrics.anlppt.applies_to`.  The multi-draft
config sweeps both `lookaheads` and `num_drafts`; `lookaheads=[4]` is the
default.

```bash
python -m experiments.run_multi_draft \
    --config experiments/configs/multi_draft_default.json
```

## Config schema

Both runners read JSON with this shape:

```jsonc
{
  "experiment": "single_draft" | "multi_draft",
  "samples": 100,
  "dataset": "cnn_dailymail" | "gsm8k",
  "max_new_tokens": 128,
  "lookaheads": [2, 4, 6, 8],     // single_draft sweep
  "num_drafts": [1, 2, 4, 8],     // multi_draft sweep (multi_draft only)
  "private_key": "1234",
  "process_logits": {"top_k": 50, "top_p": 1.0, "temperature": 1.0},
  "decoders": [...],              // decoder names
  "metrics": {
    "aatps": true,
    "token_rate": true,
    "anlppt": {
      "variants": ["U", "Li", "PL"],
      "li_delta": 0.5,
      "pl_eps": 0.1,
      // "applies_to": [...]        // multi_draft only: limit ANLPPT scoring
    },
    // single_draft only:
    "kl_ratio_pfr_only": true
  },
  "output": "outputs/<name>.json"
}
```

The runner writes a JSON containing the original config, the per-(decoder,
sweep-cell) summary, and every per-prompt row.  Override the output path
via `--output`.

## Adding a decoder

`_shared.py` exposes two registries:

- `SINGLE_DRAFT_DECODERS: Dict[str, Callable]`
- `MULTI_DRAFT_DECODERS: Dict[str, Callable]`

Each value is a callable with signature

```python
fn(target, draft, input_ids, *, lookahead, max_length, seed, base_key,
   plk, [num_drafts], **_) -> (out_ids, block_lens, private_key,
                               wm_kind, source_labels, masked_flags)
```

`wm_kind` is one of `"PFR"`, `"DeltaGumbel"`, or `"none"` and selects the
detector convention used for ANLPPT.  PFR-family decoders must also return
the per-token `source_labels` so the detector can recover uniforms; for
generators that do not emit labels in their meta (e.g. the multi-draft
cached variants), `_shared._rebuild_labels_for_prefix_scheme` reconstructs
them from the realized prefix using the appropriate scheme.

Add a new decoder by registering it in one of the two dictionaries.
