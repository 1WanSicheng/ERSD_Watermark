# Task 12: model-free PF detectors

This task evaluates the three detector families in Section 3 of
`ICLR_27_Speculative_PF_water.pdf` without changing the frozen PF generator:

- Original PF: `sum -log(V_t)`;
- Li-style binary likelihood ratio with a fixed reference `rho`;
- centered truncated power-law with a fixed `epsilon`.

All reported p-values, ANLPPT values, and TPR values use the null
`V_t ~ Uniform(0, 1)`. Li and power-law scores are Monte-Carlo calibrated for
the exact scored length. The detector never evaluates target or draft model
logits; the tokenizer is loaded only to reconstruct prompt token IDs and exact
context labels from saved generations.

The initial sweep uses:

- `rho in {0.05, 0.1, 0.2, 0.5}`;
- `epsilon in {0.01, 0.02, 0.05, 0.1}`;
- TPR at FPR `1%` and `0.1%`;
- mean and median ANLPPT over prompts.

Generation AATPS, token rate, output tokens, and quality are unchanged.

## Reproduction

```bash
python PF_sd/task12_model_free_pf_detectors/evaluate_saved_results.py \
  PF_sd/frozen_pf_decoder/results/vicuna_b2_n100_kvview.json \
  PF_sd/task12_model_free_pf_detectors/results/detectors_b2_seed7.json \
  --device cuda:0 --n-null 200000
```

The evaluator deliberately loads a tokenizer and the prompt dataset, but no
language model. See `RESULTS.md` for the initial two-seed results.
