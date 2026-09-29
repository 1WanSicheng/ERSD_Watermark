# Supplementary code: Watermarkable Multi-Draft Speculative Sampling via Poisson Processes

Anonymous code release accompanying the submission. It contains the
implementations of PFR (single-draft) and MPFR (multi-draft), all baselines,
the experiment runners and configurations, the per-seed result summaries
behind the multi-draft tables, and the curve data behind Figure 2.

## Layout

| Path | Contents |
|---|---|
| `MPFR_spec/mpfr_batched_torchgen_cached.py` | MPFR multi-draft decoder (keyed mapped-Poisson sources, context tree, batched target verification) |
| `MPFR_spec/mpfr_direct_optimized.py` | Direct finite-support MPFR primitives (Algorithm 6) |
| `accuwm/pfr.py`, `accuwm/pfr_no_watermark.py` | Single-draft PFR and its unkeyed ablation PFR-NoWM |
| `accuwm/invariant_multi.py` | GLS baseline (drafter-invariant multi-draft strategy of Rowan et al., 2025) |
| `accuwm/basic_watermark.py`, `accuwm/mc*.py`, `unbiased_watermark/` | Basic-UWM, MWS / MSE / MSE-Pseudo baselines, DeltaGumbel code, watermark scores (U / Li / PL) |
| `experiments/run_single_draft.py`, `experiments/run_multi_draft.py` | JSON-config driven experiment runners (all metrics: AATPS, token rate, ANLPPT, TPR@1%FPR, LPPL, ROUGE-L) |
| `experiments/configs/` | Configurations for all reported experiments (see below) |
| `data/multidraft_3seed/` | Per-seed summaries of the 48 multi-draft shards and the aggregated tables |
| `data/0506 single_draft_n1000_efpr_4cell/`, `data/0506 ablation_n500_qwen_cnn/` | Pre-aggregated TPR-vs-T curves for Figure 2 |

## Environment

Python 3.12, PyTorch 2.11 (CUDA 13.x build), Transformers 4.46.3, Datasets,
SciPy; see `requirements.txt`. Models are loaded from the Hugging Face Hub in
float16. Llama-3.1-8B-Instruct / Llama-3.2-1B-Instruct are gated; any
byte-identical mirror can be substituted via `target_model` / `draft_model`
in the configs. The multi-draft experiments (Table 1, Tables 6-9) were run on
NVIDIA A100-SXM4-40GB GPUs, one GPU per shard.

## Reproducing the paper

**Multi-draft efficiency and watermark strength (Table 1, Appendix Tables 6-9).**
48 shards = {Qwen2.5-7B/0.5B, Llama-3.1-8B/3.2-1B} x {CNN/DailyMail, ELI5}
x B in {2,4,6,8} x 3 seeds, L=4, 1000 prompts each. Seeds are realized by
`seed_offset` in {0, 10000, 20000} (per-prompt seed = prompt index + 7 +
offset). Each shard runs MPFR and GLS sequentially in the same process on the
same GPU.

```bash
PYTHON=python GPUS="0,1,2,3" bash scripts/run_multidraft_3seed.sh
python scripts/aggregate_3seed_tables.py --pair qwen  --latex --results-dir outputs/multidraft_3seed
python scripts/aggregate_3seed_tables.py --pair llama --latex --results-dir outputs/multidraft_3seed
```

The released summaries reproduce the tables without re-running generation:

```bash
python scripts/aggregate_3seed_tables.py --pair qwen  --results-dir data/multidraft_3seed
python scripts/aggregate_3seed_tables.py --pair llama --results-dir data/multidraft_3seed
```

(`data/multidraft_3seed/tables_*_3seed.txt` are the outputs of these commands.)

**Single-draft results (Figure 1, Appendix Tables 2-5).**
`python -m experiments.run_single_draft --config experiments/configs/single_{qwen,vicuna}_{cnn,eli5}_n1000.json`

**Detection curves and drafter substitution (Figure 2, Appendix D.5, Tables 10-11).**
Generation configs are in `experiments/configs/full_*_n1000.json` and
`experiments/configs/ablation_n1000/`; curves are aggregated by
`experiments/aggregate_tpr_curves.py` / `experiments/aggregate_drafter_curves.py`
and plotted by `experiments/plot_fig_2x4_detection_invariance.py`.

**Smoke test.**
`python -m experiments.run_multi_draft --config experiments/configs/smoke_multi_n5_qwen_cnn.json`

## License

MIT (see `License.txt`).
