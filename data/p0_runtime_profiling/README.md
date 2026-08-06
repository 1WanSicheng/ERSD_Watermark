# P0 runtime profiling

This directory contains the reviewer-response runtime profiling experiment.
It is intentionally separate from the paper's main result files.

The first-pass benchmark answers one question: where does MPFR wall-clock
time go? It reports synchronized GPU wall time for:

- target-model forwards;
- draft-model forwards;
- MPFR arrival sampling (`uniform -> exponential -> cumulative arrivals ->
  top-B`);
- keyed uniform generation (a subset of MPFR arrival sampling);
- remaining Python/tree/cache/bookkeeping time;
- peak allocated GPU memory.

The benchmark uses the paper's Qwen2.5-7B-Instruct /
Qwen2.5-0.5B-Instruct pair, `top_k=50`, `L=4`, and sweeps `B`.

Run:

```bash
python profile_mpfr_runtime.py \
  --samples 5 \
  --warmup 1 \
  --max-new-tokens 64 \
  --draft-counts 2 4 8 \
  --output p0_first_pass.json
```

Single-draft PFR/PFR-NOWM:

```bash
python profile_single_runtime.py \
  --samples 5 \
  --warmup 1 \
  --max-new-tokens 64 \
  --output p0_single_first_pass.json
```

The diagnostic results and their rebuttal interpretation are summarized in
`FIRST_PASS_SUMMARY.md`, `SINGLE_FIRST_PASS_SUMMARY.md`, and
`SINGLE_FOURWAY_SUMMARY.md`. The Table 1/2 lookahead diagnosis is in
`SINGLE_L_SWEEP_COMPONENTS.md`, and the PFR-versus-VSPS block-count
decomposition is in `SINGLE_TR_GAP_DECOMPOSITION.md`.

The consolidated claim, proposed tables, rebuttal wording, and formal
larger-run protocol are recorded in `REBUTTAL_SINGLE_DRAFT_RUNTIME.md`.

For the single-draft result, use `p0_single_topk50_corrected.json`. The older
`p0_single_first_pass.json` is an invalid top-k comparison retained only for
provenance: the single-draft decoder requires a callable `logits_warper`, not
just the scalar `top_k` field.

Four-way single-draft comparison:

```bash
python profile_single_fourway.py \
  --methods vsps mse pfr_nowm pfr \
  --samples 5 \
  --warmup 1 \
  --max-new-tokens 64 \
  --output p0_single_fourway_first_pass.json
```

The synchronized phase timers are deliberately opt-in and introduce some
measurement overhead. The script therefore also reports an uninstrumented
MPFR pass for end-to-end token rate. Use the instrumented pass only for the
time breakdown.
