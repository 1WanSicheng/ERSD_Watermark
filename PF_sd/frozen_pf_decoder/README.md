# Frozen optimized Latin-PF decoder

This directory is the frozen implementation of watermarkable independent
Latin Permute-and-Flip speculative decoding. The algorithm, keyed randomness,
detector labels, and fixed-key outputs are frozen as of 2026-08-06.

## Layout

- `decoder.py`: optimized flat-trajectory decoder with prefix compaction;
- `core/max_order_pf.py`: Algorithm-2 and Latin-PF primitives;
- `benchmark.py`: matched PF/INVARIANT benchmark runner;
- `test_decoder.py`, `core/test_core.py`, `core/test_latin_pf.py`: regression tests;
- `configs/`: final benchmark configurations;
- `results/`: final two-seed runtime results and compressed large-scale sweep;
- `RESULTS.md`: final runtime and exactness report;
- `LARGE_SCALE_RESULTS.md`: 1,000-prompts-per-seed B/L sweep.

## Frozen guarantees

- optimized and reference PF paths preserve identical fixed-key outputs;
- AATPS, block counts, PF-ANLPPT, PF-Li, and PF-PL are unchanged;
- 21 Algorithm-2/Latin-PF/tree-free regression tests pass;
- final verification covers B in {2,4}, L=4, 100 prompts and two seeds;
- the large-scale AATPS sweep covers B,L in {2,4,6}, 1,000 prompts and two seeds.

Run a benchmark from the repository root:

```bash
python PF_sd/frozen_pf_decoder/benchmark.py \
  --config PF_sd/frozen_pf_decoder/configs/vicuna_b4.json \
  --output PF_sd/frozen_pf_decoder/results/new_run.json
```

Intermediate development tasks were moved to
`data/pf_sd_intermediate_archive_20260806/` and can be restored if needed.
