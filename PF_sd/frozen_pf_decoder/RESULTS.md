# Frozen optimized Latin-PF backend

## Question

Is the explicit Python prefix tree the main reason Latin-PF token rate trails
the INVARIANT implementation, and can it be removed without changing the
coupling, AATPS, or watermark?

## Backend

`latin_pf_counter_tree_free` keeps B logical trajectories as a flat token
matrix. Verification uses an alive-row mask. A stable temporary compaction is
performed at each depth so the draft and target models evaluate every unique
prefix once. This preserves the model batch shapes and first-occurrence order
of the reference backend, but removes the persistent prefix tree, child sets,
and tree traversal.

This compaction is important. A preliminary fully dense list version evaluated
duplicate target paths, increased the B=4 target positions from about 12.4 to
19.5 per block, and could produce FP16 boundary divergence because the model
batch shape changed. That version was discarded and is not included below.

## Setup

- Target/draft: Vicuna-7B-v1.5 / Vicuna-68M
- Dataset: CNN/DailyMail test split
- 100 prompts, seed 7, 128 generated tokens per prompt
- top-k 50, temperature 1, lookahead L=4
- B in {2,4}
- Full generation warmup and rotating method order
- B=2 and B=4 ran on separate, otherwise idle GPUs
- Compared methods: existing fused tree backend, tree-free backend, INVARIANT

## Exactness audit

| B | Prompt paths identical | Block counts identical | AATPS identical | PF watermark metrics identical |
|---:|---:|---:|---:|---:|
| 2 | 100/100 | 100/100 | 100/100 | 100/100 |
| 4 | 100/100 | 100/100 | 100/100 | 100/100 |

“Prompt paths identical” means the complete fixed-key output token sequence is
equal, not merely that aggregate averages are close. The checked watermark
metrics are PF-ANLPPT, PF-Li, and PF-PL.

## Runtime and memory

| B | Method | AATPS | Token rate (tok/s) | Peak allocated GPU memory (GiB) |
|---:|---|---:|---:|---:|
| 2 | INVARIANT | 2.456 | 62.309 | 13.306 |
| 2 | Latin-PF tree | 2.446 | 60.670 | 14.366 |
| 2 | Latin-PF tree-free | 2.446 | 60.934 | 14.636 |
| 4 | INVARIANT | 2.635 | 65.030 | 13.929 |
| 4 | Latin-PF tree | 2.685 | 63.669 | 16.116 |
| 4 | Latin-PF tree-free | 2.685 | 63.684 | 16.379 |

Relative to the tree backend, tree-free changes aggregate token rate by
+0.43% at B=2 and +0.02% at B=4. The median paired latency changes are -0.08%
and +0.001%, respectively. These are effectively ties, not a meaningful or
stable speedup. Tree-free uses approximately 0.27 and 0.26 GiB more peak
allocated memory at B=2 and B=4.

## Conclusion

The explicit tree is not the main token-rate bottleneck. It can be removed as
a software representation while preserving the exact coupling and watermark,
but efficient execution still needs prefix compaction to avoid duplicate model
work. Once both backends perform the same unique-prefix computation, their TR
is essentially identical.

The remaining cost is therefore elsewhere: L autoregressively dependent draft
forwards per block, one target verification forward, and keyed selection plus
host/device dependencies. Removing the tree does not remove these operations.
Future TR work should target those execution dependencies rather than the
prefix container itself.

## Exact engineering optimizations after the tree diagnosis

The representation experiment above showed that deleting the tree alone was
not sufficient. Direct comparison with INVARIANT exposed two avoidable costs
in the PF implementation:

1. INVARIANT pre-samples its shared randomness once on GPU. PF repeatedly
   serialized the same full-prefix label and recomputed its SHA-256 seed for
   duplicate trajectories and again during target verification. The optimized
   backend computes each exact context label/seed once and reuses it across
   draft and target operations.
2. INVARIANT retains a sliced KV-cache view after verification. PF selected
   the same row but additionally called `.contiguous()` on every target and
   draft KV layer, copying the complete retained prefix at every block. The
   optimized path retains the view and materializes only when a later batched
   operation actually requires it.

The model capability check used by the PF forward wrapper is also cached by
model class. None of these changes modifies the PRF address, random value,
Latin coupling, accepted path, detector label, or watermark score.

### Final two-seed results

The final benchmark uses 100 prompts for each of seeds 7 and 107. All other
settings match the table above.

This 200-prompt run is the matched runtime validation, not the primary AATPS
estimate. In the earlier 1,000-prompts-per-seed sweep at B=4,L=4, Latin PF
achieved AATPS 2.6720 versus INVARIANT 2.6471, a +0.94% difference with 95%
CI `[+0.29%, +1.58%]`. The smaller runtime subset gives 2.6712 versus 2.6737:
PF itself is unchanged, while the INVARIANT estimate is unusually high on the
first 100 prompts of seed 107. The saved first-100 token and block counts match
the original 1,000-prompt files on all 400 method/seed/prompt comparisons.

| B | Method | Pooled AATPS | Pooled TR (tok/s) | Time/block (ms) | Peak memory (GiB) |
|---:|---|---:|---:|---:|---:|
| 2 | INVARIANT | 2.435 | 62.355 | 39.049 | 13.31 |
| 2 | Fused tree PF (KV-view) | 2.453 | 62.927 | 38.980 | 14.43 |
| 2 | Optimized tree-free PF | 2.453 | **63.763** | **38.468** | 14.41 |
| 4 | INVARIANT | 2.674 | 65.153 | 41.037 | 13.93 |
| 4 | Fused tree PF (KV-view) | 2.671 | 63.835 | 41.846 | 16.13 |
| 4 | Optimized tree-free PF | 2.671 | **65.242** | **40.943** | 16.11 |

Relative to the matched fused tree backend, the optimized backend improves pooled TR
by 1.33% at B=2 and 2.20% at B=4. Prompt-stratified bootstrap 95% intervals
are `[+1.10%, +1.55%]` and `[+1.88%, +2.53%]`, respectively.

Relative to INVARIANT, optimized PF is +2.26% in pooled TR at B=2 and tied at
+0.14% at B=4. The corresponding bootstrap intervals narrowly include zero:
`[-0.01%, +4.51%]` and `[-1.76%, +2.07%]`. Thus we report B=2 as a positive
point estimate and B=4 as parity, rather than claiming a statistically
resolved universal speed advantage.

Across the final four 100-prompt runs, all 400 optimized PF generations match
the original PF token-for-token and have identical block counts, AATPS, and
PF-ANLPPT/PF-Li/PF-PL scores. The engineering overhead has therefore been
reduced without changing the algorithm or watermark.

The final code also passes all 19 zero-argument Algorithm-2/Latin-PF
regression tests plus the two tree-free exactness tests. For seed 7, the final
outputs additionally match the saved pre-KV-optimization backend on 200/200
prompt/configuration pairs.

## Artifacts

- `decoder.py`: frozen backend implementation
- `test_decoder.py`: toy exactness tests
- `config_vicuna_b2.json`, `config_vicuna_b4.json`: benchmark configs
- `results/vicuna_b2_n100.json`, `results/vicuna_b4_n100.json`: raw rows
- `results/vicuna_b2_n100_kvview.json`,
  `results/vicuna_b4_n100_kvview.json`: final seed-7 rows
- `results/vicuna_b2_n100_seed107_kvview.json`,
  `results/vicuna_b4_n100_seed107_kvview.json`: final seed-107 rows
- `summarize.py`: exactness and paired-runtime summary
