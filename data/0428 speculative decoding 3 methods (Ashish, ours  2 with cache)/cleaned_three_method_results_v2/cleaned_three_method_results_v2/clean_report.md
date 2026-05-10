# Three-method experiment cleaning report

Target complete block size: `150` prompts. Expected full run per seed: `36` dataset/config blocks.

## File classification

- `three_method_raw_seed2155929800_2026-04-28-214320.csv`: kind=raw, rows=5400, seed_values=2155929800, complete_blocks_at_150=36.0, partial_blocks_at_150=0.0.
- `three_method_raw_seed517798609_2026-04-28-195913.csv`: kind=raw, rows=13, seed_values=517798609, complete_blocks_at_150=0.0, partial_blocks_at_150=1.0.
- `three_method_raw_seed517798609_2026-04-28-200311.csv`: kind=raw, rows=1540, seed_values=517798609, complete_blocks_at_150=10.0, partial_blocks_at_150=1.0.
- `three_method_raw_seed517798609_2026-04-28-232912.csv`: kind=raw, rows=900, seed_values=517798609, complete_blocks_at_150=6.0, partial_blocks_at_150=0.0.
- `three_method_raw_seed929093658_2026-04-28-200323.csv`: kind=empty/useless, rows=0, seed_values=nan, complete_blocks_at_150=nan, partial_blocks_at_150=nan.
- `three_method_raw_seed929093658_2026-04-28-201721.csv`: kind=raw, rows=5400, seed_values=929093658, complete_blocks_at_150=36.0, partial_blocks_at_150=0.0.

## Seed coverage

- seed `517798609`: 16/36 complete blocks selected (incomplete)
- seed `929093658`: 36/36 complete blocks selected (complete seed)
- seed `2155929800`: 36/36 complete blocks selected (complete seed)

## Critical note

The uploaded files contain two complete seeds (`2155929800` and `929093658`). Seed `517798609` has only 16 complete dataset/config blocks after stitching its resume file, so a balanced three-seed average over all methods/datasets is not available from the uploaded files. The reliable paper-style aggregate over all three datasets/configs is therefore the balanced complete-seed table using seeds `2155929800` and `929093658`. The script also writes available-seed and strict-three-seed tables separately.

## Balanced complete-seed overall table

| method                         |   max_num_drafts |   seed_count |   token_rate_mean |   aatps_mean |   be_mean |   acceptance_fraction_mean |   target_forward_calls_per_token_mean |   draft_forward_calls_per_token_mean |
|:-------------------------------|-----------------:|-------------:|------------------:|-------------:|----------:|---------------------------:|--------------------------------------:|-------------------------------------:|
| ashish_invariant               |                2 |            2 |           22.3470 |       3.0398 |    4.0398 |                     0.7600 |                                0.2528 |                               0.9986 |
| mpfr_batched_torchgen_cached   |                2 |            2 |           25.5872 |       3.0432 |    4.0192 |                     0.7697 |                                0.2539 |                               1.0037 |
| multi_draft_pfr_batched_cached |                2 |            2 |           25.8164 |       3.0500 |    4.0255 |                     0.7762 |                                0.2534 |                               1.0135 |
| ashish_invariant               |                4 |            2 |           16.9655 |       3.2800 |    4.2800 |                     0.8200 |                                0.2371 |                               0.9357 |
| mpfr_batched_torchgen_cached   |                4 |            2 |           23.7441 |       3.3107 |    4.2843 |                     0.8376 |                                0.2366 |                               0.9354 |
| multi_draft_pfr_batched_cached |                4 |            2 |           24.3088 |       3.3337 |    4.3073 |                     0.8484 |                                0.2352 |                               0.9409 |
| ashish_invariant               |                6 |            2 |           13.6467 |       3.3945 |    4.3945 |                     0.8486 |                                0.2302 |                               0.9080 |
| mpfr_batched_torchgen_cached   |                6 |            2 |           22.1842 |       3.4382 |    4.4118 |                     0.8695 |                                0.2290 |                               0.9053 |
| multi_draft_pfr_batched_cached |                6 |            2 |           22.3888 |       3.4378 |    4.4102 |                     0.8756 |                                0.2292 |                               0.9167 |
| ashish_invariant               |                8 |            2 |           12.1662 |       3.4719 |    4.4719 |                     0.8680 |                                0.2259 |                               0.8919 |
| mpfr_batched_torchgen_cached   |                8 |            2 |           21.0824 |       3.5018 |    4.4750 |                     0.8851 |                                0.2254 |                               0.8917 |
| multi_draft_pfr_batched_cached |                8 |            2 |           20.9210 |       3.5017 |    4.4739 |                     0.8920 |                                0.2255 |                               0.9021 |

## Balanced relative comparison versus ashish_invariant

| method                         |   max_num_drafts |   token_rate_rel_pct |   aatps_rel_pct |   be_rel_pct |   acceptance_fraction_rel_pct |
|:-------------------------------|-----------------:|---------------------:|----------------:|-------------:|------------------------------:|
| mpfr_batched_torchgen_cached   |                2 |               14.500 |           0.111 |       -0.511 |                         1.283 |
| multi_draft_pfr_batched_cached |                2 |               15.525 |           0.333 |       -0.356 |                         2.140 |
| mpfr_batched_torchgen_cached   |                4 |               39.955 |           0.937 |        0.101 |                         2.143 |
| multi_draft_pfr_batched_cached |                4 |               43.284 |           1.640 |        0.638 |                         3.459 |
| mpfr_batched_torchgen_cached   |                6 |               62.561 |           1.288 |        0.393 |                         2.458 |
| multi_draft_pfr_batched_cached |                6 |               64.060 |           1.276 |        0.359 |                         3.173 |
| mpfr_batched_torchgen_cached   |                8 |               73.286 |           0.861 |        0.068 |                         1.971 |
| multi_draft_pfr_batched_cached |                8 |               71.960 |           0.859 |        0.045 |                         2.770 |