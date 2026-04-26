# KL Discrete Watermark Strength Measurement

This document records the implemented measurement method for empirical discrete
KL watermark strength. It replaces the earlier implementation plan.

The code path is:

```text
code/real/unbiased_watermark/scores/kl_watermark_strength.py
code/real/my_experiment/kl_watermark_strength_experiment.py
```

The unified JSON-config entry is:

```bash
PYTHONPATH=code/real python code/real/my_experiment/kl_watermark_strength_experiment.py \
  --config code/real/my_experiment/configs/kl_watermark_strength_config.json
```

Edit only the JSON file to change models, datasets, sample counts, methods,
lookahead, SynthID depth, keys, and output paths. The config supports either one
experiment or a top-level `experiments` list.

## Target Quantity

For a watermarking scheme with keyed randomness `zeta`, the theoretical strength
is:

```text
WS(P_zeta) = E_zeta [ D_KL(P_zeta || P) ].
```

The experiment estimates this empirically over generated prefixes:

```text
KL_WS_mean  = average_context D_KL(P_zeta(. | prefix) || P(. | prefix))
KL_WS_ratio = KL_WS_sum / entropy_sum
```

where `entropy_sum` is computed from the target model full-vocabulary
distribution `P`, unless explicitly stated otherwise. This is a white-box
evaluation metric, not a deployable detector.

## Shared Conventions

- `P` is always the target model distribution at the final output prefix.
- `Q` is the draft model distribution at the same final output prefix.
- The private key is used to reconstruct the keyed distribution, not to score
  only the realized output token.
- The estimator uses generated prefixes as empirical contexts.
- The reported `KL_WS_ratio` uses full target entropy:

```text
ratio = sum_context KL(P_zeta || P) / sum_context H(P)
```

- For UWM methods, `top_k` is not used. The UWM distributions are full-vocab.
- For SynthID methods, `top_k` is still an algorithm parameter because the
  current SynthID generation/reweighting implementation watermarks only the
  top-k candidate set, then scatters the resulting distribution back into the
  full vocabulary. The ratio denominator is still full `H(P)`.

## Basic UWM

Methods:

```text
basic_uwm
```

Measurement:

```text
P_zeta = UWM_reweight(P, key, prefix)
score  = D_KL(P_zeta || P)
```

Implementation details:

- Recompute target logits on `input_ids + output_ids`.
- Reconstruct the target-side UWM code with the target private key.
- Apply the same repeated-context skip mask as the UWM code path.
- Compute full-vocab `D_KL(q || p)`.

Kind in output:

```text
target_side_full_vocab_key_avg
```

This is an exact discrete KL for each visited context, empirically averaged over
the generated contexts and optional key repeats.

## MC-UWM Strength

Methods:

```text
mc_uwm_strength
```

Measurement:

```text
P_zeta = UWM_reweight(P, key, prefix)
score  = D_KL(P_zeta || P)
```

Rationale:

`mc_uwm_strength` uses a target-side watermarked distribution in the speculative
correction target. The final marginal is intended to preserve the same
target-side watermarked distribution as Basic UWM, so the KL measurement is the
same target-side full-vocab reconstruction.

The estimator does not split output tokens by whether they were accepted from
draft or sampled from residual. Definition 3.1 is about the output distribution,
not the internal source label of a sampled token.

Kind in output:

```text
target_side_full_vocab_key_avg
```

## MC-UWM Speed

Methods:

```text
mc_uwm_speed
```

Measurement:

```text
Q_zeta = UWM_reweight(Q, target_key, prefix)
alpha(w) = min(1, P(w) / Q(w))
A = sum_v Q_zeta(v) alpha(v)
R(w) = (P(w) - Q(w))_+ / sum_v (P(v) - Q(v))_+

P_speed,zeta(w) = Q_zeta(w) alpha(w) + (1 - A) R(w)

score = D_KL(P_speed,zeta || P)
```

Rationale:

The speed variant watermarks the draft-side proposal but accepts/rejects against
raw target/draft distributions. The effective final output distribution is
therefore diluted relative to target-side UWM. We reconstruct that final
one-step marginal distribution and compare it to target `P`.

Important caveat:

This is a one-step effective-distribution estimator evaluated at visited final
prefixes. For lookahead `n > 1`, it is not the exact full block-level marginal
over every draft suffix and accept/reject path. It is the implemented empirical
proxy used to compare whether the watermark is diluted.

Kind in output:

```text
mc_uwm_speed_alpha_avg_one_step_key_avg
```

## MC-UWM SynthID Pseudo-R

Methods:

```text
mc_uwm_synthid_psedo_r
```

Despite the name, this method uses the UWM reweight class in the current Qwen
UWM experiments, with an additional keyed pseudo-r acceptance variable.

Measurement:

```text
Q_zeta = UWM_reweight(Q, target_key, prefix)
r_zeta = pseudo-r(prefix, acceptance_key)
alpha(w) = min(1, P(w) / Q(w))
I_r(w) = 1[r_zeta <= alpha(w)]
A_r = sum_v Q_zeta(v) I_r(v)

R_zeta(w) = UWM_reweight(R, target_key, prefix)
R(w) = (P(w) - Q(w))_+ / sum_v (P(v) - Q(v))_+

P_pseudo-r,zeta(w) = Q_zeta(w) I_r(w) + (1 - A_r) R_zeta(w)

score = D_KL(P_pseudo-r,zeta || P)
```

Key usage:

- The accept/reject random variable uses the dedicated `mc_private_key`.
- The rejected residual sampling is reconstructed with the target-side
  `private_key`.

This matches the current fixed implementation in both generation and KL
measurement.

Kind in output:

```text
mc_uwm_pseudo_r_fixed_zeta_one_step_key_avg
```

## SynthID Watermark

Methods:

```text
synthid_basic
```

Measurement:

```text
P_topk,zeta = SynthID_reweight(top_k(P), key, prefix)
P_zeta_full = scatter P_topk,zeta into full vocab, with zero mass outside top-k
score = D_KL(P_zeta_full || P_full)
```

Rationale:

The current SynthID implementation applies the SynthID signal on the top-k
candidate set for sampling efficiency. However, for ratio reporting we compare
against the full target baseline and divide by full target entropy. This avoids
using top-k entropy as the denominator.

Kind in output:

```text
synthid_dense_topk_full_p_key_avg
```

## SynthID MC: MSE

Methods:

```text
mc_mse
```

Measurement:

```text
Q_zeta = SynthID_reweight(top_k(Q), key, prefix)
P_mse,zeta = get_mse_target_logprob(Q_zeta, Q, P)
score = D_KL(P_mse,zeta || P)
```

Implementation notes:

- `Q_zeta` is dense after scattering the top-k SynthID distribution into the
  full vocabulary.
- `get_mse_target_logprob` constructs the effective MSE target distribution.
- The baseline and denominator are full target `P`.

Kind in output:

```text
mc_mse_synthid_mc_one_step_key_avg
```

## SynthID MC: MWS

Methods:

```text
mc_mws
```

Measurement:

```text
proposal = Q_zeta
target   = P_zeta
alpha(w) = min(1, P_zeta(w) / Q_zeta(w))
R(w) = (P_zeta(w) - Q_zeta(w))_+ / sum_v (P_zeta(v) - Q_zeta(v))_+

P_mws,zeta(w) = Q_zeta(w) alpha(w) + (1 - A) R(w)

score = D_KL(P_mws,zeta || P)
```

Rationale:

MWS makes the speculative correction target the target-side SynthID
watermarked distribution. On the same prefix distribution, it should track
basic SynthID closely; differences in full experiments can come from different
generated trajectories.

Kind in output:

```text
mc_mws_synthid_mc_one_step_key_avg
```

## SynthID MC: 2 Keys

Methods:

```text
mc_2keys
```

Measurement:

```text
proposal = Q_zeta
target_accept = P
proposal_accept = Q
alpha(w) = min(1, P(w) / Q(w))
R(w) = (P(w) - Q(w))_+ / sum_v (P(v) - Q(v))_+
R_zeta = SynthID_reweight(top_k(R), residual_target_key, prefix)

P_2keys,zeta(w) = Q_zeta(w) alpha(w) + (1 - A) R_zeta(w)

score = D_KL(P_2keys,zeta || P)
```

Key usage:

- Dedicated MC key controls the pseudo-r acceptance variable in generation.
- The residual SynthID distribution is reconstructed with the target-side
  SynthID key, not the acceptance key.

Kind in output:

```text
mc_2keys_synthid_mc_one_step_key_avg
```

## Ratio and AATPS Reporting

The JSON files store KL statistics directly. AATPS is not always stored as a
summary field in older outputs, but new outputs from the JSON-config entry write
it into each method summary.

`AATPS` is computed from output progress per generation step:

```text
AATPS = sum(chunk_lengths) / len(chunk_lengths)
```

Here `chunk_lengths` is the number of output tokens advanced by one generation
step. This includes accepted draft tokens, the residual token sampled after
rejection, and the extra target token sampled after a fully accepted speculative
block. For non-speculative methods such as `basic_uwm` and `synthid_basic`,
`AATPS` is therefore `1`.

## Interpreting Results

Expected aggregate pattern for UWM:

```text
basic_uwm ~= mc_uwm_strength ~= mc_uwm_synthid_psedo_r
mc_uwm_speed is lower
```

Expected aggregate pattern for SynthID:

```text
synthid_basic ~= mc_mws
mc_mse is lower
mc_2keys is close to SynthID/MWS when residual randomness is recoverable
```

The equality is approximate because each method can generate different prefixes.
When comparing estimator logic itself, use common-prefix diagnostics.
