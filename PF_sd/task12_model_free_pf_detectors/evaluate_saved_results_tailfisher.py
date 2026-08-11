#!/usr/bin/env python3
"""Post-hoc model-agnostic Tail-Fisher evaluation for saved PF generations.

This script DOES NOT rerun LLM generation and DOES NOT load target/draft logits.
It reuses the repository's existing pivot-recovery code, then evaluates a
PF-specific truncated/soft-thresholded Fisher family

    T_tau(V_1:M) = sum_t 1{V_t <= tau} log(tau / V_t).

For tau=1 this is exactly the original PF statistic sum_t -log(V_t).

Under the iid Uniform(0,1) null:
    K_tau = #{t: V_t <= tau} ~ Binomial(M, tau),
and conditional on K_tau=k,
    T_tau | K_tau=k ~ Gamma(k, 1).
Therefore fixed-tau p-values are calibrated EXACTLY by a Binomial-Gamma mixture;
no Monte Carlo is required.

The script also reports a conservative Bonferroni omnibus across the supplied
tau grid. For a higher-power adaptive omnibus, jointly Monte-Carlo calibrate
min_tau p_tau under one shared Uniform null sample.

Example:
CUDA_VISIBLE_DEVICES=0 \
python PF_sd/task12_model_free_pf_detectors/evaluate_saved_results_tailfisher.py \
  PF_sd/pilot_results/latin_B2_n200.json \
  PF_sd/pilot_results/tailfisher_B2_n200.json \
  --method latin_pf_counter_fused \
  --width 2 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom

from PF_sd.task12_model_free_pf_detectors.evaluate_saved_results import (
    _recover_rows,
    _recovery_settings,
)


DEFAULT_TAUS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def _clip_pivots(pivots: np.ndarray) -> np.ndarray:
    v = np.asarray(pivots, dtype=np.float64).reshape(-1)
    return np.clip(v, np.finfo(np.float64).tiny, 1.0)


def tail_fisher_statistic(pivots: np.ndarray, tau: float) -> float:
    """T_tau = sum 1{V<=tau} log(tau/V). Larger = more watermarked."""
    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0,1]")
    v = _clip_pivots(pivots)
    mask = v <= tau
    if not np.any(mask):
        return 0.0
    return float(np.log(tau / v[mask]).sum(dtype=np.float64))


def tail_fisher_null_mean(length: int, tau: float) -> float:
    return float(length) * float(tau)


def tail_fisher_null_variance(length: int, tau: float) -> float:
    tau = float(tau)
    return float(length) * (2.0 * tau - tau * tau)


def tail_fisher_p_value(statistic: float, length: int, tau: float) -> float:
    """Exact upper-tail p-value under iid Uniform pivots.

    If K ~ Bin(M,tau) and T|K=k ~ Gamma(k,1), then for statistic x>0:
       P_0(T >= x) = sum_{k=1}^M P(K=k) Q(k,x),
    where Q is the regularized upper incomplete gamma function.

    T has a point mass at zero when K=0. If x<=0 the valid p-value is 1.
    """
    x = float(statistic)
    M = int(length)
    tau = float(tau)
    if M <= 0:
        raise ValueError("length must be positive")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0,1]")
    if x <= 0.0:
        return 1.0

    k = np.arange(1, M + 1)
    weights = binom.pmf(k, M, tau)
    tails = gammaincc(k, x)
    p = float(np.dot(weights, tails))
    # Numerical guard only.
    return min(1.0, max(np.finfo(np.float64).tiny, p))


def evaluate_tail_fisher(
    payload: dict,
    method: str,
    device: str,
    width: int | None,
    taus: tuple[float, ...],
) -> dict:
    sequences = _recover_rows(payload, method, device, width=width)
    target_coupling, rng_backend = _recovery_settings(method)

    taus = tuple(sorted({float(t) for t in taus}))
    for tau in taus:
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"invalid tau={tau}")

    results = {}
    per_sequence = []

    # Fixed-tau tests.
    evaluated_by_tau = {}
    for tau in taus:
        evaluated = []
        for pivots in sequences:
            M = len(pivots)
            score = tail_fisher_statistic(pivots, tau)
            p = tail_fisher_p_value(score, M, tau)
            z = (score - tail_fisher_null_mean(M, tau)) / math.sqrt(
                tail_fisher_null_variance(M, tau)
            )
            evaluated.append(
                {
                    "score": score,
                    "score_per_token": score / M,
                    "z_null_standardized": z,
                    "p_value": p,
                    "ANLPPT": -math.log(p) / M,
                }
            )
        evaluated_by_tau[tau] = evaluated
        results[f"TailFisher-tau={tau:g}"] = {
            "n_sequences": len(evaluated),
            "mean_score_per_token": float(
                np.mean([row["score_per_token"] for row in evaluated])
            ),
            "mean_ANLPPT": float(np.mean([row["ANLPPT"] for row in evaluated])),
            "median_ANLPPT": float(np.median([row["ANLPPT"] for row in evaluated])),
            "TPR@FPR=0.01": float(
                np.mean([row["p_value"] <= 0.01 for row in evaluated])
            ),
            "TPR@FPR=0.001": float(
                np.mean([row["p_value"] <= 0.001 for row in evaluated])
            ),
        }

    # Conservative parameter-adaptive version. This is valid for any dependence
    # among the fixed-tau p-values because of Bonferroni.
    omni_rows = []
    for s in range(len(sequences)):
        ps = np.array(
            [evaluated_by_tau[tau][s]["p_value"] for tau in taus], dtype=np.float64
        )
        best = int(np.argmin(ps))
        p_omni = min(1.0, len(taus) * float(ps[best]))
        M = len(sequences[s])
        omni_rows.append(
            {
                "best_tau": taus[best],
                "raw_min_p": float(ps[best]),
                "p_value": p_omni,
                "ANLPPT": -math.log(p_omni) / M,
            }
        )
        per_sequence.append(
            {
                "sequence_idx": s,
                "length": M,
                "best_tau": taus[best],
                "TailFisher-Bonferroni-p": p_omni,
                **{
                    f"p_tau_{tau:g}": float(evaluated_by_tau[tau][s]["p_value"])
                    for tau in taus
                },
            }
        )

    results["TailFisher-Omnibus-Bonferroni"] = {
        "n_sequences": len(omni_rows),
        "mean_ANLPPT": float(np.mean([row["ANLPPT"] for row in omni_rows])),
        "median_ANLPPT": float(np.median([row["ANLPPT"] for row in omni_rows])),
        "TPR@FPR=0.01": float(
            np.mean([row["p_value"] <= 0.01 for row in omni_rows])
        ),
        "TPR@FPR=0.001": float(
            np.mean([row["p_value"] <= 0.001 for row in omni_rows])
        ),
        "best_tau_histogram": {
            f"{tau:g}": int(sum(row["best_tau"] == tau for row in omni_rows))
            for tau in taus
        },
        "note": (
            "Bonferroni is deliberately conservative. If fixed Tail-Fisher looks "
            "promising, use a jointly Monte-Carlo-calibrated omnibus for higher power."
        ),
    }

    return {
        "source_config": payload["config"],
        "evaluated_method": method,
        "evaluated_width": width,
        "target_coupling": target_coupling,
        "rng_backend": rng_backend,
        "model_access": False,
        "null_calibration": "exact Binomial-Gamma mixture for each fixed tau",
        "taus": list(taus),
        "detectors": results,
        "per_sequence": per_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="latin_pf_counter_fused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int)
    parser.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    args = parser.parse_args()

    with args.input.open() as handle:
        payload = json.load(handle)

    result = evaluate_tail_fisher(
        payload,
        args.method,
        args.device,
        args.width,
        tuple(args.taus),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result["detectors"], indent=2))


if __name__ == "__main__":
    main()
