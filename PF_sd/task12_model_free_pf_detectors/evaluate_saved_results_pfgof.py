#!/usr/bin/env python3
"""Model-agnostic PF goodness-of-fit detector evaluation on saved generations.

Detectors:
  1. PF-HC:
       one-sided Higher Criticism applied to recovered PF pivots V_t.
  2. PF-TrGoF-KL:
       a one-sided truncated Berk--Jones / Bernoulli-KL member of the
       Tr-GoF family, applied directly to PF pivots.

Under H0, V_t are iid Uniform(0,1) (for distinct context labels).
Under the PF watermark, pivots are biased toward 0, so both tests look for
an excess of small pivots.

IMPORTANT:
- This is post-hoc. It does NOT rerun LLM generation.
- It is model-agnostic: no target/draft logits are loaded.
- Fixed-length null distributions are calibrated by Monte Carlo.
- The Tr-GoF optimality theorem in Li et al. was proved for their Gumbel/edit
  model; this script uses the same GoF principle for PF but does not assume
  that theorem automatically transfers to PF.

Recommended first run:
    --n-null 50000
For final 0.1% FPR estimates:
    --n-null 200000

Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python PF_sd/task12_model_free_pf_detectors/evaluate_saved_results_pfgof.py \
  PF_sd/pilot_results/latin_B2_n200.json \
  PF_sd/pilot_results/pfgof_B2_n200.json \
  --method latin_pf_counter_fused --width 2 --device cuda:0 --n-null 200000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PF_sd.task12_model_free_pf_detectors.evaluate_saved_results import (
    _recover_rows,
    _recovery_settings,
)


DETECTORS = ("PF-HC", "PF-TrGoF-KL")


def _clip(v: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).tiny
    return np.clip(np.asarray(v, dtype=np.float64), eps, 1.0 - 1e-15)


def _candidate_arrays(sorted_v: np.ndarray, c_mult: float = 1.0):
    """Return (u, v, mask) for a lower-tail one-sided GoF scan.

    u_i = i/M is the empirical CDF evaluated at order statistic v_(i).

    We use a practical finite-sample lower truncation v >= c_mult/M and
    an upper scan cutoff v <= 1/2. This is the standard stability idea
    behind truncated HC/Tr-GoF: extremely tiny null order statistics can
    make the untruncated statistic heavy-tailed.

    c_mult=1 corresponds to the natural 1/M scale.
    """
    v = _clip(sorted_v)
    M = v.shape[-1]
    u = np.arange(1, M + 1, dtype=np.float64) / M
    lower = float(c_mult) / M
    mask = (v >= lower) & (v <= 0.5) & (u > v)
    return u, v, mask


def pf_hc_from_sorted(sorted_v: np.ndarray, c_mult: float = 1.0) -> np.ndarray:
    """One-sided Higher Criticism; accepts shape (..., M)."""
    u, v, mask = _candidate_arrays(sorted_v, c_mult)
    z = np.sqrt(v * (1.0 - v))
    values = math.sqrt(v.shape[-1]) * (u - v) / z
    values = np.where(mask, values, 0.0)
    return np.max(values, axis=-1)


def _binary_kl(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """KL(Ber(u)||Ber(v)), vectorized and numerically stable."""
    v = _clip(v)
    # xlogy(0, .) = 0 handles u=1 at the boundary.
    return xlogy(u, u / v) + xlogy(1.0 - u, (1.0 - u) / (1.0 - v))


def pf_trgof_kl_from_sorted(sorted_v: np.ndarray, c_mult: float = 1.0) -> np.ndarray:
    """Truncated one-sided Berk--Jones / KL GoF statistic.

    Returns M * sup KL(Ber(F_M(r)) || Ber(r)) over the lower-tail
    scan region where F_M(r)>r.
    """
    u, v, mask = _candidate_arrays(sorted_v, c_mult)
    values = v.shape[-1] * _binary_kl(u, v)
    values = np.where(mask, values, 0.0)
    return np.max(values, axis=-1)


def observed_score(pivots: np.ndarray, detector: str, c_mult: float) -> float:
    v = np.sort(_clip(np.asarray(pivots).reshape(-1)))
    if detector == "PF-HC":
        return float(pf_hc_from_sorted(v[None, :], c_mult)[0])
    if detector == "PF-TrGoF-KL":
        return float(pf_trgof_kl_from_sorted(v[None, :], c_mult)[0])
    raise ValueError(detector)


def simulate_null_scores(
    M: int,
    n_null: int,
    detector: str,
    c_mult: float,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Simulate the exact iid-Uniform null without sorting Uniform samples.

    Uniform order statistics can be generated from exponential spacings:
      U_(i) = (E_1+...+E_i)/(E_1+...+E_{M+1}).
    This is faster than sorting n_null independent length-M Uniform arrays.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_null, dtype=np.float64)
    pos = 0
    while pos < n_null:
        b = min(batch_size, n_null - pos)
        e = rng.exponential(scale=1.0, size=(b, M + 1))
        cs = np.cumsum(e, axis=1)
        order = cs[:, :M] / cs[:, [M]]
        if detector == "PF-HC":
            scores = pf_hc_from_sorted(order, c_mult)
        elif detector == "PF-TrGoF-KL":
            scores = pf_trgof_kl_from_sorted(order, c_mult)
        else:
            raise ValueError(detector)
        out[pos:pos+b] = scores
        pos += b
    out.sort()
    return out


def mc_p_value(sorted_null: np.ndarray, observed: float) -> float:
    left = int(np.searchsorted(sorted_null, observed, side="left"))
    exceed = len(sorted_null) - left
    return (1.0 + exceed) / (1.0 + len(sorted_null))


def null_threshold(sorted_null: np.ndarray, alpha: float) -> float:
    # Empirical upper alpha quantile.
    n = len(sorted_null)
    idx = int(math.ceil((1.0 - alpha) * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_null[idx])


def evaluate(
    payload: dict,
    method: str,
    device: str,
    width: int | None,
    n_null: int,
    c_mult: float,
    seed: int,
    batch_size: int,
) -> dict:
    sequences = _recover_rows(payload, method, device, width=width)
    target_coupling, rng_backend = _recovery_settings(method)

    by_length = defaultdict(list)
    for s, pivots in enumerate(sequences):
        by_length[len(pivots)].append((s, pivots))

    results = {}
    per_sequence = [{} for _ in sequences]

    for detector_index, detector in enumerate(DETECTORS):
        evaluated = [None] * len(sequences)
        for M, items in sorted(by_length.items()):
            null = simulate_null_scores(
                M=M,
                n_null=n_null,
                detector=detector,
                c_mult=c_mult,
                seed=seed + 1000003 * detector_index + 9176 * M,
                batch_size=batch_size,
            )
            t01 = null_threshold(null, 0.01)
            t001 = null_threshold(null, 0.001)

            for seq_idx, pivots in items:
                score = observed_score(pivots, detector, c_mult)
                p = mc_p_value(null, score)
                row = {
                    "score": score,
                    "score_per_token": score / M,
                    "p_value": p,
                    "ANLPPT": -math.log(p) / M,
                    "detect_0.01": bool(score >= t01),
                    "detect_0.001": bool(score >= t001),
                }
                evaluated[seq_idx] = row
                per_sequence[seq_idx].update({
                    "sequence_idx": seq_idx,
                    "length": M,
                    f"{detector}_score": score,
                    f"{detector}_p": p,
                    f"{detector}_ANLPPT": row["ANLPPT"],
                })

        results[detector] = {
            "n_sequences": len(evaluated),
            "mean_score_per_token": float(np.mean([x["score_per_token"] for x in evaluated])),
            "mean_ANLPPT": float(np.mean([x["ANLPPT"] for x in evaluated])),
            "median_ANLPPT": float(np.median([x["ANLPPT"] for x in evaluated])),
            "TPR@FPR=0.01": float(np.mean([x["detect_0.01"] for x in evaluated])),
            "TPR@FPR=0.001": float(np.mean([x["detect_0.001"] for x in evaluated])),
        }

    return {
        "source_config": payload["config"],
        "evaluated_method": method,
        "evaluated_width": width,
        "target_coupling": target_coupling,
        "rng_backend": rng_backend,
        "model_access": False,
        "null_calibration": "Monte Carlo iid Uniform order-statistic null",
        "n_null": n_null,
        "gof_lower_truncation": f"{c_mult:g}/M",
        "gof_upper_scan": 0.5,
        "detectors": results,
        "per_sequence": per_sequence,
        "notes": {
            "PF-HC": "One-sided Higher Criticism for excess small PF pivots.",
            "PF-TrGoF-KL": (
                "One-sided truncated Bernoulli-KL/Berk-Jones GoF statistic. "
                "This is a PF adaptation of the Tr-GoF principle; Li et al.'s "
                "Gumbel/edit optimality theorem is not automatically a PF theorem."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method", default="latin_pf_counter_fused")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int)
    parser.add_argument("--n-null", type=int, default=50_000)
    parser.add_argument("--c-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    with args.input.open() as handle:
        payload = json.load(handle)

    result = evaluate(
        payload=payload,
        method=args.method,
        device=args.device,
        width=args.width,
        n_null=args.n_null,
        c_mult=args.c_mult,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["detectors"], indent=2))


if __name__ == "__main__":
    main()
