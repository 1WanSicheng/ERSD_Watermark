#!/usr/bin/env python3
"""Evaluate PF Fractional-Fisher (PF-FF) scores on saved PF generations.

PF-FF is a PF-specific one-parameter family connecting the original Fisher/PF
score to stronger lower-tail fractional-power scores.

For 0 < q < 1/2 define
    S_q(v) = (v^{-q} - 1)/q - 1/(1-q).

For q=0 define the continuous limit
    S_0(v) = -log(v) - 1.

Under U~Uniform(0,1):
    E[S_q(U)] = 0,
and for 0<q<1/2
    Var[S_q(U)]
      = (1/q^2) [ 1/(1-2q) - 1/(1-q)^2 ].

Thus q=0 is exactly the centered original PF/Fisher score, while q approaching
1/2 increasingly emphasizes very small pivots. At q=1/2 the null variance
diverges, explaining why the Lattimore endpoint requires truncation.

This script:
- reuses saved generations and keyed pivot recovery;
- is fully model-agnostic;
- does NOT rerun LLM generation;
- calibrates q=0 analytically using Gamma(M,1);
- calibrates q>0 by iid-Uniform Monte Carlo for each observed sequence length.

Example:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python PF_sd/task12_model_free_pf_detectors/evaluate_saved_results_fractional_fisher.py \
  PF_sd/pilot_results/latin_B2_n200.json \
  PF_sd/pilot_results/pfff_B2_n200.json \
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
from scipy.stats import gamma as gamma_dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PF_sd.task12_model_free_pf_detectors.evaluate_saved_results import (
    _recover_rows,
    _recovery_settings,
)


DEFAULT_QS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


def _clip(v: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(v, dtype=np.float64),
        np.finfo(np.float64).tiny,
        1.0,
    )


def pf_ff_contrib(v: np.ndarray, q: float) -> np.ndarray:
    v = _clip(v)
    q = float(q)
    if abs(q) < 1e-14:
        return -np.log(v) - 1.0
    if not 0.0 < q < 0.5:
        raise ValueError("PF-FF requires q in [0, 0.5)")
    return (np.power(v, -q) - 1.0) / q - 1.0 / (1.0 - q)


def pf_ff_statistic(pivots: np.ndarray, q: float) -> float:
    return float(pf_ff_contrib(np.asarray(pivots), q).sum(dtype=np.float64))


def pf_ff_null_variance_per_token(q: float) -> float:
    q = float(q)
    if abs(q) < 1e-14:
        return 1.0
    return (
        1.0 / (q * q)
        * (1.0 / (1.0 - 2.0 * q) - 1.0 / ((1.0 - q) ** 2))
    )


def original_exact_p(pivots: np.ndarray) -> float:
    v = _clip(np.asarray(pivots).reshape(-1))
    M = len(v)
    fisher = float((-np.log(v)).sum())
    return float(gamma_dist.sf(fisher, a=M, scale=1.0))


def simulate_null_for_q(
    lengths: list[int],
    n_null: int,
    q: float,
    seed: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Simulate additive PF-FF nulls for all needed lengths in one pass.

    We draw Uniform vectors up to max(lengths), compute cumulative PF-FF scores,
    and retain only the columns corresponding to observed sequence lengths.
    """
    max_M = max(lengths)
    col = {M: j for j, M in enumerate(lengths)}
    out = np.empty((n_null, len(lengths)), dtype=np.float32)
    rng = np.random.default_rng(seed)

    pos = 0
    while pos < n_null:
        b = min(batch_size, n_null - pos)
        u = rng.random((b, max_M), dtype=np.float64)
        contrib = pf_ff_contrib(u, q)
        cs = np.cumsum(contrib, axis=1)
        for M, j in col.items():
            out[pos:pos+b, j] = cs[:, M - 1]
        pos += b

    result = {}
    for M, j in col.items():
        arr = out[:, j].astype(np.float64, copy=True)
        arr.sort()
        result[M] = arr
    return result


def mc_p_value(sorted_null: np.ndarray, observed: float) -> float:
    left = int(np.searchsorted(sorted_null, observed, side="left"))
    exceed = len(sorted_null) - left
    return (1.0 + exceed) / (1.0 + len(sorted_null))


def threshold(sorted_null: np.ndarray, alpha: float) -> float:
    n = len(sorted_null)
    idx = int(math.ceil((1.0 - alpha) * n)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_null[idx])


def evaluate(
    payload: dict,
    method: str,
    device: str,
    width: int | None,
    qs: tuple[float, ...],
    n_null: int,
    seed: int,
    batch_size: int,
) -> dict:
    sequences = _recover_rows(payload, method, device, width=width)
    target_coupling, rng_backend = _recovery_settings(method)
    lengths = sorted({len(v) for v in sequences})

    qs = tuple(sorted({float(q) for q in qs}))
    if not qs or qs[0] < 0 or qs[-1] >= 0.5:
        raise ValueError("all q values must lie in [0,0.5)")

    results = {}
    per_sequence = [
        {"sequence_idx": i, "length": len(v)}
        for i, v in enumerate(sequences)
    ]

    for qi, q in enumerate(qs):
        name = f"PF-FF-q={q:g}"
        evaluated = []

        if abs(q) < 1e-14:
            for i, pivots in enumerate(sequences):
                M = len(pivots)
                centered = pf_ff_statistic(pivots, 0.0)
                p = max(np.finfo(np.float64).tiny, original_exact_p(pivots))
                # Gamma calibration is exact for the original uncentered score.
                # Centering by M does not affect ordering or p-values.
                # Exact alpha rejection is p <= alpha.
                row = {
                    "score": centered,
                    "score_per_token": centered / M,
                    "z_null_standardized": centered / math.sqrt(M),
                    "p_value": p,
                    "ANLPPT": -math.log(p) / M,
                    "detect_0.01": p <= 0.01,
                    "detect_0.001": p <= 0.001,
                }
                evaluated.append(row)
                per_sequence[i].update({
                    f"{name}_score": centered,
                    f"{name}_p": p,
                    f"{name}_ANLPPT": row["ANLPPT"],
                })
        else:
            nulls = simulate_null_for_q(
                lengths=lengths,
                n_null=n_null,
                q=q,
                seed=seed + 1000003 * qi,
                batch_size=batch_size,
            )
            for i, pivots in enumerate(sequences):
                M = len(pivots)
                score = pf_ff_statistic(pivots, q)
                null = nulls[M]
                p = mc_p_value(null, score)
                row = {
                    "score": score,
                    "score_per_token": score / M,
                    "z_null_standardized": (
                        score / math.sqrt(M * pf_ff_null_variance_per_token(q))
                    ),
                    "p_value": p,
                    "ANLPPT": -math.log(p) / M,
                    "detect_0.01": score >= threshold(null, 0.01),
                    "detect_0.001": score >= threshold(null, 0.001),
                }
                evaluated.append(row)
                per_sequence[i].update({
                    f"{name}_score": score,
                    f"{name}_p": p,
                    f"{name}_ANLPPT": row["ANLPPT"],
                })

        results[name] = {
            "n_sequences": len(evaluated),
            "mean_score_per_token": float(np.mean([r["score_per_token"] for r in evaluated])),
            "mean_ANLPPT": float(np.mean([r["ANLPPT"] for r in evaluated])),
            "median_ANLPPT": float(np.median([r["ANLPPT"] for r in evaluated])),
            "TPR@FPR=0.01": float(np.mean([r["detect_0.01"] for r in evaluated])),
            "TPR@FPR=0.001": float(np.mean([r["detect_0.001"] for r in evaluated])),
            "null_variance_per_token": pf_ff_null_variance_per_token(q),
        }

    return {
        "source_config": payload["config"],
        "evaluated_method": method,
        "evaluated_width": width,
        "target_coupling": target_coupling,
        "rng_backend": rng_backend,
        "model_access": False,
        "family_name": "PF Fractional-Fisher (PF-FF)",
        "qs": list(qs),
        "n_null_for_q_gt_0": n_null,
        "detectors": results,
        "per_sequence": per_sequence,
        "notes": {
            "q=0": "Exactly the centered Original PF/Fisher statistic; exact Gamma calibration.",
            "q>0": (
                "Fractional lower-tail amplification with finite Uniform-null variance "
                "for every q<1/2. As q approaches 1/2, extreme small pivots receive "
                "increasing weight."
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
    parser.add_argument("--qs", type=float, nargs="+", default=list(DEFAULT_QS))
    parser.add_argument("--n-null", type=int, default=50_000)
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
        qs=tuple(args.qs),
        n_null=args.n_null,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["detectors"], indent=2))


if __name__ == "__main__":
    main()
