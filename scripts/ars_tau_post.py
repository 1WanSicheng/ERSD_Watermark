"""Ars-tau post-hoc analysis (improving_KL run_repeated_threshold_training).

Reads a single_draft JSON with mc_uwm_pseudo_r dual-key components
(`dual_Us_pk`, `dual_Us_mc`, `dual_r` per prompt) and runs:

  for trial in 1..N_trials:
      tr, te = random 50/50 split of prompts
      grid-search tau over [0, 0.01, ..., 1.0] on tr to maximize
          ars_score(mix_by_tau(Y_pk_tr, Y_mc_tr, R_tr))[-1]
          (= TPR @ final token under theoretical Gamma threshold)
      apply tau* on te -> test detection curve

Reports: per-token TPR mean +/- std across test trials, chosen tau
distribution, final-token TPR.  Mirrors improving_KL/my_experiment/
gumbel_detect_utils.run_repeated_threshold_training with `objective="final"`,
`grid_size=101`, theoretical Gamma threshold from `ars_score.compute_gamma`.

Usage:
    python scripts/ars_tau_post.py outputs/2x2/2x2_n1000/qwen_eli5_n1000.json \\
        --output outputs/2x2/2x2_n1000/qwen_eli5_ars_tau.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gamma
from sklearn.model_selection import train_test_split


def ars_score_curve(mixed_u: np.ndarray, alpha: float = 0.01):
    """Per-token TPR @ FPR=alpha using theoretical Gamma(t,1) (1-alpha)-pct.

    Args:
        mixed_u: (P, T) per-prompt per-token uniforms
    Returns:
        curve: (T,) TPR at each token position
        cum:   (P, T) cumulative -log(1-u) per prompt
    """
    mixed = np.clip(mixed_u, 1e-12, 1 - 1e-12)
    cum = (-np.log(1 - mixed)).cumsum(axis=1)
    T = cum.shape[1]
    thr = gamma.ppf(1 - alpha, np.arange(1, T + 1))
    return (cum >= thr).mean(axis=0), cum


def mix_by_tau(Y_pk, Y_mc, R, tau):
    return np.where(R > tau, Y_mc, Y_pk)


def fit_tau_train(Y_pk_tr, Y_mc_tr, R_tr, alpha=0.01, grid_size=101):
    best_t, best_score = 0.5, -np.inf
    for t in np.linspace(0.0, 1.0, grid_size):
        mixed = mix_by_tau(Y_pk_tr, Y_mc_tr, R_tr, float(t))
        curve, _ = ars_score_curve(mixed, alpha=alpha)
        s = float(curve[-1])
        if s > best_score:
            best_t, best_score = float(t), s
    return best_t, best_score


def run_repeated_threshold(Y_pk, Y_mc, R, *, n_trials=10, alpha=0.01,
                           grid_size=101, seed=0):
    m = Y_pk.shape[0]
    rng = np.random.RandomState(seed)
    test_curves, chosen = [], []
    for _ in range(n_trials):
        idx = np.arange(m)
        tr, te = train_test_split(idx, test_size=0.5,
                                  random_state=rng.randint(0, 10**9))
        tau_star, _ = fit_tau_train(Y_pk[tr], Y_mc[tr], R[tr],
                                    alpha=alpha, grid_size=grid_size)
        mixed_te = mix_by_tau(Y_pk[te], Y_mc[te], R[te], tau_star)
        curve, _ = ars_score_curve(mixed_te, alpha=alpha)
        test_curves.append(curve)
        chosen.append(tau_star)
    return np.array(test_curves), np.array(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--grid_size", type=int, default=101)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length", type=int, default=None,
                    help="filter to prompts with output_len >= length and "
                         "truncate to length (matches improving_KL get_pivotals)."
                         " Default: use min length across all valid prompts.")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    rows = data["rows"]
    Ls = sorted({r["lookahead"] for r in rows
                 if r["decoder"] == "mc_uwm_pseudo_r"})
    out = {"alpha": args.alpha, "n_trials": args.n_trials,
           "grid_size": args.grid_size, "seed": args.seed, "by_L": {}}

    for L in Ls:
        ps_all = [r for r in rows
                  if r["decoder"] == "mc_uwm_pseudo_r" and r["lookahead"] == L
                  and len(r.get("dual_Us_pk", [])) > 0]
        if not ps_all:
            continue
        # Filter prompts whose output_len >= --length, then truncate to length.
        # Mirrors improving_KL.get_pivotals to avoid pad-induced bias.
        target = args.length or min(len(r["dual_Us_pk"]) for r in ps_all)
        ps = [r for r in ps_all if len(r["dual_Us_pk"]) >= target]
        n_drop = len(ps_all) - len(ps)
        if not ps:
            print(f"L={L}: no prompts with output_len >= {target}, skip")
            continue
        T = int(target)
        Y_pk = np.zeros((len(ps), T))
        Y_mc = np.zeros((len(ps), T))
        R = np.zeros((len(ps), T))
        for i, r in enumerate(ps):
            Y_pk[i] = np.asarray(r["dual_Us_pk"][:T], dtype=np.float64)
            Y_mc[i] = np.asarray(r["dual_Us_mc"][:T], dtype=np.float64)
            R[i] = np.asarray(r["dual_r"][:T], dtype=np.float64)
        max_T = T
        curves, taus = run_repeated_threshold(
            Y_pk, Y_mc, R, n_trials=args.n_trials, alpha=args.alpha,
            grid_size=args.grid_size, seed=args.seed,
        )
        out["by_L"][str(L)] = {
            "n_prompts": int(len(ps)),
            "n_dropped_short": int(n_drop),
            "max_T": int(max_T),
            "test_tpr_mean": curves.mean(axis=0).tolist(),
            "test_tpr_std": curves.std(axis=0).tolist(),
            "chosen_taus": taus.tolist(),
            "tau_mean": float(taus.mean()),
            "tau_std": float(taus.std()),
            "final_tpr_mean": float(curves.mean(axis=0)[-1]),
            "final_tpr_std": float(curves.std(axis=0)[-1]),
        }
        print(f"L={L}: n={len(ps)} (dropped {n_drop} short) T={max_T} "
              f"tau_mean={taus.mean():.3f}+/-{taus.std():.3f}  "
              f"TPR_final={curves.mean(axis=0)[-1]:.3f}+/-{curves.std(axis=0)[-1]:.3f}")

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
