"""TPR @ FPR=1% as a function of emitted-token count.

Reads a single_draft JSON with per-token uniforms stored
(`u_per_token`, `skipped_per_token`) and plots one TPR curve per decoder.

Threshold convention: theoretical Gamma(n_eff, 1) 99-pct, matching
improving_KL/my_experiment/gumbel_detect_utils.py:ars_score (`compute_gamma`
with `q=1-alpha`).  PrevN(3) repeated-context tokens are masked: the
contribution from skipped positions is set to 0 and the per-prompt
threshold uses the effective count n_eff_t = #(non-skipped, position <= t).

Usage:
    python scripts/plot_tpr_curve.py outputs/2x2/dg_compare_n50_v3.json \\
        --output outputs/2x2/tpr_vs_token_n50.png
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import gamma


DEFAULT_DECODER_ORDER = [
    "mc",
    "pfr_no_watermark",
    "basic_uwm",
    "mc_uwm_strength",
    "pfr",
    "mc_uwm_pseudo_r",
    "mc_uwm_speed",
]
DEFAULT_COLORS = {
    "mc": "tab:gray",
    "pfr_no_watermark": "tab:olive",
    "basic_uwm": "tab:blue",
    "mc_uwm_strength": "tab:cyan",
    "pfr": "tab:green",
    "mc_uwm_pseudo_r": "tab:red",
    "mc_uwm_speed": "tab:orange",
}


def collect(rows, decoder, max_T=None):
    """Collect per-prompt per-token uniforms for one decoder.

    Pads shorter prompts (those that hit EOS early) up to ``max_T`` with
    skipped=True so they contribute 0 to the cumulative score and 0 to
    n_eff at those positions.  This keeps the x-axis uniform across
    decoders without inflating their TPR.
    """
    rows_d = [r for r in rows if r.get("decoder") == decoder]
    if not rows_d:
        return None, None
    Ts = [len(r.get("u_per_token", [])) for r in rows_d]
    if not Ts or max(Ts) == 0:
        return None, None
    T = max(Ts) if max_T is None else int(max_T)
    n_prompts = len(rows_d)
    U = np.zeros((n_prompts, T), dtype=np.float64)
    S = np.ones((n_prompts, T), dtype=bool)  # default: skipped (no contribution)
    for i, r in enumerate(rows_d):
        u = r.get("u_per_token", [])
        s = r.get("skipped_per_token", [False] * len(u))
        L = min(len(u), T)
        if L > 0:
            U[i, :L] = u[:L]
            S[i, :L] = s[:L] if len(s) >= L else (
                list(s) + [True] * (L - len(s))
            )[:L]
    return U, S


def cum_score_and_n(U, S):
    """Cumulative -log(1-u) and effective n at each position."""
    U_clip = np.clip(U, 1e-10, 1 - 1e-10)
    contrib = np.where(S, 0.0, -np.log1p(-U_clip))
    cum = np.cumsum(contrib, axis=1)
    n_eff = np.cumsum((~S).astype(int), axis=1)
    return cum, n_eff


def tpr_at_each_token(cum, n_eff, alpha=0.01):
    """Per-token TPR @ theoretical Gamma(n_eff, 1) (1-alpha) threshold."""
    T = cum.shape[1]
    tpr = np.zeros(T)
    for t in range(T):
        ks = np.maximum(n_eff[:, t], 1)
        thr_per_prompt = gamma.ppf(1 - alpha, ks)
        tpr[t] = float((cum[:, t] > thr_per_prompt).mean())
    return tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="path to single_draft v3 result JSON")
    ap.add_argument("--output", "-o", required=True, help="output figure path")
    ap.add_argument("--alpha", type=float, default=0.01, help="FPR (default 1%)")
    ap.add_argument("--decoders", nargs="+", default=None,
                    help="decoders to include; defaults to all non-empty")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    rows = data["rows"]
    available = sorted({r["decoder"] for r in rows})
    decoders = args.decoders or [d for d in DEFAULT_DECODER_ORDER if d in available]

    # Find global max T across all decoders so x-axis is uniform.
    global_T = max(
        (len(r.get("u_per_token", [])) for r in rows
         if r.get("decoder") in decoders),
        default=0,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for dec in decoders:
        U, S = collect(rows, dec, max_T=global_T)
        if U is None:
            print(f"skip {dec}: no per-token data")
            continue
        cum, n_eff = cum_score_and_n(U, S)
        tpr = tpr_at_each_token(cum, n_eff, alpha=args.alpha)
        x = np.arange(1, cum.shape[1] + 1)
        ax.plot(x, tpr, label=f"{dec}  (n={U.shape[0]})",
                color=DEFAULT_COLORS.get(dec))

    ax.axhline(args.alpha, color="black", lw=0.5, ls="--",
               label=f"FPR={args.alpha:.0%} reference")
    ax.set_xlabel("emitted token count")
    ax.set_ylabel(f"TPR @ FPR={args.alpha:.0%}")
    ax.set_title("Aaronson Gamma-tail TPR (theoretical threshold)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(1, cum.shape[1])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
