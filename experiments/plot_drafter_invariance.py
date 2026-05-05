"""TPR-vs-T_eval drafter-invariance plot for ablation Exp 2 (NeurIPS-ready).

Two figures, both PDF + PNG:
  fig1: 2x2 grid, TPR vs T_eval per drafter condition, one panel per decoder.
        Drafter-invariance manifests visually as the 4 lines stacking.
  fig2: Spread (max - min) bar chart at T_eval=64 and T_eval=128, ranking the
        schemes by drafter-substitution drift.

Style: sans-serif Helvetica-equivalent, Okabe-Ito colorblind-safe palette,
NeurIPS body-width (5.5 in single-column figure / 6.5 in for 2x2).
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import gammaincc

ROOT = Path("outputs/ablation_smoke")
OUT_DIR = Path("outputs/ablation_smoke/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# NeurIPS-style rcParams
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":           9,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "legend.fontsize":     8,
    "legend.frameon":      False,
    "lines.linewidth":     1.6,
    "lines.markersize":    4.0,
    "grid.linewidth":      0.5,
    "grid.linestyle":      ":",
    "grid.alpha":          0.5,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# Okabe-Ito colorblind-safe palette
COLOR_D0 = "#0072B2"   # blue        — default drafter
COLOR_D1 = "#D55E00"   # vermillion  — model swap (axis 1)
COLOR_D2 = "#009E73"   # bluish-green — T=0.5 sharper (axis 2 lo)
COLOR_D3 = "#E69F00"   # orange      — T=1.5 diffuse (axis 2 hi)

DRAFTERS = [
    ("D0",  "default",          COLOR_D0, "o", "-"),
    ("D1",  "model swap (1.5B)", COLOR_D1, "s", "--"),
    ("D2",  "T = 0.5 (sharp)",   COLOR_D2, "^", "-."),
    ("D3",  "T = 1.5 (diffuse)", COLOR_D3, "D", ":"),
]

DECODER_PANELS = [
    ("pfr",              "PFR  (ours)",                  True),
    ("mc_uwm_strength",  "MWS  (Hu & Huang, 2024)",      False),
    ("mc_uwm_speed",     "MSE  (Hu & Huang, 2024)",      False),
    ("mc_uwm_pseudo_r",  "Algo 1  (He et al., 2026)",    False),
]

T_GRID = [16, 24, 32, 48, 64, 80, 96, 112, 128]
T_HIGHLIGHT = [64, 128]


# ----------------------------------------------------------------------------
# Detection score (Aaronson Gamma-tail, FPR=1% via gamma-tail p-value)
# ----------------------------------------------------------------------------
def gamma_tail_log_p(n: int, score: float) -> float:
    if n <= 0 or score <= 0 or not math.isfinite(score):
        return 0.0
    p = gammaincc(n, score)
    return math.log(p) if p > 0 else float("-inf")


def aaronson_score_at_T(u_per_token, skipped_per_token, T):
    if u_per_token is None:
        return float("nan"), 0
    u = np.asarray(u_per_token[:T], dtype=float)
    if skipped_per_token is not None:
        sk = np.asarray(skipped_per_token[:T], dtype=bool)
        u = u[~sk]
    n_eff = int(u.size)
    if n_eff <= 0:
        return float("nan"), 0
    u_clip = np.clip(u, 0.0, 1.0 - 1e-10)
    score = float(np.sum(-np.log1p(-u_clip)))
    return score, n_eff


def tpr_gamma_tail(rows, T, fpr=0.01):
    log_p_thresh = math.log(fpr)
    detections = []
    for r in rows:
        score, n_eff = aaronson_score_at_T(
            r.get("u_per_token"), r.get("skipped_per_token"), T,
        )
        if not math.isfinite(score) or n_eff <= 0:
            detections.append(0)
            continue
        log_p = gamma_tail_log_p(n_eff, score)
        detections.append(int(log_p <= log_p_thresh))
    return float(np.mean(detections)) if detections else float("nan")


def load_rows():
    files = {
        "D0": "exp2_drafter_inv_qwen_cnn_n100_D0.json",
        "D1": "exp2_drafter_inv_qwen_cnn_n100_D1.json",
        "D2": "exp2_drafter_inv_qwen_cnn_n100_D2.json",
        "D3": "exp2_drafter_inv_qwen_cnn_n100_D3.json",
    }
    rows_by_drafter = {}
    for k, fn in files.items():
        d = json.load(open(ROOT / fn))
        by_dec = defaultdict(list)
        for r in d["rows"]:
            by_dec[r["decoder"]].append(r)
        rows_by_drafter[k] = by_dec
    return rows_by_drafter


def spread_at_T(rows_by_drafter, dec, T):
    vals = [tpr_gamma_tail(rows_by_drafter[k].get(dec, []), T)
            for k in ["D0", "D1", "D2", "D3"]]
    finite = [v for v in vals if not math.isnan(v)]
    return (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")


# ----------------------------------------------------------------------------
# Figure 1: 2x2 TPR-vs-T_eval per decoder
# ----------------------------------------------------------------------------
def make_main_figure(rows_by_drafter):
    fig, axes = plt.subplots(
        2, 2, figsize=(6.5, 4.4),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.30, wspace=0.10),
    )

    legend_handles = []

    for ax, (dec, title, is_ours) in zip(axes.flat, DECODER_PANELS):
        for k, label, color, marker, ls in DRAFTERS:
            rows = rows_by_drafter[k].get(dec, [])
            ys = [tpr_gamma_tail(rows, T) for T in T_GRID]
            ln, = ax.plot(
                T_GRID, ys,
                color=color, linestyle=ls, marker=marker,
                linewidth=1.6, markersize=4.2,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.5,
            )
            if ax is axes[0, 0]:
                legend_handles.append(
                    Line2D([0], [0], color=color, marker=marker, linestyle=ls,
                           linewidth=1.6, markersize=5,
                           markerfacecolor=color, markeredgecolor="white",
                           markeredgewidth=0.5,
                           label=f"$\\mathrm{{{k}}}$  {label}"),
                )

        # Spread annotation: small unobtrusive box, lower-right
        s64  = spread_at_T(rows_by_drafter, dec, 64)
        s128 = spread_at_T(rows_by_drafter, dec, 128)
        ax.text(
            0.97, 0.04,
            f"$\\Delta_{{T=64}} = {s64*100:.0f}\\,\\mathrm{{pp}}$\n"
            f"$\\Delta_{{T=128}} = {s128*100:.0f}\\,\\mathrm{{pp}}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="0.7", linewidth=0.4, alpha=0.92),
        )
        ax.set_title(title, fontweight="bold" if is_ours else "normal",
                     pad=4, color="#0a3d62" if is_ours else "black")
        ax.grid(True)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(8, 136)
        ax.set_xticks([16, 32, 64, 96, 128])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(width=0.6)

    # Y-axis label only on left; X-axis label only on bottom
    for ax in axes[1, :]:
        ax.set_xlabel(r"$T_{\mathrm{eval}}$ (# generated tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("TPR @ FPR = 1%")

    # Single combined legend below
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        handlelength=2.6, handletextpad=0.6, columnspacing=1.4,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "drafter_invariance_2x2.pdf")
    fig.savefig(OUT_DIR / "drafter_invariance_2x2.png")
    print(f"wrote {OUT_DIR / 'drafter_invariance_2x2.pdf'}")
    print(f"wrote {OUT_DIR / 'drafter_invariance_2x2.png'}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: spread bar chart
# ----------------------------------------------------------------------------
def make_spread_figure(rows_by_drafter):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))

    decoders = [
        ("pfr",              "PFR\n(ours)",        True),
        ("mc_uwm_strength",  "MWS",                False),
        ("basic_uwm",        "basic_uwm\n(no drafter)", False),
        ("mc_uwm_pseudo_r",  "Algo 1",             False),
        ("mc_uwm_speed",     "MSE",                False),
    ]

    s64  = [spread_at_T(rows_by_drafter, dec, 64)  * 100 for dec, _, _ in decoders]
    s128 = [spread_at_T(rows_by_drafter, dec, 128) * 100 for dec, _, _ in decoders]

    x = np.arange(len(decoders))
    w = 0.36
    c64  = "#4C72B0"
    c128 = "#DD8452"

    b1 = ax.bar(x - w/2, s64,  w, color=c64,  edgecolor="white",
                linewidth=0.6, label=r"$T_{\mathrm{eval}} = 64$")
    b2 = ax.bar(x + w/2, s128, w, color=c128, edgecolor="white",
                linewidth=0.6, label=r"$T_{\mathrm{eval}} = 128$")

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab, _ in decoders])
    for tick, (_, _, is_ours) in zip(ax.get_xticklabels(), decoders):
        if is_ours:
            tick.set_fontweight("bold")
            tick.set_color("#0a3d62")
    ax.set_ylabel("TPR spread $\\max_i - \\min_i$  (pp)")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_ylim(0, max(max(s64), max(s128)) * 1.18 + 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "drafter_invariance_spread.pdf")
    fig.savefig(OUT_DIR / "drafter_invariance_spread.png")
    print(f"wrote {OUT_DIR / 'drafter_invariance_spread.pdf'}")
    print(f"wrote {OUT_DIR / 'drafter_invariance_spread.png'}")
    plt.close(fig)


def main():
    rows_by_drafter = load_rows()
    make_main_figure(rows_by_drafter)
    make_spread_figure(rows_by_drafter)


if __name__ == "__main__":
    main()
