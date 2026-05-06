"""Composite figure (NeurIPS-ready):

  Top   : TPR vs T_eval, all decoders compared on Qwen2.5-7B + CNN/DailyMail
          (data from outputs/ablation_n500_cnn/tpr_vs_T_2x2_data.json,
          cells.qwen_cnn).

  Bottom: 1x4 row of per-decoder drafter-substitution panels
          (PFR, MWS, MSE, Algo 1), each with the four drafter curves
          D0/D1/D2/D3, recomputed from u_per_token in the four exp2 JSONs.
"""
from __future__ import annotations
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.special import gammaincc

ROOT = Path("outputs/ablation_n500_cnn")
OUT_DIR = ROOT / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NeurIPS rcParams (compact, sans-serif, embed-friendly)
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":           9,
    "axes.titlesize":      9.5,
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
    "legend.fontsize":     7.8,
    "legend.frameon":      False,
    "lines.linewidth":     1.5,
    "lines.markersize":    3.5,
    "grid.linewidth":      0.5,
    "grid.linestyle":      ":",
    "grid.alpha":          0.5,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ---- top panel colors / styles ----
TOP_STYLE = {
    "No watermark (H_0)":  dict(color="#444444", ls=(0,(2,2)), lw=1.1, marker="",  zorder=1),
    "Basic UWM":           dict(color="#0072B2", ls="-",       lw=1.4, marker="o", zorder=2),
    "MC-UWM (speed)":      dict(color="#CC79A7", ls=(0,(3,1.5)), lw=1.4, marker="v", zorder=2),
    "MC-UWM (strength)":   dict(color="#E69F00", ls="-",       lw=1.4, marker="s", zorder=2),
    "MC-UWM (pseudo-r)":   dict(color="#56B4E9", ls=(0,(1,1.2)), lw=1.4, marker="^", zorder=2),
    "PFR (ours)":          dict(color="#D55E00", ls="-",       lw=2.2, marker="D", zorder=4),
    "MPFR (ours)":         dict(color="#009E73", ls="-",       lw=2.2, marker="P", zorder=4),
}
TOP_ORDER = [
    "PFR (ours)", "MPFR (ours)",
    "Basic UWM", "MC-UWM (strength)",
    "MC-UWM (pseudo-r)", "MC-UWM (speed)",
    "No watermark (H_0)",
]

# ---- bottom panel: drafter conditions ----
DRAFTERS = [
    ("D0",  "default",            "#0072B2", "o", "-"),
    ("D1",  "model swap (1.5B)",  "#D55E00", "s", "--"),
    ("D2",  "T = 0.5 (sharp)",    "#009E73", "^", "-."),
    ("D3",  "T = 1.5 (diffuse)",  "#E69F00", "D", ":"),
]

DECODER_PANELS = [
    ("pfr",              "PFR  (ours)",            True),
    ("mc_uwm_strength",  "MWS  (Hu & Huang)",      False),
    ("mc_uwm_speed",     "MSE  (Hu & Huang)",      False),
    ("mc_uwm_pseudo_r",  "mse_pseudo  (He et al.)", False),
]

T_GRID_BOT = [16, 24, 32, 48, 64, 80, 96, 112, 128]


def gamma_tail_log_p(n, score):
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
    return float(np.sum(-np.log1p(-u_clip))), n_eff


def tpr_gamma_tail(rows, T, fpr=0.01):
    log_p_thresh = math.log(fpr)
    detections = []
    for r in rows:
        score, n_eff = aaronson_score_at_T(
            r.get("u_per_token"), r.get("skipped_per_token"), T,
        )
        if not math.isfinite(score) or n_eff <= 0:
            detections.append(0); continue
        log_p = gamma_tail_log_p(n_eff, score)
        detections.append(int(log_p <= log_p_thresh))
    return float(np.mean(detections)) if detections else float("nan")


def load_drafter_rows():
    files = {
        "D0": "exp2_drafter_inv_qwen_cnn_n500_D0.json",
        "D1": "exp2_drafter_inv_qwen_cnn_n500_D1.json",
        "D2": "exp2_drafter_inv_qwen_cnn_n500_D2.json",
        "D3": "exp2_drafter_inv_qwen_cnn_n500_D3.json",
    }
    out = {}
    for k, fn in files.items():
        d = json.load(open(ROOT / fn))
        by_dec = defaultdict(list)
        for r in d["rows"]:
            by_dec[r["decoder"]].append(r)
        out[k] = by_dec
    return out


def spread_at_T(rows_by_drafter, dec, T):
    vals = [tpr_gamma_tail(rows_by_drafter[k].get(dec, []), T)
            for k in ["D0","D1","D2","D3"]]
    finite = [v for v in vals if not math.isnan(v)]
    return (max(finite)-min(finite)) if len(finite)>=2 else float("nan")


# =============================================================================
# Build figure
# =============================================================================
def main():
    # --- Top panel data ---
    top_data = json.load(open(ROOT / "tpr_vs_T_2x2_data.json"))
    qwen_cnn = top_data["cells"]["qwen_cnn"]["curves"]
    T_top = top_data["T_grid"]

    # --- Bottom panel data ---
    rows_by_drafter = load_drafter_rows()

    # --- Layout ---
    # (a) on the LEFT (full height), (b) 2x2 panels on the RIGHT
    fig = plt.figure(figsize=(8.6, 3.9))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.32, wspace=0.18,
        left=0.06, right=0.985, top=0.97, bottom=0.18,
    )
    ax_top = fig.add_subplot(gs[:, 0])
    axes_bot = [
        fig.add_subplot(gs[0, 1]),  # PFR
        fig.add_subplot(gs[0, 2]),  # MWS
        fig.add_subplot(gs[1, 1]),  # MSE
        fig.add_subplot(gs[1, 2]),  # mse_pseudo
    ]

    # ===== TOP: TPR comparison =====
    legend_top = []
    for name in TOP_ORDER:
        if name not in qwen_cnn:
            continue
        ys = qwen_cnn[name]
        st = TOP_STYLE[name]
        ln, = ax_top.plot(
            T_top, ys,
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=4.0,
            markerfacecolor=st["color"], markeredgecolor="white",
            markeredgewidth=0.4, zorder=st["zorder"],
        )
        # Legend entry — pretty names
        pretty = name.replace("MC-UWM (speed)", "MSE") \
                     .replace("MC-UWM (strength)", "MWS") \
                     .replace("MC-UWM (pseudo-r)", "mse_pseudo") \
                     .replace("Basic UWM", "Basic UWM") \
                     .replace("No watermark (H_0)", "no watermark $H_0$")
        legend_top.append(Line2D([0],[0],
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=4.5,
            markerfacecolor=st["color"], markeredgecolor="white",
            markeredgewidth=0.4, label=pretty,
        ))

    ax_top.set_xlim(min(T_top)-3, max(T_top)+3)
    ax_top.set_ylim(-0.03, 1.03)
    ax_top.set_xticks([8, 16, 32, 48, 64, 96, 128])
    ax_top.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_top.set_xlabel(r"$T_{\mathrm{eval}}$ (# generated tokens)")
    ax_top.set_ylabel("TPR @ FPR = 1%")
    ax_top.grid(True)
    ax_top.legend(
        handles=legend_top,
        loc="upper left", ncol=2,
        handlelength=2.2, handletextpad=0.5, columnspacing=1.0,
        labelspacing=0.30, borderpad=0.4,
    )

    # ===== BOTTOM: 4 panels per decoder, 4 drafter curves each =====
    legend_bot = []
    for ax, (dec, title, is_ours) in zip(axes_bot, DECODER_PANELS):
        for k, label, color, marker, ls in DRAFTERS:
            rows = rows_by_drafter[k].get(dec, [])
            ys = [tpr_gamma_tail(rows, T) for T in T_GRID_BOT]
            ax.plot(
                T_GRID_BOT, ys,
                color=color, linestyle=ls, marker=marker,
                linewidth=1.4, markersize=3.2,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.4,
            )
            if ax is axes_bot[0]:
                legend_bot.append(Line2D([0],[0],
                    color=color, linestyle=ls, marker=marker,
                    linewidth=1.4, markersize=4,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.4,
                    label=f"$\\mathrm{{{k}}}$  {label}",
                ))
        s64  = spread_at_T(rows_by_drafter, dec, 64)
        s128 = spread_at_T(rows_by_drafter, dec, 128)
        ax.text(
            0.97, 0.05,
            f"$\\Delta_{{T=64}}\\!=\\!{s64*100:.0f}$ pp\n"
            f"$\\Delta_{{T=128}}\\!=\\!{s128*100:.0f}$ pp",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.0,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="0.7", linewidth=0.4, alpha=0.92),
        )
        ax.set_title(title, fontweight="bold" if is_ours else "normal",
                     pad=3, color="#0a3d62" if is_ours else "black",
                     fontsize=9)
        ax.grid(True)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(8, 136)
        ax.set_xticks([16, 64, 128])
        ax.set_yticks([0, 0.5, 1.0])

    # 2x2 layout: only left-column shows ylabel, only bottom-row shows xlabel.
    # axes_bot indices: 0=PFR (TL), 1=MWS (TR), 2=MSE (BL), 3=mse_pseudo (BR)
    axes_bot[0].set_ylabel("TPR @ FPR = 1%")
    axes_bot[2].set_ylabel("TPR @ FPR = 1%")
    for ax in (axes_bot[1], axes_bot[3]):
        ax.set_yticklabels([])
    for ax in (axes_bot[0], axes_bot[1]):
        ax.set_xticklabels([])
    for ax in (axes_bot[2], axes_bot[3]):
        ax.set_xlabel(r"$T_{\mathrm{eval}}$", labelpad=1)

    # Shared drafter legend at the bottom, aligned to the RIGHT (b) sub-grid.
    # The right grid spans approximately x in [0.475, 0.985] of the figure,
    # so use a bbox covering that x-range and let "lower center" center within it.
    fig.legend(
        handles=legend_bot,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.475, 0.0, 0.510, 0.05),
        handlelength=2.4, handletextpad=0.5, columnspacing=1.2,
        frameon=False, fontsize=7.8,
    )

    out_png = OUT_DIR / "drafter_invariance_composite.png"
    out_pdf = OUT_DIR / "drafter_invariance_composite.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
