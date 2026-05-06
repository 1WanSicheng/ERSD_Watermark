"""Composite figure (NeurIPS-ready):

  Left  : TPR vs T_eval, all decoders compared on default drafter D_0
          (data: data/0506 ablation_n500_qwen_cnn/tpr_vs_T_2x2_data.json,
          cells.qwen_cnn).

  Right : 2x2 panels per-decoder (PFR, MWS, MSE, mse_pseudo), each with the
          four drafter curves D0/D1/D2/D3 read from a small precomputed JSON
          (data/0506 ablation_n500_qwen_cnn/tpr_vs_T_per_drafter.json).

Both inputs are pre-aggregated TPR-vs-T_eval curves; no raw experiment JSONs
are needed at plot time.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/0506 ablation_n500_qwen_cnn")
LEFT_DATA  = DATA_DIR / "tpr_vs_T_2x2_data.json"
RIGHT_DATA = DATA_DIR / "tpr_vs_T_per_drafter.json"
OUT_DIR = DATA_DIR  # write figure next to data files
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NeurIPS rcParams
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Top (left-half) panel styles
# ---------------------------------------------------------------------------
TOP_STYLE = {
    "No watermark (H_0)":  dict(color="#444444", ls=(0,(2,2)),   lw=1.1, marker="",  zorder=1),
    "Basic UWM":           dict(color="#0072B2", ls="-",         lw=1.4, marker="o", zorder=2),
    "MC-UWM (speed)":      dict(color="#CC79A7", ls=(0,(3,1.5)), lw=1.4, marker="v", zorder=2),
    "MC-UWM (strength)":   dict(color="#E69F00", ls="-",         lw=1.4, marker="s", zorder=2),
    "MC-UWM (pseudo-r)":   dict(color="#56B4E9", ls=(0,(1,1.2)), lw=1.4, marker="^", zorder=2),
    "PFR (ours)":          dict(color="#D55E00", ls="-",         lw=2.2, marker="D", zorder=4),
    "MPFR (ours)":         dict(color="#009E73", ls="-",         lw=2.2, marker="P", zorder=4),
}
TOP_ORDER = [
    "PFR (ours)", "MPFR (ours)",
    "Basic UWM", "MC-UWM (strength)",
    "MC-UWM (pseudo-r)", "MC-UWM (speed)",
    "No watermark (H_0)",
]

# ---------------------------------------------------------------------------
# Right (bottom) panel: 4 drafter conditions
# ---------------------------------------------------------------------------
DRAFTERS = [
    ("D0",  "default",            "#0072B2", "o", "-"),
    ("D1",  "model swap (1.5B)",  "#D55E00", "s", "--"),
    ("D2",  "T = 0.5 (sharp)",    "#009E73", "^", "-."),
    ("D3",  "T = 1.5 (diffuse)",  "#E69F00", "D", ":"),
]

DECODER_PANELS = [
    ("PFR (ours)",  "PFR  (ours)",            True),
    ("MWS",         "MWS  (Hu & Huang)",      False),
    ("MSE",         "MSE  (Hu & Huang)",      False),
    ("mse_pseudo",  "mse_pseudo  (He et al.)", False),
]


def spread_across_drafters(curves_by_drafter, dec_key, T_grid, T):
    """curves_by_drafter[k]['curves'][dec_key] is a list aligned with T_grid."""
    if T not in T_grid:
        return float("nan")
    idx = T_grid.index(T)
    vals = []
    for k in ("D0", "D1", "D2", "D3"):
        c = curves_by_drafter[k]["curves"].get(dec_key)
        if c is not None and idx < len(c):
            v = c[idx]
            if v is not None and math.isfinite(v):
                vals.append(v)
    if len(vals) < 2:
        return float("nan")
    return max(vals) - min(vals)


# ===========================================================================
# Build figure
# ===========================================================================
def main():
    left  = json.load(open(LEFT_DATA))
    right = json.load(open(RIGHT_DATA))

    qwen_cnn = left["cells"]["qwen_cnn"]["curves"]
    T_left   = left["T_grid"]

    drafters = right["drafters"]
    T_right  = right["T_grid"]

    # Layout: (a) on the LEFT (full height), (b) 2x2 panels on the RIGHT
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

    # ---- LEFT panel: TPR comparison ----
    legend_top = []
    for name in TOP_ORDER:
        if name not in qwen_cnn:
            continue
        ys = qwen_cnn[name]
        st = TOP_STYLE[name]
        ax_top.plot(
            T_left, ys,
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=4.0,
            markerfacecolor=st["color"], markeredgecolor="white",
            markeredgewidth=0.4, zorder=st["zorder"],
        )
        pretty = (name.replace("MC-UWM (speed)", "MSE")
                      .replace("MC-UWM (strength)", "MWS")
                      .replace("MC-UWM (pseudo-r)", "mse_pseudo")
                      .replace("No watermark (H_0)", "no watermark $H_0$"))
        legend_top.append(Line2D([0],[0],
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
            marker=st["marker"], markersize=4.5,
            markerfacecolor=st["color"], markeredgecolor="white",
            markeredgewidth=0.4, label=pretty,
        ))

    ax_top.set_xlim(min(T_left)-3, max(T_left)+3)
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

    # ---- RIGHT panels: per-decoder drafter curves ----
    legend_bot = []
    for ax, (dec_key, title, is_ours) in zip(axes_bot, DECODER_PANELS):
        for k, label, color, marker, ls in DRAFTERS:
            ys = drafters[k]["curves"].get(dec_key)
            if ys is None:
                continue
            ax.plot(
                T_right, ys,
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
        s64  = spread_across_drafters(drafters, dec_key, T_right, 64)
        s128 = spread_across_drafters(drafters, dec_key, T_right, 128)
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
    axes_bot[0].set_ylabel("TPR @ FPR = 1%")
    axes_bot[2].set_ylabel("TPR @ FPR = 1%")
    for ax in (axes_bot[1], axes_bot[3]):
        ax.set_yticklabels([])
    for ax in (axes_bot[0], axes_bot[1]):
        ax.set_xticklabels([])
    for ax in (axes_bot[2], axes_bot[3]):
        ax.set_xlabel(r"$T_{\mathrm{eval}}$", labelpad=1)

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
