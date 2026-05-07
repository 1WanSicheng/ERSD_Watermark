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
    "font.size":           8,
    "axes.titlesize":      8,
    "axes.labelsize":      8,
    "axes.linewidth":      0.7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "legend.fontsize":     6.5,
    "legend.frameon":      False,
    "lines.linewidth":     1.4,
    "lines.markersize":    3.0,
    "grid.linewidth":      0.4,
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
    ("D0",  "default",       "#0072B2", "o", "-"),
    ("D1",  "model swap",    "#D55E00", "s", "--"),
    ("D2",  "$T\\!=\\!0.5$", "#009E73", "^", "-."),
    ("D3",  "$T\\!=\\!1.5$", "#E69F00", "D", ":"),
]

DECODER_PANELS = [
    ("PFR (ours)",  "PFR",        True),
    ("MWS",         "MWS",        False),
    ("MSE",         "MSE",        False),
    ("mse_pseudo",  "MSE-Pseudo", False),
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

    # Layout (per Figure 2 revision guide):
    # - figsize 7.2 x 3.2 (NeurIPS double-column main-text figure).
    # - Left:right region width ratio 1.25:1.0 (left ~55%, right ~45%);
    #   right region is a 2x2, so widths sum to [2.5, 1.0, 1.0].
    # - Bottom margin reserved for shared right-side legend (D0..D3).
    # - Top margin reserved for two panel titles "(a)" / "(b)".
    fig = plt.figure(figsize=(7.4, 3.2))
    # Outer split: left (a) | right (b) — explicit wider gap so right-side
    # y-tick labels do not crowd the (a) panel.
    gs_outer = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.25, 1.0],
        wspace=0.32,
        left=0.075, right=0.985, top=0.88, bottom=0.20,
    )
    ax_top = fig.add_subplot(gs_outer[0, 0])
    # Inner 2x2 grid for (b)
    gs_right = gs_outer[0, 1].subgridspec(
        2, 2, hspace=0.30, wspace=0.20,
    )
    axes_bot = [
        fig.add_subplot(gs_right[0, 0]),  # PFR
        fig.add_subplot(gs_right[0, 1]),  # MWS
        fig.add_subplot(gs_right[1, 0]),  # MSE
        fig.add_subplot(gs_right[1, 1]),  # MSE-Pseudo
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
                      .replace("MC-UWM (pseudo-r)", "MSE-Pseudo")
                      .replace("Basic UWM", "Basic-UWM")
                      .replace("No watermark (H_0)", "No watermark ($H_0$)"))
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
        handlelength=1.8, handletextpad=0.4, columnspacing=0.8,
        labelspacing=0.22, borderpad=0.3,
        fontsize=6.5,
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
                    linewidth=1.3, markersize=3.5,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.4,
                    label=f"{k} {label}",
                ))
        ax.set_title(title, fontweight="bold" if is_ours else "normal",
                     pad=2, color="#0a3d62" if is_ours else "black",
                     fontsize=8)
        ax.grid(True)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(8, 136)
        ax.set_xticks([16, 64, 128])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

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
        bbox_to_anchor=(0.645, 0.005, 0.340, 0.05),
        handlelength=2.0, handletextpad=0.4, columnspacing=1.0,
        frameon=False, fontsize=6.5,
    )

    # Panel (a) / (b) titles, placed via figure-level text so (b) can span
    # the right 2x2 sub-grid (above the PFR / MWS panels).
    fig.text(
        0.075, 0.92,
        "(a) Detection on the default drafter",
        ha="left", va="bottom", fontsize=9, fontweight="bold",
    )
    fig.text(
        0.645, 0.92,
        "(b) Robustness to drafter substitution",
        ha="left", va="bottom", fontsize=9, fontweight="bold",
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
