"""4-panel TPR-vs-T_eval comparison plot, one panel per (model, dataset) cell.

Reads pre-computed curves (produced by experiments/aggregate_tpr_curves.py
or hand-edited equivalents) from a JSON whose path is configurable via
--data-file (default: data/0506 ablation_n500_qwen_cnn/tpr_vs_T_2x2_data.json).

The JSON's `cells` block holds up to four cells (qwen_cnn, qwen_eli5,
vicuna_cnn, vicuna_eli5); each present cell is rendered as one subplot.
Missing cells get an empty / hidden panel.  The same 7 schemes are
overlaid in each subplot using the colour/marker scheme as in
plot_drafter_invariance_v2.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_DATA = Path("data/0506 ablation_n500_qwen_cnn") / "tpr_vs_T_2x2_data.json"

# NeurIPS rcParams
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

STYLE = {
    "No watermark (H_0)":  dict(color="#444444", ls=(0,(2,2)),   lw=1.1, marker="",  zorder=1),
    "Basic UWM":           dict(color="#0072B2", ls="-",         lw=1.4, marker="o", zorder=2),
    "MC-UWM (speed)":      dict(color="#CC79A7", ls=(0,(3,1.5)), lw=1.4, marker="v", zorder=2),
    "MC-UWM (strength)":   dict(color="#E69F00", ls="-",         lw=1.4, marker="s", zorder=2),
    "MC-UWM (pseudo-r)":   dict(color="#56B4E9", ls=(0,(1,1.2)), lw=1.4, marker="^", zorder=2),
    "PFR (ours)":          dict(color="#D55E00", ls="-",         lw=2.2, marker="D", zorder=4),
    "MPFR (ours)":         dict(color="#009E73", ls="-",         lw=2.2, marker="P", zorder=4),
}

ORDER = [
    "PFR (ours)", "MPFR (ours)",
    "Basic UWM", "MC-UWM (strength)",
    "MC-UWM (pseudo-r)", "MC-UWM (speed)",
    "No watermark (H_0)",
]

CELLS = [  # (cell_key, panel_label_topleft)
    ("qwen_cnn",    "Qwen2.5-7B  +  CNN/DailyMail"),
    ("qwen_eli5",   "Qwen2.5-7B  +  ELi5"),
    ("vicuna_cnn",  "Vicuna-7B  +  CNN/DailyMail"),
    ("vicuna_eli5", "Vicuna-7B  +  ELi5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", type=Path, default=DEFAULT_DATA,
                    help="Aggregated TPR-curves JSON (cf. aggregate_tpr_curves.py).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Folder for the .png/.pdf outputs.  Default: same "
                         "folder as --data-file.")
    args = ap.parse_args()
    data = json.load(open(args.data_file))
    T_grid = data["T_grid"]
    cells = data["cells"]
    out_dir = args.out_dir if args.out_dir is not None else args.data_file.parent

    fig, axes = plt.subplots(
        2, 2, figsize=(8.4, 5.6),
        sharex=True, sharey=True,
        gridspec_kw=dict(hspace=0.20, wspace=0.10,
                         left=0.075, right=0.99, top=0.99, bottom=0.16),
    )

    legend_handles = []
    for ax, (cell_key, panel_label) in zip(axes.flat, CELLS):
        if cell_key not in cells:
            ax.set_visible(False)
            continue
        curves = cells[cell_key]["curves"]
        for name in ORDER:
            if name not in curves:
                continue
            ys = curves[name]
            st = STYLE[name]
            ax.plot(
                T_grid, ys,
                color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                marker=st["marker"], markersize=4.0,
                markerfacecolor=st["color"], markeredgecolor="white",
                markeredgewidth=0.4, zorder=st["zorder"],
            )
            if ax is axes[0, 0]:
                pretty = (name.replace("MC-UWM (speed)", "MSE")
                              .replace("MC-UWM (strength)", "MWS")
                              .replace("MC-UWM (pseudo-r)", "mse_pseudo")
                              .replace("No watermark (H_0)", "no watermark $H_0$"))
                legend_handles.append(Line2D([0],[0],
                    color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                    marker=st["marker"], markersize=4.5,
                    markerfacecolor=st["color"], markeredgecolor="white",
                    markeredgewidth=0.4, label=pretty,
                ))
        # cell label inside panel (top-left)
        ax.text(
            0.025, 0.965, panel_label,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color="#1a1a1a", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor="0.7", linewidth=0.4, alpha=0.92),
        )
        ax.grid(True)
        ax.set_xlim(min(T_grid)-3, max(T_grid)+3)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([8, 16, 32, 48, 64, 96, 128])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    for ax in axes[1, :]:
        ax.set_xlabel(r"$T_{\mathrm{eval}}$ (# generated tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("TPR @ FPR = 1%")

    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=7,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.0, handletextpad=0.4, columnspacing=1.3,
        frameon=False, fontsize=7.5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tpr_comparison_4cells.png"
    out_pdf = out_dir / "tpr_comparison_4cells.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    main()
