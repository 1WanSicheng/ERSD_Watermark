"""2x4 main-text composite figure (Figure 2).

Row (a): robustness to drafter substitution — one panel per decoder
         (PFR, MWS, MSE, MSE-Pseudo), four drafter conditions D0-D3 each.
         Data: data/0506 ablation_n500_qwen_cnn/tpr_vs_T_per_drafter.json
         (Qwen2.5-7B x CNN/DailyMail, n=500).

Row (b): detection TPR@1%FPR vs T_eval across the four (target, dataset)
         cells, seven schemes each (previously the appendix 2x2 figure).
         Data: data/0506 single_draft_n1000_efpr_4cell/tpr_vs_T_2x2_data.json
         (n=1000 per cell).

Both inputs are pre-aggregated curves; no raw experiment JSONs are needed.
Output: data/figures/fig_2x4_detection_invariance.{pdf,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ROW_A_DATA = ROOT / "data/0506 ablation_n500_qwen_cnn/tpr_vs_T_per_drafter.json"
ROW_B_DATA = ROOT / "data/0506 single_draft_n1000_efpr_4cell/tpr_vs_T_2x2_data.json"
OUT_DIR = ROOT / "data/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "legend.fontsize":     7,
    "legend.frameon":      False,
    "lines.linewidth":     1.3,
    "lines.markersize":    2.8,
    "grid.linewidth":      0.4,
    "grid.linestyle":      ":",
    "grid.alpha":          0.5,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# Scheme styles keyed by JSON curve names; display labels use paper naming.
SCHEME_STYLE = {
    "No watermark (H_0)":  dict(color="#444444", ls=(0, (2, 2)),   lw=1.0, marker=""),
    "Basic UWM":           dict(color="#0072B2", ls="-",           lw=1.3, marker="o"),
    "MC-UWM (speed)":      dict(color="#CC79A7", ls=(0, (3, 1.5)), lw=1.3, marker="v"),
    "MC-UWM (strength)":   dict(color="#E69F00", ls="-",           lw=1.3, marker="s"),
    "MC-UWM (pseudo-r)":   dict(color="#56B4E9", ls=(0, (1, 1.2)), lw=1.3, marker="^"),
    "PFR (ours)":          dict(color="#D55E00", ls="-",           lw=2.0, marker="D"),
    "MPFR (ours)":         dict(color="#009E73", ls="-",           lw=2.0, marker="P"),
}
SCHEME_LABEL = {
    "PFR (ours)":         "PFR (ours)",
    "MPFR (ours)":        "MPFR (ours)",
    "Basic UWM":          "Basic-UWM",
    "MC-UWM (strength)":  "MWS",
    "MC-UWM (speed)":     "MSE",
    "MC-UWM (pseudo-r)":  "MSE-Pseudo",
    "No watermark (H_0)": "No watermark ($H_0$)",
}
SCHEME_ORDER = [
    "PFR (ours)", "MPFR (ours)", "Basic UWM", "MC-UWM (strength)",
    "MC-UWM (pseudo-r)", "MC-UWM (speed)", "No watermark (H_0)",
]
SCHEME_Z = {"PFR (ours)": 4, "MPFR (ours)": 4, "No watermark (H_0)": 1}

DRAFTERS = [
    ("D0", "D0 default",        "#0072B2", "o", "-"),
    ("D1", "D1 model swap",     "#D55E00", "s", "--"),
    ("D2", "D2 $T\\!=\\!0.5$",  "#009E73", "^", "-."),
    ("D3", "D3 $T\\!=\\!1.5$",  "#E69F00", "D", ":"),
]
DECODER_PANELS = [
    ("PFR (ours)", "PFR"),
    ("MWS",        "MWS"),
    ("MSE",        "MSE"),
    ("mse_pseudo", "MSE-Pseudo"),
]

CELL_PANELS = [
    ("qwen_cnn",    "Qwen2.5-7B + CNN/DailyMail"),
    ("qwen_eli5",   "Qwen2.5-7B + ELI5"),
    ("vicuna_cnn",  "Vicuna-7B + CNN/DailyMail"),
    ("vicuna_eli5", "Vicuna-7B + ELI5"),
]

XTICKS = [8, 32, 64, 96, 128]


def _decorate(ax, first_in_row: bool):
    ax.set_xlim(4, 132)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(XTICKS)
    ax.grid(True, axis="both")
    if first_in_row:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("TPR @ FPR = 1%")
    else:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])


def main() -> None:
    a = json.load(open(ROW_A_DATA))
    b = json.load(open(ROW_B_DATA))
    Ta, Tb = a["T_grid"], b["T_grid"]

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.1), sharex=False)

    # ---- Row (a): per-decoder drafter-substitution panels -----------------
    for j, (dec_key, dec_title) in enumerate(DECODER_PANELS):
        ax = axes[0][j]
        for dk, dlabel, color, marker, ls in DRAFTERS:
            curve = a["drafters"][dk]["curves"].get(dec_key)
            if curve is None:
                continue
            n = min(len(Ta), len(curve))
            ax.plot(Ta[:n], curve[:n], color=color, marker=marker, ls=ls,
                    label=dlabel, markerfacecolor="none", markeredgewidth=0.9)
        ax.set_title(dec_title)
        _decorate(ax, first_in_row=(j == 0))
        ax.tick_params(labelbottom=True)

    # ---- Row (b): per-cell scheme comparison panels ----------------------
    for j, (cell_key, cell_title) in enumerate(CELL_PANELS):
        ax = axes[1][j]
        curves = b["cells"][cell_key]["curves"]
        for name in SCHEME_ORDER:
            curve = curves.get(name)
            if curve is None:
                continue
            st = SCHEME_STYLE[name]
            n = min(len(Tb), len(curve))
            ax.plot(Tb[:n], curve[:n], zorder=SCHEME_Z.get(name, 2), **st)
        ax.set_title(cell_title)
        _decorate(ax, first_in_row=(j == 0))
        ax.set_xlabel(r"$T_{\mathrm{eval}}$ (# generated tokens)")

    # ---- Row labels and legends ------------------------------------------
    fig.text(0.005, 0.985, "(a) Robustness to drafter substitution "
             "(Qwen2.5-7B + CNN/DailyMail)",
             fontsize=8.5, fontweight="bold", va="top", ha="left")
    fig.text(0.005, 0.520, "(b) Detection across models and datasets",
             fontsize=8.5, fontweight="bold", va="top", ha="left")

    drafter_handles = [
        Line2D([], [], color=c, marker=m, ls=ls, label=lbl,
               markerfacecolor="none", markeredgewidth=0.9)
        for _, lbl, c, m, ls in DRAFTERS
    ]
    scheme_handles = [
        Line2D([], [], label=SCHEME_LABEL[n], **SCHEME_STYLE[n])
        for n in SCHEME_ORDER
    ]
    # Row (a) legend sits directly under row 1; row (b) legend under row 2.
    fig.legend(handles=drafter_handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.53, 0.575), handlelength=2.2,
               columnspacing=1.4, handletextpad=0.5)
    fig.legend(handles=scheme_handles, ncol=7, loc="upper center",
               bbox_to_anchor=(0.53, 0.045), handlelength=1.8,
               columnspacing=0.8, handletextpad=0.4, fontsize=6.3)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.155,
                        hspace=0.72, wspace=0.10)

    for ext in ("pdf", "png"):
        p = OUT_DIR / f"fig_2x4_detection_invariance.{ext}"
        fig.savefig(p)
        print("wrote", p)


if __name__ == "__main__":
    main()
