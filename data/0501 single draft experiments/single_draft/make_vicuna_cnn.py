#!/usr/bin/env python3
"""
Qwen-style plot for the Vicuna CNN single-draft experiment with complement files.

Example:
    python make_vicuna_cnn_qwen_style_with_complements.py \
      single_draft_vicuna_n1000_len128_L1-2-3-4_topk50_cnn_gpu3.json \
      single_draft_vicuna_n1000_len128_L2-4-6-8_topk50_complement_mc.json \
      single_draft_vicuna_n1000_len128_L2-4-6-8_topk50_complement_pfr_no_watermark.json \
      --out single_draft_vicuna_cnn_qwen_style_with_complements.png \
      --csv-out single_draft_vicuna_cnn_qwen_style_with_complements_pairs.csv

The --csv-out file is the pair-comparison table used by the annotations.
For a full metric summary, use make_vicuna_cnn_tradeoff_figure.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANEL_B_METHODS = ["pfr_no_watermark", "pfr"]
TRADEOFF_METHODS = ["mc", "mc_uwm_strength", "mc_uwm_speed", "pfr_no_watermark", "pfr"]
ALL_METHODS = ["pfr_no_watermark", "pfr", "mc", "mc_uwm_strength", "mc_uwm_speed"]
PAIRS = [
    ("pfr_no_watermark", "pfr"),
    ("mc", "mc_uwm_strength"),
    ("mc", "mc_uwm_speed"),
]
NO_WM_ZERO_BASELINES = {"mc", "basic"}

DISPLAY = {
    "pfr_no_watermark": "PFR no-wm",
    "pfr": "PFR (ours)",
    "mc": "VSpS",
    "mc_uwm_strength": "MWS",
    "mc_uwm_speed": "MSE",
}
COLORS = {
    "pfr_no_watermark": "sienna",
    "pfr": "firebrick",
    "mc": "teal",
    "mc_uwm_strength": "royalblue",
    "mc_uwm_speed": "mediumpurple",
}
LOOKAHEAD_MARKERS = {1: "o", 2: "s", 3: "D", 4: "X"}
FALLBACK_MARKERS = ["v", "P", "*", "<", ">", "h", "8"]
ORDER = {m: i for i, m in enumerate(["mc", "mc_uwm_strength", "mc_uwm_speed", "pfr_no_watermark", "pfr"])}

MARKER_SIZE = 105
MARKER_LW = 2.2
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 13
TITLE_SIZE = 15
LEGEND_FONT_SIZE = 14
LEGEND_TITLE_SIZE = 15
ANNOT_FONT_SIZE = 12
ARROW_LW = 2.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="JSON result files. If omitted, all *.json files in current folder are used.")
    ap.add_argument("--out", default="combined_spec_watermark_figure_v10.png")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--figwidth", type=float, default=11.4)
    ap.add_argument("--figheight", type=float, default=6.5)
    return ap.parse_args()


def resolve_inputs(inputs: list[str]) -> list[Path]:
    paths = [Path(p) for p in inputs] if inputs else sorted(Path(".").glob("*.json"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError("No JSON input files found.")
    return paths


def load_rows(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        for r in obj.get("rows", []):
            rr = dict(r)
            rr["source_file"] = path.name
            rows.append(rr)
    if not rows:
        raise RuntimeError("No rows found in input JSON files.")
    return pd.DataFrame(rows)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["decoder", "lookahead", "AATPS"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    if "ANLPPT_U" not in df.columns:
        df = df.copy()
        df["ANLPPT_U"] = np.nan

    rows = []
    for (decoder, lookahead), g in df.groupby(["decoder", "lookahead"], dropna=False):
        decoder = str(decoder)
        if decoder not in ALL_METHODS:
            continue
        a = safe_numeric(g["AATPS"]).dropna()
        u = safe_numeric(g["ANLPPT_U"]).dropna()
        if len(a) == 0:
            continue
        if len(u) > 0:
            u_mean = float(u.mean())
        elif decoder in NO_WM_ZERO_BASELINES:
            u_mean = 0.0
        else:
            continue
        rows.append({
            "decoder": decoder,
            "lookahead": int(lookahead),
            "AATPS_mean": float(a.mean()),
            "ANLPPT_U_mean": u_mean,
        })
    s = pd.DataFrame(rows)
    if s.empty:
        raise RuntimeError("No valid summarized rows found.")
    s["_order"] = s["decoder"].map(ORDER).fillna(999)
    return s.sort_values(["_order", "decoder", "lookahead"]).reset_index(drop=True)


def pair_table(summary: pd.DataFrame) -> pd.DataFrame:
    idx = {(str(r.decoder), int(r.lookahead)): r for r in summary.itertuples(index=False)}
    rows = []
    for base, wm in PAIRS:
        lookaheads = sorted(set(summary.loc[summary["decoder"].isin([base, wm]), "lookahead"].tolist()))
        for L in lookaheads:
            if (base, L) not in idx or (wm, L) not in idx:
                continue
            r0, r1 = idx[(base, L)], idx[(wm, L)]
            rows.append({
                "base": base,
                "watermarked": wm,
                "lookahead": L,
                "base_AATPS": float(r0.AATPS_mean),
                "wm_AATPS": float(r1.AATPS_mean),
                "base_ANLPPT_U": float(r0.ANLPPT_U_mean),
                "wm_ANLPPT_U": float(r1.ANLPPT_U_mean),
                "delta_AATPS": float(r1.AATPS_mean - r0.AATPS_mean),
                "delta_ANLPPT_U": float(r1.ANLPPT_U_mean - r0.ANLPPT_U_mean),
            })
    return pd.DataFrame(rows)


def marker_map(lookaheads: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    fb = 0
    for L in sorted(set(int(x) for x in lookaheads)):
        if L in LOOKAHEAD_MARKERS:
            out[L] = LOOKAHEAD_MARKERS[L]
        else:
            out[L] = FALLBACK_MARKERS[fb % len(FALLBACK_MARKERS)]
            fb += 1
    return out


def set_limits(ax, xvals: np.ndarray, yvals: np.ndarray, xpad_frac: float, ypad_frac: float, extra_x: float = 3.0) -> None:
    x_min, x_max = float(np.min(xvals)), float(np.max(xvals))
    y_min, y_max = float(np.min(yvals)), float(np.max(yvals))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - extra_x * xpad_frac * x_span, x_max + extra_x * xpad_frac * x_span)
    ax.set_ylim(y_min - ypad_frac * y_span, y_max + ypad_frac * y_span)


def style_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.2)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.title.set_fontsize(TITLE_SIZE)


def annotate_red_deltas(ax, pairs_panel: pd.DataFrame, x_range: float) -> None:
    for _, row in pairs_panel.sort_values("lookahead").iterrows():
        if row["watermarked"] != "pfr":
            continue
        x, y = float(row["wm_ANLPPT_U"]), float(row["wm_AATPS"])
        delta = float(row["delta_AATPS"])
        dx = max(0.02 * x_range, 0.0018)
        label = f"+{delta:.3f}" if delta > 0 else "0"
        ax.annotate(
            label,
            (x - 1.35 * dx, y),
            textcoords="offset points",
            xytext=(-4, -2),
            ha="right",
            va="center",
            fontsize=ANNOT_FONT_SIZE,
            color=COLORS["pfr"],
            zorder=4,
        )


def plot_pfr_effect(ax, summary: pd.DataFrame, pairs: pd.DataFrame) -> None:
    panel = summary[summary["decoder"].isin(PANEL_B_METHODS)].copy()
    pairs_panel = pairs[pairs["base"] == "pfr_no_watermark"].copy()
    mmap = marker_map(panel["lookahead"].tolist())

    for dec in ["pfr_no_watermark", "pfr"]:
        if dec not in panel["decoder"].unique():
            continue
        g = panel[panel["decoder"] == dec].sort_values("lookahead")
        for _, row in g.iterrows():
            L = int(row["lookahead"])
            ax.scatter(
                row["ANLPPT_U_mean"],
                row["AATPS_mean"],
                marker=mmap[L],
                s=MARKER_SIZE,
                facecolors="none",
                edgecolors=COLORS[dec],
                linewidths=MARKER_LW,
                zorder=3,
            )

    xvals, yvals = panel["ANLPPT_U_mean"].to_numpy(), panel["AATPS_mean"].to_numpy()
    x_range = float(np.max(xvals) - np.min(xvals)) if len(xvals) else 1.0
    annotate_red_deltas(ax, pairs_panel, x_range)

    ax.set_title("PFR no-wm vs PFR")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("ANLPPT-U")
    ax.set_ylabel("AATPS")
    set_limits(ax, xvals, yvals, 0.045, 0.09, extra_x=3.8)
    style_axis(ax)


def annotate_blue_loss(ax, row: pd.Series, x_range: float, y_range: float) -> None:
    x = float(row["wm_AATPS"])
    y = float(row["wm_ANLPPT_U"])
    loss = float(-row["delta_AATPS"])
    arrow_len = max(0.065 * x_range, 0.025)
    dy = max(0.048 * y_range, 0.0045)
    y0 = y - dy
    x_start = x + 0.20 * arrow_len
    x_end = x - 0.80 * arrow_len
    ax.annotate(
        "",
        xy=(x_end, y0),
        xytext=(x_start, y0),
        arrowprops=dict(arrowstyle="-|>", lw=ARROW_LW, color=COLORS["mc"], shrinkA=0, shrinkB=0),
        zorder=4,
    )
    ax.annotate(
        f"{loss:.3f}",
        ((x_start + x_end) / 2, y0),
        textcoords="offset points",
        xytext=(0, -13),
        ha="center",
        va="top",
        fontsize=ANNOT_FONT_SIZE,
        color=COLORS["mc"],
        zorder=4,
    )


def annotate_purple_trend(ax, purple_points: list[tuple[float, float]], x_range: float, y_range: float) -> None:
    if len(purple_points) < 2:
        return
    purple_points = sorted(purple_points, key=lambda t: t[0])
    for (x0, y0), (x1, y1) in zip(purple_points[:-1], purple_points[1:]):
        vx, vy = x1 - x0, y1 - y0
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        scale = 0.30
        xs, ys = mx - scale * vx, my - scale * vy
        xe, ye = mx + scale * vx, my + scale * vy
        if abs(xe - xs) > 0.15 * x_range:
            xs, xe = mx - 0.075 * x_range, mx + 0.075 * x_range
        if abs(ye - ys) > 0.15 * y_range:
            ys, ye = my + 0.075 * y_range, my - 0.075 * y_range
        ax.annotate(
            "",
            xy=(xe, ye),
            xytext=(xs, ys),
            arrowprops=dict(arrowstyle="-|>", lw=ARROW_LW, color=COLORS["mc_uwm_speed"], shrinkA=0, shrinkB=0),
            zorder=4,
        )


def plot_tradeoff(ax, summary: pd.DataFrame, pairs: pd.DataFrame) -> None:
    panel = summary[summary["decoder"].isin(TRADEOFF_METHODS)].copy()
    mmap = marker_map(panel["lookahead"].tolist())
    ordered = [d for d in ["mc", "pfr_no_watermark", "mc_uwm_strength", "mc_uwm_speed", "pfr"] if d in panel["decoder"].unique()]
    purple_pts: list[tuple[float, float]] = []

    for dec in ordered:
        g = panel[panel["decoder"] == dec].sort_values("lookahead")
        for _, row in g.iterrows():
            L = int(row["lookahead"])
            x, y = row["AATPS_mean"], row["ANLPPT_U_mean"]
            ax.scatter(
                x, y,
                marker=mmap[L],
                s=MARKER_SIZE,
                facecolors="none",
                edgecolors=COLORS[dec],
                linewidths=MARKER_LW,
                zorder=3,
            )
            if dec == "mc_uwm_speed":
                purple_pts.append((float(x), float(y)))

    ax.set_title("Tradeoff between watermark strength and sampling efficiency")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("AATPS")
    ax.set_ylabel("ANLPPT-U")

    xvals, yvals = panel["AATPS_mean"].to_numpy(), panel["ANLPPT_U_mean"].to_numpy()
    x_range = float(np.max(xvals) - np.min(xvals)) if len(xvals) else 1.0
    y_range = float(np.max(yvals) - np.min(yvals)) if len(yvals) else 1.0
    set_limits(ax, xvals, yvals, 0.055, 0.09, extra_x=1.9)

    blue_pairs = pairs[pairs["watermarked"] == "mc_uwm_strength"].sort_values("lookahead")
    for _, row in blue_pairs.iterrows():
        annotate_blue_loss(ax, row, x_range, y_range)

    annotate_purple_trend(ax, purple_pts, x_range, y_range)
    style_axis(ax)


def draw_combined_legends(ax, summary: pd.DataFrame) -> None:
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerPatch

    class HandlerWideRectangle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            rect = mpatches.Rectangle(
                (xdescent, ydescent + 0.16 * height),
                width,
                0.68 * height,
                facecolor=orig_handle.get_facecolor(),
                edgecolor=orig_handle.get_edgecolor(),
                lw=0,
                transform=trans,
            )
            return [rect]

    ax.axis("off")

    mmap = marker_map(summary["lookahead"].tolist())
    look_handles = [
        Line2D(
            [0], [0],
            marker=mmap[L],
            linestyle="None",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2.0,
            label=f"L={L}",
        )
        for L in sorted(mmap)
    ]

    method_order = ["pfr_no_watermark", "pfr", "mc", "mc_uwm_strength", "mc_uwm_speed"]
    method_handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=COLORS[d], edgecolor="none", label=DISPLAY[d])
        for d in method_order if d in summary["decoder"].unique()
    ]

    leg1 = ax.legend(
        handles=look_handles,
        title="Lookahead",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        bbox_to_anchor=(-0.14, 0.98),  # moved a bit left
        ncol=1,
        framealpha=0.9,
        columnspacing=1.2,
        handletextpad=0.8,
        borderpad=0.70,
        labelspacing=0.52,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=method_handles,
        title="Method",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc="upper left",
        bbox_to_anchor=(0.33, 0.98),
        ncol=1,
        framealpha=0.9,
        handlelength=3.1,
        handleheight=1.10,
        handletextpad=0.65,
        borderpad=0.55,
        labelspacing=0.30,
        handler_map={mpatches.Rectangle: HandlerWideRectangle()},
    )
    # Move each method label slightly upward; keep rectangles unchanged.
    for t in leg2.get_texts():
        x, y = t.get_position()
        t.set_position((x, y + 2.0))
    ax.add_artist(leg2)


def make_figure(summary: pd.DataFrame, pairs: pd.DataFrame, out: Path, dpi: int, figwidth: float, figheight: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(figwidth, figheight))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.32, 0.70], wspace=0.24)
    ax_trade = fig.add_subplot(outer[0])

    right = GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer[1],
        height_ratios=[1.55, 1.08],
        hspace=0.28,
    )
    ax_b = fig.add_subplot(right[0])
    ax_leg = fig.add_subplot(right[1])

    plot_tradeoff(ax_trade, summary, pairs)
    plot_pfr_effect(ax_b, summary, pairs)
    draw_combined_legends(ax_leg, summary)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.93)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args.inputs)
    df = load_rows(paths)
    summary = summarize(df)
    pairs = pair_table(summary)
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.csv_out, index=False)
    out = Path(args.out)
    make_figure(summary, pairs, out, args.dpi, args.figwidth, args.figheight)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
