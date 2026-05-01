#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Methods shown in the final figure.
PANEL_A_METHODS = ["pfr_no_watermark", "pfr"]
PANEL_B_METHODS = ["mc", "mc_uwm_strength", "mc_uwm_speed"]
TRADEOFF_METHODS = ["mc", "mc_uwm_strength", "mc_uwm_speed", "pfr_no_watermark", "pfr"]
ALL_METHODS = PANEL_A_METHODS + PANEL_B_METHODS
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="JSON result files. If omitted, all *.json files in current folder are used.")
    ap.add_argument("--out", default="combined_spec_watermark_figure_v2.png")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument("--figwidth", type=float, default=12.6)
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


def annotate_vertical_drop(ax, x: float, y: float, val: float, color: str, y_range: float, x_range: float, side: str) -> None:
    arrow_len = max(0.04 * y_range, 0.035)
    dx = max(0.018 * x_range, 0.0018)
    if side == "left":
        x_arrow = x - 1.65 * dx
        text_xy, ha = (-8, -2), "right"
    else:
        x_arrow = x + 1.65 * dx
        text_xy, ha = (8, -2), "left"
    y_top = y + 0.55 * arrow_len
    y_bottom = y - 0.45 * arrow_len
    ax.annotate("", xy=(x_arrow, y_bottom), xytext=(x_arrow, y_top),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color, shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate(f"{val:.3f}", (x_arrow, y), textcoords="offset points", xytext=text_xy,
                ha=ha, va="center", fontsize=7.5, color=color, zorder=4)


def annotate_red_deltas(ax, pairs_panel: pd.DataFrame, x_range: float) -> None:
    for _, row in pairs_panel.sort_values("lookahead").iterrows():
        if row["watermarked"] != "pfr":
            continue
        x, y = float(row["wm_ANLPPT_U"]), float(row["wm_AATPS"])
        delta = float(row["delta_AATPS"])
        dx = max(0.02 * x_range, 0.0018)
        label = f"+{delta:.3f}" if delta > 0 else "0"
        ax.annotate(label, (x - 1.35 * dx, y), textcoords="offset points", xytext=(-2, -2),
                    ha="right", va="center", fontsize=7.5, color=COLORS["pfr"], zorder=4)


def plot_effect_panel(ax, summary_panel: pd.DataFrame, pairs_panel: pd.DataFrame, title: str, tag: str, add_red_text: bool) -> None:
    mmap = marker_map(summary_panel["lookahead"].tolist())
    ordered = list(dict.fromkeys(summary_panel["decoder"].tolist()))
    for dec in ordered:
        g = summary_panel[summary_panel["decoder"] == dec].sort_values("lookahead")
        for _, row in g.iterrows():
            L = int(row["lookahead"])
            ax.scatter(row["ANLPPT_U_mean"], row["AATPS_mean"], marker=mmap[L], s=66,
                       facecolors="none", edgecolors=COLORS[dec], linewidths=1.5, zorder=3)
    xvals, yvals = summary_panel["ANLPPT_U_mean"].to_numpy(), summary_panel["AATPS_mean"].to_numpy()
    x_range = float(np.max(xvals) - np.min(xvals)) if len(xvals) else 1.0
    y_range = float(np.max(yvals) - np.min(yvals)) if len(yvals) else 1.0
    for _, row in pairs_panel.iterrows():
        if row["delta_AATPS"] < 0:
            wm = row["watermarked"]
            side = "left" if wm == "mc_uwm_strength" else "right"
            ann_color = COLORS[wm]
            if wm == "mc_uwm_strength":
                ann_color = COLORS["mc_uwm_speed"]
            elif wm == "mc_uwm_speed":
                ann_color = COLORS["mc"]
            annotate_vertical_drop(ax, float(row["wm_ANLPPT_U"]), float(row["wm_AATPS"]), float(-row["delta_AATPS"]),
                                   ann_color, y_range, x_range, side)
    if add_red_text:
        annotate_red_deltas(ax, pairs_panel, x_range)
    ax.set_title(f"({tag}) {title}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("ANLPPT-U")
    ax.set_ylabel("AATPS")
    set_limits(ax, xvals, yvals, 0.04, 0.08, extra_x=3.5)


def annotate_blue_loss(ax, row: pd.Series, x_range: float, y_range: float) -> None:
    # Small left-pointing arrow below each blue point, with efficiency-loss number.
    # The annotation color indicates the comparison baseline (VSpS), so it is green.
    x = float(row["wm_AATPS"])
    y = float(row["wm_ANLPPT_U"])
    loss = float(-row["delta_AATPS"])
    arrow_len = max(0.06 * x_range, 0.02)
    dy = max(0.045 * y_range, 0.004)
    y0 = y - dy
    x_start = x + 0.20 * arrow_len
    x_end = x - 0.80 * arrow_len
    ax.annotate("", xy=(x_end, y0), xytext=(x_start, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COLORS["mc"], shrinkA=0, shrinkB=0), zorder=4)
    ax.annotate(f"{loss:.3f}", ((x_start + x_end) / 2, y0), textcoords="offset points", xytext=(0, -10),
                ha="center", va="top", fontsize=7.5, color=COLORS["mc"], zorder=4)


def annotate_purple_trend(ax, purple_points: list[tuple[float, float]], x_range: float, y_range: float) -> None:
    # Small right-down arrows between neighboring purple points.
    if len(purple_points) < 2:
        return
    purple_points = sorted(purple_points, key=lambda t: t[0])
    for (x0, y0), (x1, y1) in zip(purple_points[:-1], purple_points[1:]):
        vx, vy = x1 - x0, y1 - y0
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        scale = 0.28
        xs, ys = mx - scale * vx, my - scale * vy
        xe, ye = mx + scale * vx, my + scale * vy
        # keep the arrow visually small if two points are far apart
        if abs(xe - xs) > 0.14 * x_range:
            xs, xe = mx - 0.07 * x_range, mx + 0.07 * x_range
        if abs(ye - ys) > 0.14 * y_range:
            ys, ye = my + 0.07 * y_range, my - 0.07 * y_range
        ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COLORS["mc_uwm_speed"], shrinkA=0, shrinkB=0), zorder=4)


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
            ax.scatter(x, y, marker=mmap[L], s=60, facecolors="none", edgecolors=COLORS[dec], linewidths=1.4, zorder=3)
            if dec == "mc_uwm_speed":
                purple_pts.append((float(x), float(y)))
    ax.set_title("(a) Tradeoff between watermark strength and sampling efficiency", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("AATPS")
    ax.set_ylabel("ANLPPT-U")
    xvals, yvals = panel["AATPS_mean"].to_numpy(), panel["ANLPPT_U_mean"].to_numpy()
    x_range = float(np.max(xvals) - np.min(xvals)) if len(xvals) else 1.0
    y_range = float(np.max(yvals) - np.min(yvals)) if len(yvals) else 1.0
    set_limits(ax, xvals, yvals, 0.05, 0.08, extra_x=1.8)

    # Blue efficiency-loss annotations relative to VSpS.
    blue_pairs = pairs[pairs["watermarked"] == "mc_uwm_strength"].sort_values("lookahead")
    for _, row in blue_pairs.iterrows():
        annotate_blue_loss(ax, row, x_range, y_range)

    # Purple small right-down arrows showing the watermark decreasing trend.
    annotate_purple_trend(ax, purple_pts, x_range, y_range)


def add_global_legends(fig, ax_look, ax_method, summary: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D
    ax_look.axis("off")
    ax_method.axis("off")
    mmap = marker_map(summary["lookahead"].tolist())
    look_handles = [
        Line2D([0], [0], marker=mmap[L], linestyle="None", markersize=8,
               markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5, label=f"L={L}")
        for L in sorted(mmap)
    ]
    ax_look.legend(handles=look_handles, title="Lookahead", fontsize=9.5, title_fontsize=10.5,
                   loc="center", ncol=2, framealpha=0.9)

    method_order = ["pfr_no_watermark", "pfr", "mc", "mc_uwm_strength", "mc_uwm_speed"]
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8,
               markerfacecolor="none", markeredgecolor=COLORS[d], markeredgewidth=1.5, label=DISPLAY[d])
        for d in method_order if d in summary["decoder"].unique()
    ]
    ax_method.legend(handles=handles, title="Method", fontsize=9.5, title_fontsize=10.5,
                     loc="center", ncol=2, framealpha=0.9)


def make_figure(summary: pd.DataFrame, pairs: pd.DataFrame, out: Path, dpi: int, figwidth: float, figheight: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(figwidth, figheight))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.18], wspace=0.22)

    # Left = tradeoff view.
    ax_trade = fig.add_subplot(outer[0])

    # Right = top row legends + bottom row two equal-height panels.
    right = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], width_ratios=[0.78, 1.40],
                                    height_ratios=[0.25, 1.0], wspace=0.30, hspace=0.10)
    ax_look = fig.add_subplot(right[0, 0])
    ax_method = fig.add_subplot(right[0, 1])
    ax_a = fig.add_subplot(right[1, 0])
    ax_b = fig.add_subplot(right[1, 1])

    summary_a = summary[summary["decoder"].isin(PANEL_A_METHODS)].copy()
    summary_b = summary[summary["decoder"].isin(PANEL_B_METHODS)].copy()
    pairs_a = pairs[pairs["base"] == "pfr_no_watermark"].copy()
    pairs_b = pairs[pairs["base"] == "mc"].copy()

    plot_tradeoff(ax_trade, summary, pairs)
    plot_effect_panel(ax_a, summary_a, pairs_a, "PFR no-wm vs PFR", "b1", add_red_text=True)
    plot_effect_panel(ax_b, summary_b, pairs_b, "VSpS vs MWS / MSE", "b2", add_red_text=False)
    add_global_legends(fig, ax_look, ax_method, summary)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.95)
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
