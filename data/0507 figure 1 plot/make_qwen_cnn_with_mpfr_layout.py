#!/usr/bin/env python3
"""
Custom Qwen+CNN layout with single-draft methods and multi-draft MPFR points.

Layout:
  - Top-wide panel: AATPS vs ANLPPT-U for Qwen+CNN single-draft schemes
    plus all MPFR points for B in {2,4,6,8}, L in {2,4,6,8}.
  - Bottom-left panel: ANLPPT-U vs AATPS for PFR no-wm and PFR.
  - Bottom-middle panel: ANLPPT-U vs AATPS for VSpS, MWS, and MSE.
  - Bottom-right panel: compact Lookahead and Method legends.

Example:
    python make_qwen_cnn_with_mpfr_layout.py \
      single_draft_qwen_n1000_len128_L1-2-3-4_topk50_cnn_a.json \
      single_draft_qwen_n1000_len128_L1-2-3-4_topk50_cnn_b.json \
      multi_draft_qwen_cnn_n500_L8642_B2-4-6-8.json \
      --out qwen_cnn_with_mpfr_custom_layout.png \
      --summary-out qwen_cnn_with_mpfr_custom_layout_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DISPLAY = {
    "mc": "VSpS",
    "mc_uwm_strength": "MWS",
    "mc_uwm_speed": "MSE",
    "pfr_no_watermark": "PFR no-wm",
    "pfr": "PFR",
    "mpfr_torchgen_cached": "MPFR",
}

COLORS = {
    "mc": "teal",
    "mc_uwm_strength": "royalblue",
    "mc_uwm_speed": "mediumpurple",
    "pfr_no_watermark": "dimgray",
    "pfr": "firebrick",
    "mpfr_torchgen_cached": "darkorange",
}

METHOD_ORDER = [
    "mc",
    "mc_uwm_strength",
    "mc_uwm_speed",
    "pfr_no_watermark",
    "pfr",
    "mpfr_torchgen_cached",
]
SINGLE_METHODS = ["mc", "mc_uwm_strength", "mc_uwm_speed", "pfr_no_watermark", "pfr"]
PFR_PANEL_METHODS = ["pfr_no_watermark", "pfr"]
OTHER_PANEL_METHODS = ["mc", "mc_uwm_strength", "mc_uwm_speed"]
NO_WM_ZERO_BASELINES = {"mc", "basic"}

LOOKAHEAD_MARKERS = {1: "o", 2: "s", 3: "D", 4: "X", 6: "P", 8: "*"}
FALLBACK_MARKERS = ["v", "<", ">", "h", "8"]

MARKER_SIZE = 120
MPFR_MARKER_SIZE = 156
MARKER_LW = 2.3
AXIS_LABEL_SIZE = 17
TICK_LABEL_SIZE = 15
TITLE_SIZE = 18
LEGEND_FONT_SIZE = 15
LEGEND_TITLE_SIZE = 16
ANNOT_SIZE = 10.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="Two single-draft Qwen+CNN JSON files and one multi-draft Qwen+CNN JSON file.")
    ap.add_argument("--out", default="qwen_cnn_with_mpfr_custom_layout.png")
    ap.add_argument("--summary-out", default="qwen_cnn_with_mpfr_custom_layout_summary.csv")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figwidth", type=float, default=15.8)
    ap.add_argument("--figheight", type=float, default=9.8)
    return ap.parse_args()


def resolve_inputs(inputs: list[str]) -> list[Path]:
    paths = [Path(p) for p in inputs] if inputs else sorted(Path(".").glob("*.json"))
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))
    if not paths:
        raise FileNotFoundError("No input JSON files found.")
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def config_label(obj: dict[str, Any]) -> tuple[str, str, str, str]:
    cfg = obj.get("config", {})
    return (
        str(cfg.get("experiment", "")),
        str(cfg.get("dataset", "")),
        str(cfg.get("target_model", "")),
        str(cfg.get("draft_model", "")),
    )


def validate_qwen_cnn(paths: list[Path]) -> None:
    """Lightweight consistency checks; prints warnings but does not block plotting."""
    for path in paths:
        obj = load_json(path)
        cfg = obj.get("config", {})
        exp = cfg.get("experiment")
        dataset = cfg.get("dataset")
        target = str(cfg.get("target_model", ""))
        draft = str(cfg.get("draft_model", ""))
        if dataset != "cnn_dailymail":
            print(f"[warn] {path.name}: dataset is {dataset!r}, expected 'cnn_dailymail'.")
        if "Qwen2.5-7B-Instruct" not in target and "Qwen" not in target:
            print(f"[warn] {path.name}: target model does not look like Qwen: {target}")
        if "Qwen2.5-0.5B-Instruct" not in draft and "Qwen" not in draft:
            print(f"[warn] {path.name}: draft model does not look like Qwen: {draft}")
        print(f"[ok] {path.name}: experiment={exp}, dataset={dataset}, samples={cfg.get('samples')}, lookaheads={cfg.get('lookaheads')}, decoders={cfg.get('decoders')}")


def load_single_rows(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        obj = load_json(path)
        if obj.get("config", {}).get("experiment") == "multi_draft":
            continue
        for r in obj.get("rows", []):
            d = dict(r)
            d["source_file"] = path.name
            rows.append(d)
    if not rows:
        raise RuntimeError("No single-draft rows were found.")
    return pd.DataFrame(rows)


def summarize_single(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"decoder", "lookahead", "AATPS"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Single-draft rows missing required column(s): {sorted(missing)}")
    if "ANLPPT_U" not in df.columns:
        df = df.copy()
        df["ANLPPT_U"] = np.nan

    rows: list[dict[str, Any]] = []
    for (decoder, lookahead), g in df.groupby(["decoder", "lookahead"], dropna=False):
        decoder = str(decoder)
        if decoder not in SINGLE_METHODS:
            continue
        a = safe_numeric(g["AATPS"]).dropna()
        u = safe_numeric(g["ANLPPT_U"]).dropna()
        if a.empty:
            continue
        if not u.empty:
            u_mean = float(u.mean())
        elif decoder in NO_WM_ZERO_BASELINES:
            # mc has wm_kind="none" and NaN ANLPPT in the run file; use zero in the tradeoff plot.
            u_mean = 0.0
        else:
            continue
        rows.append({
            "decoder": decoder,
            "display": DISPLAY.get(decoder, decoder),
            "lookahead": int(lookahead),
            "num_drafts": np.nan,
            "AATPS_mean": float(a.mean()),
            "ANLPPT_U_mean": u_mean,
            "n_prompts": int(len(a)),
            "source": "single_draft",
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid single-draft summary rows were found.")
    return out


def summarize_mpfr_from_rows(obj: dict[str, Any]) -> pd.DataFrame:
    rows = obj.get("rows", [])
    if not rows:
        raise RuntimeError("The MPFR file has no row-level data.")
    df = pd.DataFrame(rows)
    needed = {"decoder", "lookahead", "num_drafts", "AATPS", "ANLPPT_U"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"MPFR rows missing required column(s): {sorted(missing)}")

    out_rows: list[dict[str, Any]] = []
    for (decoder, lookahead, num_drafts), g in df.groupby(["decoder", "lookahead", "num_drafts"], dropna=False):
        decoder = str(decoder)
        if decoder != "mpfr_torchgen_cached":
            continue
        a = safe_numeric(g["AATPS"]).dropna()
        u = safe_numeric(g["ANLPPT_U"]).dropna()
        if a.empty or u.empty:
            continue
        out_rows.append({
            "decoder": decoder,
            "display": DISPLAY.get(decoder, decoder),
            "lookahead": int(lookahead),
            "num_drafts": int(num_drafts),
            "AATPS_mean": float(a.mean()),
            "ANLPPT_U_mean": float(u.mean()),
            "n_prompts": int(min(len(a), len(u))),
            "source": "multi_draft",
        })
    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No valid MPFR summary rows were found from row-level data.")
    return out


def summarize_mpfr_from_summary(obj: dict[str, Any]) -> pd.DataFrame:
    out_rows = []
    pat = re.compile(r"^mpfr_torchgen_cached_L(?P<L>\d+)_B(?P<B>\d+)$")
    for key, val in obj.get("summary", {}).items():
        m = pat.match(key)
        if not m or "AATPS" not in val or "ANLPPT_U" not in val:
            continue
        out_rows.append({
            "decoder": "mpfr_torchgen_cached",
            "display": DISPLAY["mpfr_torchgen_cached"],
            "lookahead": int(m.group("L")),
            "num_drafts": int(m.group("B")),
            "AATPS_mean": float(val["AATPS"]),
            "ANLPPT_U_mean": float(val["ANLPPT_U"]),
            "n_prompts": np.nan,
            "source": "multi_draft",
        })
    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No valid MPFR summary rows were found.")
    return out


def load_mpfr(paths: list[Path]) -> pd.DataFrame:
    candidates = []
    for path in paths:
        obj = load_json(path)
        if obj.get("config", {}).get("experiment") == "multi_draft":
            candidates.append((path, obj))
    if not candidates:
        raise RuntimeError("No multi-draft MPFR JSON file was found among inputs.")
    if len(candidates) > 1:
        print("[warn] Multiple multi-draft files found; using the first one:", candidates[0][0])
    _, obj = candidates[0]
    try:
        return summarize_mpfr_from_rows(obj)
    except Exception as exc:
        print(f"[warn] Could not summarize MPFR from rows ({exc}); falling back to summary.")
        return summarize_mpfr_from_summary(obj)


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


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.1)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.title.set_fontsize(TITLE_SIZE)


def set_padded_limits(ax, xvals: np.ndarray, yvals: np.ndarray, xpad=0.08, ypad=0.10, force_y_zero=False) -> None:
    xvals = np.asarray(xvals, dtype=float)
    yvals = np.asarray(yvals, dtype=float)
    xvals = xvals[np.isfinite(xvals)]
    yvals = yvals[np.isfinite(yvals)]
    if xvals.size == 0 or yvals.size == 0:
        return
    xmin, xmax = float(xvals.min()), float(xvals.max())
    ymin, ymax = float(yvals.min()), float(yvals.max())
    if force_y_zero:
        ymin = min(0.0, ymin)
    xs = max(xmax - xmin, 1e-6)
    ys = max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - xpad * xs, xmax + xpad * xs)
    ax.set_ylim(ymin - ypad * ys, ymax + ypad * ys)


def scatter_points(ax, data: pd.DataFrame, *, xcol: str, ycol: str, mmap: dict[int, str], include_b_labels=False) -> None:
    for method in METHOD_ORDER:
        g = data[data["decoder"] == method].copy()
        if g.empty:
            continue
        g = g.sort_values(["lookahead", "num_drafts"], na_position="first")
        for _, row in g.iterrows():
            L = int(row["lookahead"])
            is_mpfr = method == "mpfr_torchgen_cached"
            ax.scatter(
                float(row[xcol]),
                float(row[ycol]),
                marker=mmap.get(L, "o"),
                s=MPFR_MARKER_SIZE if is_mpfr else MARKER_SIZE,
                facecolors="none",
                edgecolors=COLORS[method],
                linewidths=MARKER_LW if not is_mpfr else MARKER_LW + 0.2,
                alpha=0.95,
                zorder=4 if is_mpfr else 3,
            )
            if include_b_labels and is_mpfr and np.isfinite(row.get("num_drafts", np.nan)):
                b = int(row["num_drafts"])
                # Compact B labels; L is encoded by marker shape.
                ax.annotate(
                    f"B{b}",
                    (float(row[xcol]), float(row[ycol])),
                    textcoords="offset points",
                    xytext=(4, 4 if b in {2, 6} else -8),
                    ha="left",
                    va="bottom" if b in {2, 6} else "top",
                    fontsize=ANNOT_SIZE,
                    color=COLORS[method],
                    alpha=0.95,
                )


def make_top_panel(ax, summary_all: pd.DataFrame, mmap: dict[int, str]) -> None:
    scatter_points(ax, summary_all, xcol="AATPS_mean", ycol="ANLPPT_U_mean", mmap=mmap, include_b_labels=True)
    ax.set_title("Qwen+CNN: AATPS--ANLPPT-U tradeoff with MPFR")
    ax.set_xlabel("AATPS")
    ax.set_ylabel("ANLPPT-U")
    set_padded_limits(ax, summary_all["AATPS_mean"].to_numpy(), summary_all["ANLPPT_U_mean"].to_numpy(), xpad=0.045, ypad=0.10, force_y_zero=True)
    style_axis(ax)


def make_pfr_panel(ax, summary_single: pd.DataFrame, mmap: dict[int, str]) -> None:
    panel = summary_single[summary_single["decoder"].isin(PFR_PANEL_METHODS)].copy()
    scatter_points(ax, panel, xcol="ANLPPT_U_mean", ycol="AATPS_mean", mmap=mmap)
    ax.set_title("PFR no-wm vs PFR")
    ax.set_xlabel("ANLPPT-U")
    ax.set_ylabel("AATPS")
    set_padded_limits(ax, panel["ANLPPT_U_mean"].to_numpy(), panel["AATPS_mean"].to_numpy(), xpad=0.14, ypad=0.10)
    style_axis(ax)


def make_other_panel(ax, summary_single: pd.DataFrame, mmap: dict[int, str]) -> None:
    panel = summary_single[summary_single["decoder"].isin(OTHER_PANEL_METHODS)].copy()
    scatter_points(ax, panel, xcol="ANLPPT_U_mean", ycol="AATPS_mean", mmap=mmap)
    ax.set_title("VSpS, MWS, and MSE")
    ax.set_xlabel("ANLPPT-U")
    ax.set_ylabel("AATPS")
    set_padded_limits(ax, panel["ANLPPT_U_mean"].to_numpy(), panel["AATPS_mean"].to_numpy(), xpad=0.12, ypad=0.10)
    style_axis(ax)


def draw_legends(ax, summary_all: pd.DataFrame, mmap: dict[int, str]) -> None:
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerPatch

    class HandlerWideRectangle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            # Use most of the available legend-row height so the color blocks remain
            # visually balanced with the larger fonts.
            rect = mpatches.Rectangle(
                (xdescent, ydescent + 0.12 * height),
                width,
                0.76 * height,
                facecolor=orig_handle.get_facecolor(),
                edgecolor=orig_handle.get_edgecolor(),
                lw=0,
                transform=trans,
            )
            return [rect]

    ax.axis("off")
    lookahead_values = sorted(set(int(x) for x in summary_all["lookahead"].dropna().tolist()))
    lookahead_handles = [
        Line2D(
            [0], [0],
            marker=mmap[L],
            linestyle="None",
            markersize=12.5,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2.4,
            label=f"L={L}",
        )
        for L in lookahead_values
    ]
    method_values = [m for m in METHOD_ORDER if m in set(summary_all["decoder"])]
    method_handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="none", label=DISPLAY[m])
        for m in method_values
    ]

    # Anchor both legends to the right edge of the legend column.  Since the top
    # panel spans all GridSpec columns, this makes the right boundary of the
    # legend boxes match the right boundary of the upper panel.
    leg1 = ax.legend(
        handles=lookahead_handles,
        title="Lookahead",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.99),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        framealpha=0.94,
        borderpad=0.52,
        columnspacing=0.95,
        handletextpad=0.50,
        labelspacing=0.35,
        borderaxespad=0.0,
    )
    ax.add_artist(leg1)

    leg2 = ax.legend(
        handles=method_handles,
        title="Method",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.00),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        framealpha=0.94,
        borderpad=0.52,
        handlelength=2.8,
        handleheight=1.00,
        handletextpad=0.70,
        labelspacing=0.28,
        borderaxespad=0.0,
        handler_map={mpatches.Rectangle: HandlerWideRectangle()},
    )
    ax.add_artist(leg2)


def validate_grid(mpfr: pd.DataFrame) -> None:
    grid = sorted((int(r.lookahead), int(r.num_drafts)) for r in mpfr.itertuples(index=False))
    expected = sorted((L, B) for L in [2, 4, 6, 8] for B in [2, 4, 6, 8])
    missing = sorted(set(expected) - set(grid))
    extra = sorted(set(grid) - set(expected))
    if missing:
        print("[warn] Missing MPFR grid points:", missing)
    if extra:
        print("[warn] Extra MPFR grid points:", extra)
    if not missing:
        print("[ok] MPFR grid contains all B={2,4,6,8}, L={2,4,6,8} points.")


def make_figure(summary_single: pd.DataFrame, mpfr: pd.DataFrame, out: Path, *, dpi: int, figwidth: float, figheight: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    summary_all = pd.concat([summary_single, mpfr], ignore_index=True)
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary_all["_method_rank"] = summary_all["decoder"].map(method_rank).fillna(999)
    summary_all = summary_all.sort_values(["_method_rank", "lookahead", "num_drafts"], na_position="first")
    mmap = marker_map(summary_all["lookahead"].astype(int).tolist())

    fig = plt.figure(figsize=(figwidth, figheight))
    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.10, 1.0],
        width_ratios=[2.0, 2.0, 1.18],
        hspace=0.46,
        wspace=0.34,
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_pfr = fig.add_subplot(gs[1, 0])
    ax_other = fig.add_subplot(gs[1, 1])
    ax_leg = fig.add_subplot(gs[1, 2])

    make_top_panel(ax_top, summary_all, mmap)
    make_pfr_panel(ax_pfr, summary_single, mmap)
    make_other_panel(ax_other, summary_single, mmap)
    draw_legends(ax_leg, summary_all, mmap)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.925, bottom=0.095)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = resolve_inputs(args.inputs)
    validate_qwen_cnn(paths)
    single_df = load_single_rows(paths)
    summary_single = summarize_single(single_df)
    mpfr = load_mpfr(paths)
    validate_grid(mpfr)

    summary_all = pd.concat([summary_single, mpfr], ignore_index=True)
    summary_all = summary_all.sort_values(["source", "decoder", "lookahead", "num_drafts"], na_position="first")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        summary_all.to_csv(args.summary_out, index=False)
        print(f"wrote {args.summary_out}")

    make_figure(summary_single, mpfr, Path(args.out), dpi=args.dpi, figwidth=args.figwidth, figheight=args.figheight)
    print(f"wrote {args.out}")

    print("\nPlotted coordinate summary:")
    cols = ["source", "display", "lookahead", "num_drafts", "AATPS_mean", "ANLPPT_U_mean", "n_prompts"]
    print(summary_all[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
