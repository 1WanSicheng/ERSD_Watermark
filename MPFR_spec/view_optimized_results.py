#!/usr/bin/env python3
"""Pretty-print optimized MPFR vs list-level comparison results."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


KEY_METRICS = [
    "token_rate",
    "aatps",
    "be",
    "acceptance_fraction",
    "target_forward_calls_per_token",
    "draft_forward_calls_per_token",
    "target_contexts_per_token",
    "draft_tree_nodes_per_token",
    "speedup_vs_target_only_pct",
    "speedup_vs_single_draft_pct",
]


def latest_file(patterns: list[str]) -> Path:
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(Path(".").glob(pat))
    if not candidates:
        raise SystemExit("No result CSV found. Pass a raw or summary CSV path explicitly.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def is_summary(df: pd.DataFrame) -> bool:
    return "token_rate_mean" in df.columns


def summarize_raw(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be",
        "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_contexts_total", "draft_tree_nodes_total", "target_forward_calls_total",
        "draft_forward_calls_total", "target_contexts_per_token", "draft_tree_nodes_per_token",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(["dataset", "algorithm", "config_name"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary["speedup_vs_target_only_pct"] = np.nan
    summary["speedup_vs_single_draft_pct"] = np.nan
    for dataset, idx in summary.groupby("dataset").groups.items():
        sub = summary.loc[idx]
        target = sub.loc[sub["algorithm"].eq("target_only"), "token_rate_mean"]
        single = sub.loc[sub["algorithm"].eq("single_draft"), "token_rate_mean"]
        if len(target) and float(target.iloc[0]) > 0:
            b = float(target.iloc[0])
            summary.loc[idx, "speedup_vs_target_only_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)
        if len(single) and float(single.iloc[0]) > 0:
            b = float(single.iloc[0])
            summary.loc[idx, "speedup_vs_single_draft_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)
    return summary


def fmt(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="Raw compare_raw_*.csv or summary compare_summary_*.csv")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--sort", default="token_rate_mean", help="Column to sort by")
    ap.add_argument("--ascending", action="store_true")
    ap.add_argument("--max_rows", type=int, default=200)
    args = ap.parse_args()

    path = Path(args.csv) if args.csv else latest_file([
        "outputs_optimized/compare_summary_*.csv",
        "outputs_optimized/compare_raw_*.csv",
        "outputs_compare/compare_summary_*.csv",
        "outputs_compare/compare_raw_*.csv",
    ])
    df = pd.read_csv(path)
    summary = df if is_summary(df) else summarize_raw(df)

    if args.dataset:
        summary = summary[summary["dataset"].eq(args.dataset)]

    cols = ["dataset", "algorithm", "config_name"]
    for m in KEY_METRICS:
        c = m if m in summary.columns else f"{m}_mean"
        if c in summary.columns:
            cols.append(c)
    cols = [c for c in cols if c in summary.columns]

    if args.sort in summary.columns:
        summary = summary.sort_values(args.sort, ascending=args.ascending)
    elif args.sort + "_mean" in summary.columns:
        summary = summary.sort_values(args.sort + "_mean", ascending=args.ascending)

    show = summary[cols].head(args.max_rows).copy()
    for c in show.columns:
        if c not in {"dataset", "algorithm", "config_name"}:
            show[c] = show[c].map(lambda x: fmt(x))
    print(f"\nFile: {path}\n")
    print(show.to_string(index=False))

    # Compact best-by-dataset table.
    if "token_rate_mean" in summary.columns:
        print("\nBest token_rate per dataset:")
        rows = []
        for dataset, sub in summary.groupby("dataset"):
            best = sub.loc[sub["token_rate_mean"].idxmax()]
            rows.append({
                "dataset": dataset,
                "config_name": best["config_name"],
                "algorithm": best["algorithm"],
                "token_rate": best["token_rate_mean"],
                "aatps": best.get("aatps_mean", np.nan),
                "be": best.get("be_mean", np.nan),
            })
        best_df = pd.DataFrame(rows)
        for c in ["token_rate", "aatps", "be"]:
            best_df[c] = best_df[c].map(lambda x: fmt(x))
        print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
