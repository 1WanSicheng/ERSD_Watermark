#!/usr/bin/env python3
"""Summarize raw optimized comparison CSV files from compare_decoding_benchmark.py."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Raw CSV written by compare_decoding_benchmark.py")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    raw = Path(args.csv)
    out = Path(args.out) if args.out else raw.with_name(raw.stem + "_summary.csv")
    df = pd.read_csv(raw)

    metrics = [
        "token_rate", "aatps", "be", "tokens_per_step", "acceptance_fraction",
        "normalized_aatps", "total_time", "num_steps", "num_generated_tokens",
        "target_contexts_per_token", "draft_tree_nodes_per_token",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
    ]
    metrics = [m for m in metrics if m in df.columns]

    group_cols = ["dataset", "algorithm", "config_name"]
    summary = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    summary["speedup_vs_target_only_pct"] = np.nan
    summary["speedup_vs_single_draft_pct"] = np.nan
    for dataset, idx in summary.groupby("dataset").groups.items():
        sub = summary.loc[idx]
        base_target = sub.loc[sub["algorithm"].eq("target_only"), "token_rate_mean"]
        base_single = sub.loc[sub["algorithm"].eq("single_draft"), "token_rate_mean"]
        if len(base_target) and float(base_target.iloc[0]) > 0:
            b = float(base_target.iloc[0])
            summary.loc[idx, "speedup_vs_target_only_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)
        if len(base_single) and float(base_single.iloc[0]) > 0:
            b = float(base_single.iloc[0])
            summary.loc[idx, "speedup_vs_single_draft_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)

    summary.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 240):
        print(summary.to_string(index=False))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
