#!/usr/bin/env python3
"""Estimate full-run time from a pilot raw CSV produced by compare_optimized_benchmark.py."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def latest_raw() -> Path:
    candidates = list(Path("outputs_optimized").glob("compare_raw_*.csv")) + list(Path("outputs_compare").glob("compare_raw_*.csv"))
    if not candidates:
        raise SystemExit("No compare_raw_*.csv found. Pass a raw CSV path.")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv", nargs="?")
    ap.add_argument("--target_prompts", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    raw = Path(args.raw_csv) if args.raw_csv else latest_raw()
    df = pd.read_csv(raw)
    if df.empty:
        raise SystemExit("Raw CSV is empty.")

    # Each row is one measured prompt for one dataset/config.
    total_time = float(df["total_time"].sum())
    rows = len(df)
    datasets = df["dataset"].nunique()
    configs = df["config_name"].nunique()
    measured_prompts_per_dataset_config = rows / max(datasets * configs, 1)

    time_per_prompt_config_dataset = total_time / max(rows, 1)
    target_rows = args.target_prompts * datasets * configs * args.seeds
    est = time_per_prompt_config_dataset * target_rows

    print(f"Pilot file: {raw}")
    print(f"Measured rows: {rows}")
    print(f"Datasets: {datasets}")
    print(f"Configs: {configs}")
    print(f"Measured prompts per dataset/config: {measured_prompts_per_dataset_config:.2f}")
    print(f"Pilot measured total_time sum: {total_time/60:.2f} minutes")
    print(f"Mean time per prompt/config/dataset row: {time_per_prompt_config_dataset:.2f} seconds")
    print()
    print(f"Estimated rows for target_prompts={args.target_prompts}, seeds={args.seeds}: {target_rows}")
    print(f"Estimated full measured generation time: {est/3600:.2f} hours")
    print("This excludes one-time model loading and warmup overhead, and assumes similar prompt lengths.")


if __name__ == "__main__":
    main()
