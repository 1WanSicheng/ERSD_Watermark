from pathlib import Path
import sys
import pandas as pd

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    paths = sorted(Path("outputs_three_method").glob("three_method_summary_*.csv"))
if not paths:
    raise SystemExit("No summary CSVs found. Pass paths or run from the experiment folder.")

dfs = [pd.read_csv(p) for p in paths]
df = pd.concat(dfs, ignore_index=True)
# Add B for display.
df["B"] = df["config_name"].str.extract(r"_(\d+)$").astype(float)
cols = [
    "seed", "dataset", "method", "config_name",
    "token_rate_mean", "aatps_mean", "be_mean",
    "acceptance_fraction_mean",
    "target_forward_calls_per_token_mean",
    "draft_forward_calls_per_token_mean",
]
print("\n=== Per summary row ===")
print(df[cols].sort_values(["seed", "dataset", "B", "method"]).to_string(index=False))
print("\n=== Mean over seeds and datasets, by method and B ===")
agg = df.groupby(["method", "B"])[[
    "token_rate_mean", "aatps_mean", "be_mean",
    "acceptance_fraction_mean", "target_forward_calls_per_token_mean",
    "draft_forward_calls_per_token_mean",
]].mean().reset_index()
print(agg.sort_values(["B", "method"]).to_string(index=False))
