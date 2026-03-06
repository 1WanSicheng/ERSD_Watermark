import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

TAUS = [1.0, 2.0, 4.0, 8.0]
METHODS = ["ersd", "ersd_wm", "ersd_nocc", "ersd_nocc_wm"]


def _flatten_gap(df):
    return df.explode("target_gaps").filter(pl.col("target_gaps").is_not_null())


def _summary_table(df):
    rows = []
    for (method, n), g in df.group_by(["method", "n"]):
        vals = g["target_gaps"].to_list()
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        mean = float(np.mean(vals))
        median = float(np.median(vals))
        row = {
            "method": method,
            "n": n,
            "gap_mean": mean,
            "gap_median": median,
        }
        for tau in TAUS:
            row[f"p_gap_gt_{tau}"] = float(np.mean(np.array(vals) > tau))
        rows.append(row)
    rows = sorted(rows, key=lambda r: (r["method"], r["n"]))
    return rows


def _write_table(rows, out_path):
    headers = ["method", "n", "gap_mean", "gap_median"] + [f"p_gap_gt_{t}" for t in TAUS]
    lines = ["	".join(headers)]
    for r in rows:
        line = [str(r[h]) for h in headers]
        lines.append("	".join(line))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_cdf(df, n_val, out_path):
    plt.figure(figsize=(7.5, 5))
    for method in METHODS:
        vals = df.filter((pl.col("method") == method) & (pl.col("n") == n_val))["target_gaps"].to_list()
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=method)
    plt.title(f"Gap CDF (K={n_val})")
    plt.xlabel("gap")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _plot_box(df, out_path):
    data = []
    labels = []
    for n_val in sorted(df["n"].unique().to_list()):
        for method in METHODS:
            vals = df.filter((pl.col("method") == method) & (pl.col("n") == n_val))["target_gaps"].to_list()
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            data.append(vals)
            labels.append(f"{method}_K{n_val}")
    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Gap Distribution by Method and K")
    plt.ylabel("gap")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(
        repo_root,
        "data_root",
        "extra_exp1_data",
        "summarization_scan_n",
        "Qwen2.5-7B-Instruct_Qwen2.5-0.5B-Instruct",
    )
    ds = pl.read_parquet(os.path.join(folder, "*"))
    ds = ds.filter(pl.col("method").is_in(METHODS))
    ds = _flatten_gap(ds)

    rows = _summary_table(ds)
    out_table = os.path.join(repo_root, "logs", "exp1_gap_summary.txt")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    _write_table(rows, out_table)

    _plot_cdf(ds, 4, os.path.join(repo_root, "logs", "exp1_gap_cdf_k4.png"))
    _plot_cdf(ds, 8, os.path.join(repo_root, "logs", "exp1_gap_cdf_k8.png"))
    _plot_box(ds, os.path.join(repo_root, "logs", "exp1_gap_box.png"))


if __name__ == "__main__":
    main()
