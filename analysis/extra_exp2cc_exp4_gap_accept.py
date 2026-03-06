import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

METHODS = ["ersd", "ersd_cc", "ersd_wm"]
BINS = [0.0, 1.0, 2.0, 4.0, 8.0, float("inf")]


def _expand(df):
    return df.select(["method", "n", "target_gaps", "accept_indicators"]).explode(
        ["target_gaps", "accept_indicators"]
    )


def _bin_label(start, end):
    if end == float("inf"):
        return f"[{start}, +inf)"
    return f"[{start}, {end})"


def _bin_gap(gap):
    for i in range(len(BINS) - 1):
        if BINS[i] <= gap < BINS[i + 1]:
            return _bin_label(BINS[i], BINS[i + 1])
    return _bin_label(BINS[-2], BINS[-1])


def _accept_rate_by_bin(df):
    df = df.filter(pl.col("method").is_in(METHODS))
    df = df.with_columns(
        pl.col("target_gaps").map_elements(_bin_gap, return_dtype=pl.Utf8).alias("gap_bin")
    )
    return (
        df.group_by(["method", "gap_bin"])
        .agg(rate=pl.col("accept_indicators").mean())
        .sort(["method", "gap_bin"])
    )


def _plot_accept_rate(df, out_path):
    rates = _accept_rate_by_bin(df)
    bins = [_bin_label(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    plt.figure(figsize=(8, 5))
    for method in METHODS:
        ys = []
        for b in bins:
            val = rates.filter((pl.col("method") == method) & (pl.col("gap_bin") == b))["rate"]
            ys.append(float(val[0]) if len(val) else float("nan"))
        plt.plot(bins, ys, marker="o", label=method)
    plt.title("Acceptance rate vs gap bin")
    plt.xlabel("gap bin")
    plt.ylabel("accept_rate")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _plot_gap_cdf(df, out_path):
    plt.figure(figsize=(7.5, 5))
    for method in METHODS:
        gaps = df.filter(pl.col("method") == method)["target_gaps"].to_numpy()
        gaps = gaps[~np.isnan(gaps)]
        if gaps.size == 0:
            continue
        xs = np.sort(gaps)
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.plot(xs, ys, label=method)
    plt.title("Gap CDF")
    plt.xlabel("gap")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _write_table(df, out_path):
    rates = _accept_rate_by_bin(df)
    lines = ["method\tgap_bin\taccept_rate"]
    for row in rates.iter_rows(named=True):
        lines.append(f"{row['method']}\t{row['gap_bin']}\t{row['rate']}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(
        repo_root,
        "data_root",
        "extra_exp2cc_exp4_data",
        "summarization_scan_n",
        "Qwen2.5-7B-Instruct_Qwen2.5-0.5B-Instruct",
    )
    ds = pl.read_parquet(os.path.join(folder, "*"))
    ds = ds.filter(pl.col("method").is_in(METHODS))
    ds = _expand(ds)

    out_table = os.path.join(repo_root, "logs", "exp2cc_exp4_gap_accept_summary.txt")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    _write_table(ds, out_table)

    _plot_accept_rate(ds, os.path.join(repo_root, "logs", "exp2cc_exp4_accept_rate_by_gap.png"))
    _plot_gap_cdf(ds, os.path.join(repo_root, "logs", "exp2cc_exp4_gap_cdf.png"))


if __name__ == "__main__":
    main()
