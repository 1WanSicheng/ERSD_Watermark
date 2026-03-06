import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

METHODS = ["ersd", "ersd_wm", "ersd_nocc", "ersd_nocc_wm"]
BINS = [0.0, 1.0, 2.0, 4.0, 8.0, float("inf")]


def _collect_pairs(df):
    records = []
    for row in df.iter_rows(named=True):
        gaps = row.get("target_gaps") or []
        accs = row.get("accept_indicators") or []
        m = min(len(gaps), len(accs))
        for i in range(m):
            g = gaps[i]
            if g is None or not np.isfinite(g):
                continue
            records.append(
                {
                    "method": row["method"],
                    "n": row["n"],
                    "gap": float(g),
                    "accept": int(accs[i]),
                }
            )
    if not records:
        return pl.DataFrame({"method": [], "n": [], "gap": [], "accept": []})
    return pl.DataFrame(records)


def _bin_label(lo, hi):
    if hi == float("inf"):
        return f"[{lo},+inf)"
    return f"[{lo},{hi})"


def _assign_bin(gap):
    for i in range(len(BINS) - 1):
        if BINS[i] <= gap < BINS[i + 1]:
            return i
    return len(BINS) - 2


def _summary_table(pair_df):
    rows = []
    for (method, n_val), g in pair_df.group_by(["method", "n"]):
        gaps = g["gap"].to_list()
        accepts = g["accept"].to_list()
        for i in range(len(BINS) - 1):
            lo, hi = BINS[i], BINS[i + 1]
            idx = [j for j, gap in enumerate(gaps) if lo <= gap < hi]
            if not idx:
                continue
            acc = [accepts[j] for j in idx]
            rows.append(
                {
                    "method": method,
                    "n": n_val,
                    "gap_bin": _bin_label(lo, hi),
                    "count": len(acc),
                    "accept_rate": float(np.mean(acc)),
                }
            )
    rows = sorted(rows, key=lambda r: (r["method"], r["n"], r["gap_bin"]))
    return rows


def _write_table(rows, out_path):
    headers = ["method", "n", "gap_bin", "count", "accept_rate"]
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append("\t".join(str(r[h]) for h in headers))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_accept_curve(pair_df, n_val, out_path):
    plt.figure(figsize=(7.5, 5))
    x_labels = [_bin_label(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    x = np.arange(len(x_labels))
    for method in METHODS:
        data = pair_df.filter((pl.col("method") == method) & (pl.col("n") == n_val))
        if data.height == 0:
            continue
        gaps = data["gap"].to_list()
        accepts = data["accept"].to_list()
        ys = []
        for i in range(len(BINS) - 1):
            lo, hi = BINS[i], BINS[i + 1]
            idx = [j for j, gap in enumerate(gaps) if lo <= gap < hi]
            if not idx:
                ys.append(float("nan"))
                continue
            ys.append(float(np.mean([accepts[j] for j in idx])))
        plt.plot(x, ys, marker="o", label=method)
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.title(f"Acceptance vs Gap (K={n_val})")
    plt.xlabel("gap bin")
    plt.ylabel("accept_rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _plot_gap_hist(pair_df, n_val, out_path):
    plt.figure(figsize=(7.5, 5))
    for method in ["ersd", "ersd_wm"]:
        data = pair_df.filter((pl.col("method") == method) & (pl.col("n") == n_val))
        if data.height == 0:
            continue
        gaps = data["gap"].to_list()
        max_gap = max(max(gaps), 8.0) + 1.0
        edges = [0.0, 1.0, 2.0, 4.0, 8.0, max_gap]
        plt.hist(gaps, bins=edges, alpha=0.5, label=method, density=True)
    plt.title(f"Gap Distribution (K={n_val})")
    plt.xlabel("gap")
    plt.ylabel("density")
    plt.legend(loc="best")
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
    pair_df = _collect_pairs(ds)

    out_table = os.path.join(repo_root, "logs", "exp4_gap_accept_summary.txt")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    _write_table(_summary_table(pair_df), out_table)

    _plot_accept_curve(pair_df, 4, os.path.join(repo_root, "logs", "exp4_accept_vs_gap_k4.png"))
    _plot_accept_curve(pair_df, 8, os.path.join(repo_root, "logs", "exp4_accept_vs_gap_k8.png"))
    _plot_gap_hist(pair_df, 4, os.path.join(repo_root, "logs", "exp4_gap_hist_k4.png"))
    _plot_gap_hist(pair_df, 8, os.path.join(repo_root, "logs", "exp4_gap_hist_k8.png"))


if __name__ == "__main__":
    main()
