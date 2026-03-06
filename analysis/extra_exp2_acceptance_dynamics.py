import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

METHODS = ["ersd", "ersd_wm", "ersd_nocc", "ersd_nocc_wm"]


def _expand(df):
    return df.select(["method", "n", "accepted_draft_lens", "rejected_positions"]).explode(
        ["accepted_draft_lens", "rejected_positions"]
    )


def _acceptance_profile(df, n_val):
    prof = {}
    for method in METHODS:
        vals = df.filter((pl.col("method") == method) & (pl.col("n") == n_val))["accepted_draft_lens"].to_list()
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        prof[method] = [float(np.mean(np.array(vals) >= r)) for r in range(1, n_val + 1)]
    return prof


def _plot_acceptance_curve(df, n_val, out_path):
    prof = _acceptance_profile(df, n_val)
    plt.figure(figsize=(7.5, 5))
    for method, ys in prof.items():
        xs = list(range(1, n_val + 1))
        plt.plot(xs, ys, label=method, marker='o')
    plt.title(f"Acceptance Curve (K={n_val})")
    plt.xlabel("r")
    plt.ylabel("P(accepted_len >= r)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _plot_reject_hist(df, n_val, out_path):
    plt.figure(figsize=(7.5, 5))
    for method in METHODS:
        vals = df.filter((pl.col("method") == method) & (pl.col("n") == n_val))["rejected_positions"].to_list()
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        plt.hist(vals, bins=range(0, n_val + 2), alpha=0.5, label=method, density=True)
    plt.title(f"Reject Position Histogram (K={n_val})")
    plt.xlabel("rejected_pos (0 means no reject)")
    plt.ylabel("density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _summary_table(df):
    rows = []
    for (method, n_val), g in df.group_by(["method", "n"]):
        acc = [v for v in g["accepted_draft_lens"].to_list() if v is not None]
        rej = [v for v in g["rejected_positions"].to_list() if v is not None]
        if not acc:
            continue
        rows.append(
            {
                "method": method,
                "n": n_val,
                "accepted_len_mean": float(np.mean(acc)),
                "accepted_len_median": float(np.median(acc)),
                "num_retries_mean": float(np.mean([1 if r > 0 else 0 for r in rej])) if rej else float("nan"),
            }
        )
    rows = sorted(rows, key=lambda r: (r["method"], r["n"]))
    return rows


def _write_table(rows, out_path):
    headers = ["method", "n", "accepted_len_mean", "accepted_len_median", "num_retries_mean"]
    lines = ["	".join(headers)]
    for r in rows:
        lines.append("	".join(str(r[h]) for h in headers))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(
        repo_root,
        "data_root",
        "extra_exp2_data",
        "summarization_scan_n",
        "Qwen2.5-7B-Instruct_Qwen2.5-0.5B-Instruct",
    )
    ds = pl.read_parquet(os.path.join(folder, "*"))
    ds = ds.filter(pl.col("method").is_in(METHODS))
    ds = _expand(ds)

    out_table = os.path.join(repo_root, "logs", "exp2_acceptance_summary.txt")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    _write_table(_summary_table(ds), out_table)

    _plot_acceptance_curve(ds, 4, os.path.join(repo_root, "logs", "exp2_acceptance_curve_k4.png"))
    _plot_acceptance_curve(ds, 8, os.path.join(repo_root, "logs", "exp2_acceptance_curve_k8.png"))
    _plot_reject_hist(ds, 4, os.path.join(repo_root, "logs", "exp2_reject_hist_k4.png"))
    _plot_reject_hist(ds, 8, os.path.join(repo_root, "logs", "exp2_reject_hist_k8.png"))


if __name__ == "__main__":
    main()
