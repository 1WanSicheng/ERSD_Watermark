import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

METHODS = ["ersd", "ersd_wm", "ersd_nocc", "ersd_nocc_wm"]


def _summary(df):
    rows = []
    for (method, n_val), g in df.group_by(["method", "n"]):
        out_lens = [sum(x) for x in g["gen_seq_lens"].to_list()]
        target_fwd = g["target_fwd"].to_list()
        draft_fwd = g["draft_fwd"].to_list()
        verify_ops = g["verify_ops"].to_list()
        wall_times = (g["t_got_last_output"] - g["t_got_input"]).to_list()
        cost_target = [t / o if o else float("nan") for t, o in zip(target_fwd, out_lens)]
        cost_draft = [d / o if o else float("nan") for d, o in zip(draft_fwd, out_lens)]
        wall_per_tok = [w / o if o else float("nan") for w, o in zip(wall_times, out_lens)]
        rows.append(
            {
                "method": method,
                "n": n_val,
                "N_out": float(np.mean(out_lens)),
                "cost_target": float(np.mean(cost_target)),
                "draft_cost": float(np.mean(cost_draft)),
                "verify_ops": float(np.mean(verify_ops)),
                "wall_time_token": float(np.mean(wall_per_tok)),
                "AATPS_proxy": float(
                    np.mean([1 / c if c > 0 else float("nan") for c in cost_target])
                ),
            }
        )
    rows = sorted(rows, key=lambda r: (r["method"], r["n"]))
    return rows


def _write_table(rows, out_path):
    headers = [
        "method",
        "n",
        "N_out",
        "cost_target",
        "draft_cost",
        "verify_ops",
        "wall_time_token",
        "AATPS_proxy",
    ]
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append("\t".join(str(r[h]) for h in headers))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_cost_target(rows, out_path):
    plt.figure(figsize=(8, 5))
    for method in METHODS:
        data = [r for r in rows if r["method"] == method]
        if not data:
            continue
        ks = [r["n"] for r in data]
        vals = [r["cost_target"] for r in data]
        plt.plot(ks, vals, marker="o", label=method)
    plt.title("cost_target vs K")
    plt.xlabel("K")
    plt.ylabel("cost_target")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    folder = os.path.join(
        repo_root,
        "data_root",
        "extra_exp3_data",
        "summarization_scan_n",
        "Qwen2.5-7B-Instruct_Qwen2.5-0.5B-Instruct",
    )
    ds = pl.read_parquet(os.path.join(folder, "*"))
    ds = ds.filter(pl.col("method").is_in(METHODS))
    rows = _summary(ds)

    out_table = os.path.join(repo_root, "logs", "exp3_decompose_summary.txt")
    os.makedirs(os.path.dirname(out_table), exist_ok=True)
    _write_table(rows, out_table)

    _plot_cost_target(rows, os.path.join(repo_root, "logs", "exp3_cost_target_vs_k.png"))


if __name__ == "__main__":
    main()
