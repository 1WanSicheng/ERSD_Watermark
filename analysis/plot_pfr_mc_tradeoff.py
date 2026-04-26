from pathlib import Path
import json

import matplotlib.pyplot as plt


ROOT = Path("/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main")
DATA_DIR = Path("/tmp/pfr_mc_sweep_fast_200")
OUT_DIR = ROOT / "analysis" / "plots"


FILES = {
    "pfr_prefix": DATA_DIR / "pfr_prefix.json",
    "mc_uwm_speed": DATA_DIR / "mc_uwm_speed.json",
    "mc_uwm_strength": DATA_DIR / "mc_uwm_strength.json",
}


COLORS = {
    "pfr_prefix": "#1b6ef3",
    "mc_uwm_speed": "#d94841",
    "mc_uwm_strength": "#16803c",
}


LABELS = {
    "pfr_prefix": "PFR Prefix",
    "mc_uwm_speed": "MC UWM Speed",
    "mc_uwm_strength": "MC UWM Strength",
}


def load_results():
    data = {}
    for method, path in FILES.items():
        obj = json.load(open(path))
        data[method] = {}
        for n in [2, 4, 6, 8]:
            summary = obj[str(n)]["summary"][method]
            data[method][n] = summary
    return data


def plot_metric(data, metric_key, ylabel, filename):
    plt.figure(figsize=(7.5, 5.0))
    for method, per_n in data.items():
        xs = sorted(per_n.keys())
        ys = [per_n[n][metric_key] for n in xs]
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=COLORS[method],
            label=LABELS[method],
        )
    plt.xlabel("Draft Length")
    plt.ylabel(ylabel)
    plt.xticks([2, 4, 6, 8])
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def plot_tradeoff(data, metric_key, ylabel, filename):
    plt.figure(figsize=(7.5, 5.0))
    for method, per_n in data.items():
        xs = []
        ys = []
        labels = []
        for n in sorted(per_n.keys()):
            xs.append(per_n[n]["AATPS_mean"])
            ys.append(per_n[n][metric_key])
            labels.append(str(n))
        plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=COLORS[method],
            label=LABELS[method],
        )
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel("AATPS")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    plot_metric(data, "AATPS_mean", "AATPS", "aatps_vs_draft_length.png")
    plot_metric(data, "ANLPPT_U_mean", "ANLPPT-U", "anlppt_u_vs_draft_length.png")
    plot_tradeoff(data, "ANLPPT_U_mean", "ANLPPT-U", "tradeoff_aatps_vs_u.png")
    print("Wrote plots to", OUT_DIR)


if __name__ == "__main__":
    main()
