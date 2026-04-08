import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from unbiased_watermark.lm import get_rng


PRIVATE_KEY = b"ersd_wm_race_detect"
VOCAB_SIZE = 10
SEQ_LENGTHS = [20, 50, 80, 100, 150, 200]
NUM_TRIALS = 800
TARGET_FPR = 0.01


@dataclass(frozen=True)
class Regime:
    name: str
    probs: np.ndarray


def _build_regimes():
    low = np.array([0.60, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03])
    medium = np.array([0.10, 0.13, 0.155, 0.115, 0.235, 0.065, 0.055, 0.05, 0.06, 0.035])
    high_rng = np.random.default_rng(20260408)
    high = high_rng.dirichlet(np.full(VOCAB_SIZE, 10.0))
    return [
        Regime("low", low / low.sum()),
        Regime("medium", medium / medium.sum()),
        Regime("high", high / high.sum()),
    ]


def _seed_bytes(trial_idx: int, pos_idx: int) -> tuple[bytes, bytes, bytes]:
    trial_b = trial_idx.to_bytes(8, byteorder="big", signed=False)
    pos_b = pos_idx.to_bytes(8, byteorder="big", signed=False)
    tag_b = b"race"
    return trial_b, pos_b, tag_b


def _sample_u_vector(trial_idx: int, pos_idx: int) -> np.ndarray:
    trial_b, pos_b, tag_b = _seed_bytes(trial_idx, pos_idx)
    rng = get_rng(trial_b, pos_b, tag_b, PRIVATE_KEY)
    u = np.array([rng.random() for _ in range(VOCAB_SIZE)], dtype=np.float64)
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def _race_from_u(probs: np.ndarray, u: np.ndarray):
    arrival = -np.log(u) / probs
    order = np.argsort(arrival)
    winner = int(order[0])
    second = int(order[1])
    t1 = float(arrival[winner])
    t2 = float(arrival[second])
    return arrival, winner, second, t1, t2


def _token_stats(observed_token: int, winner: int, t1: float, t2: float, u_obs: float):
    # Existing ERSD_Aaronson implementation uses -log(1 - U_y).
    z = float(-np.log(1.0 - u_obs))
    ratio_raw = float(t1 / t2)
    gap_raw = float(t2 - t1)
    is_winner = int(observed_token == winner)
    # Winner-conditioned variants are the usable keyed race statistics.
    ratio_win = float(-np.log(ratio_raw)) if is_winner else 0.0
    gap_win = float(gap_raw) if is_winner else 0.0
    return {
        "z": z,
        "ratio_raw": ratio_raw,
        "gap_raw": gap_raw,
        "winner_match": is_winner,
        "ratio_win": ratio_win,
        "gap_win": gap_win,
    }


def _simulate_scores(regime: Regime, seq_len: int):
    h0 = {k: [] for k in ["aaronson", "ratio_raw", "gap_raw", "match", "ratio_win", "gap_win", "joint_ratio", "joint_gap"]}
    h1 = {k: [] for k in ["aaronson", "ratio_raw", "gap_raw", "match", "ratio_win", "gap_win", "joint_ratio", "joint_gap"]}

    h0_token_ratio = []
    h1_token_ratio = []
    h0_token_gap = []
    h1_token_gap = []
    h1_token_z = []

    null_rng = np.random.default_rng(20260408 + seq_len + int(regime.probs[0] * 1000))

    for trial_idx in range(NUM_TRIALS):
        accum0 = {k: 0.0 for k in h0}
        accum1 = {k: 0.0 for k in h1}
        for pos_idx in range(seq_len):
            u = _sample_u_vector(trial_idx, pos_idx)
            _, winner, _second, t1, t2 = _race_from_u(regime.probs, u)

            observed_h1 = winner
            stats_h1 = _token_stats(observed_h1, winner, t1, t2, u[observed_h1])
            observed_h0 = int(null_rng.choice(VOCAB_SIZE, p=regime.probs))
            stats_h0 = _token_stats(observed_h0, winner, t1, t2, u[observed_h0])

            accum1["aaronson"] += stats_h1["z"]
            accum1["ratio_raw"] += -np.log(stats_h1["ratio_raw"])
            accum1["gap_raw"] += stats_h1["gap_raw"]
            accum1["match"] += stats_h1["winner_match"]
            accum1["ratio_win"] += stats_h1["ratio_win"]
            accum1["gap_win"] += stats_h1["gap_win"]
            accum1["joint_ratio"] += stats_h1["z"] + stats_h1["ratio_win"]
            accum1["joint_gap"] += stats_h1["z"] + stats_h1["gap_win"]

            accum0["aaronson"] += stats_h0["z"]
            accum0["ratio_raw"] += -np.log(stats_h0["ratio_raw"])
            accum0["gap_raw"] += stats_h0["gap_raw"]
            accum0["match"] += stats_h0["winner_match"]
            accum0["ratio_win"] += stats_h0["ratio_win"]
            accum0["gap_win"] += stats_h0["gap_win"]
            accum0["joint_ratio"] += stats_h0["z"] + stats_h0["ratio_win"]
            accum0["joint_gap"] += stats_h0["z"] + stats_h0["gap_win"]

            if trial_idx < 1000:
                h0_token_ratio.append(stats_h0["ratio_win"])
                h1_token_ratio.append(stats_h1["ratio_win"])
                h0_token_gap.append(stats_h0["gap_win"])
                h1_token_gap.append(stats_h1["gap_win"])
                h1_token_z.append(stats_h1["z"])

        for key in h0:
            h0[key].append(accum0[key])
            h1[key].append(accum1[key])

    token_diag = {
        "h0_ratio": np.array(h0_token_ratio, dtype=np.float64),
        "h1_ratio": np.array(h1_token_ratio, dtype=np.float64),
        "h0_gap": np.array(h0_token_gap, dtype=np.float64),
        "h1_gap": np.array(h1_token_gap, dtype=np.float64),
        "h1_z": np.array(h1_token_z, dtype=np.float64),
    }
    return {k: np.array(v, dtype=np.float64) for k, v in h0.items()}, {k: np.array(v, dtype=np.float64) for k, v in h1.items()}, token_diag


def _tpr_at_fpr(h0_scores: np.ndarray, h1_scores: np.ndarray, fpr: float) -> tuple[float, float]:
    threshold = float(np.quantile(h0_scores, 1.0 - fpr))
    tpr = float(np.mean(h1_scores > threshold))
    return threshold, tpr


def _cohens_d(h0_vals: np.ndarray, h1_vals: np.ndarray) -> float:
    var = 0.5 * (np.var(h0_vals) + np.var(h1_vals))
    if var <= 0:
        return 0.0
    return float((np.mean(h1_vals) - np.mean(h0_vals)) / np.sqrt(var))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _plot_tpr(rows, out_path):
    regimes = sorted({r["regime"] for r in rows})
    metrics = ["aaronson", "ratio_win", "gap_win", "joint_ratio", "joint_gap"]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 4), sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = [r for r in rows if r["regime"] == regime]
        for metric in metrics:
            xs = [r["seq_len"] for r in sub if r["metric"] == metric]
            ys = [r["tpr"] for r in sub if r["metric"] == metric]
            ax.plot(xs, ys, marker="o", label=metric)
        ax.set_title(f"{regime} entropy")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("TPR @ FPR=1%")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def _plot_token_hist(token_diag_map, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    diag = token_diag_map["medium"]
    axes[0].hist(diag["h0_ratio"], bins=40, alpha=0.5, density=True, label="H0")
    axes[0].hist(diag["h1_ratio"], bins=40, alpha=0.5, density=True, label="H1")
    axes[0].set_title("Winner-conditioned ratio score")
    axes[0].set_xlabel("-log(T1 / T2) gated by winner match")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].hist(diag["h0_gap"], bins=40, alpha=0.5, density=True, label="H0")
    axes[1].hist(diag["h1_gap"], bins=40, alpha=0.5, density=True, label="H1")
    axes[1].set_title("Winner-conditioned gap score")
    axes[1].set_xlabel("T2 - T1 gated by winner match")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def _write_summary(rows, token_diag_map, out_path):
    lines = [
        "ERSD_WM race detector validation",
        "Note: raw ratio/gap are included as sanity checks; only winner-conditioned variants are token-aware detectors.",
        "",
        "regime\tseq_len\tmetric\tthreshold\ttpr\th0_mean\th1_mean",
    ]
    for row in rows:
        lines.append(
            f"{row['regime']}\t{row['seq_len']}\t{row['metric']}\t{row['threshold']:.6f}\t"
            f"{row['tpr']:.6f}\t{row['h0_mean']:.6f}\t{row['h1_mean']:.6f}"
        )
    lines.append("")
    lines.append("token_level_diagnostics")
    lines.append("regime\tmetric\tcohens_d")
    for regime, diag in token_diag_map.items():
        lines.append(f"{regime}\tratio_win\t{_cohens_d(diag['h0_ratio'], diag['h1_ratio']):.6f}")
        lines.append(f"{regime}\tgap_win\t{_cohens_d(diag['h0_gap'], diag['h1_gap']):.6f}")
        lines.append(f"{regime}\tcorr_z_ratio_h1\t{_safe_corr(diag['h1_z'], diag['h1_ratio']):.6f}")
        lines.append(f"{regime}\tcorr_z_gap_h1\t{_safe_corr(diag['h1_z'], diag['h1_gap']):.6f}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    log_dir = os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    rows = []
    token_diag_map = {}
    for regime in _build_regimes():
        for seq_len in SEQ_LENGTHS:
            h0_scores, h1_scores, token_diag = _simulate_scores(regime, seq_len)
            if seq_len == 100:
                token_diag_map[regime.name] = token_diag
            for metric, h0 in h0_scores.items():
                h1 = h1_scores[metric]
                threshold, tpr = _tpr_at_fpr(h0, h1, TARGET_FPR)
                rows.append(
                    {
                        "regime": regime.name,
                        "seq_len": seq_len,
                        "metric": metric,
                        "threshold": threshold,
                        "tpr": tpr,
                        "h0_mean": float(np.mean(h0)),
                        "h1_mean": float(np.mean(h1)),
                    }
                )

    _write_summary(rows, token_diag_map, os.path.join(log_dir, "ersd_wm_race_detect_summary.txt"))
    _plot_tpr(rows, os.path.join(log_dir, "ersd_wm_race_detect_tpr.png"))
    _plot_token_hist(token_diag_map, os.path.join(log_dir, "ersd_wm_race_detect_token_hist.png"))


if __name__ == "__main__":
    main()
