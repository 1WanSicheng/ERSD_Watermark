import gc
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.tasks import get_summarization_ds
from experiments.worker import Worker


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "logs")

MODEL_STR = "/mnt/workspace0/A24738/model-weights/Qwen2.5-7B-Instruct"
REF_MODEL_STR = "/mnt/workspace0/A24738/model-weights/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0"
PRIVATE_KEY = b"1234"

DATASET_SIZE = int(os.environ.get("ERSD_TEMP_DATASET_SIZE", "100"))
MAX_LENGTH = int(os.environ.get("ERSD_TEMP_MAX_LENGTH", "32"))
N = int(os.environ.get("ERSD_TEMP_N", "4"))
SEED = int(os.environ.get("ERSD_TEMP_SEED", "1"))
TOP_K = int(os.environ.get("ERSD_TEMP_TOP_K", "0"))
TOP_P = float(os.environ.get("ERSD_TEMP_TOP_P", "1.0"))
TEMPERATURES = [
    float(x) for x in os.environ.get("ERSD_TEMP_LIST", "0.7,1.3").split(",") if x
]
LOG_P_THRESHOLD = math.log(0.01)


@dataclass
class RunResult:
    temperature: float
    gamma_tpr: float
    u_tpr: float
    gamma_mean_logp: float
    u_mean_logp: float
    mean_aatps: float
    mean_accept_len: float
    mean_verify_ops: float
    rows: list[dict]


def _load_prompts(limit: int) -> list[dict]:
    ds = get_summarization_ds(limit)
    return [{"idx": d["idx"], "prompt": d["prompt"]} for d in ds]


def _make_worker(temperature: float) -> Worker:
    return Worker(
        param={
            "model_str": MODEL_STR,
            "ref_model_str": REF_MODEL_STR,
            "task": "summarization_scan_n",
            "device": DEVICE,
            "print_output": False,
            "assert_cch": True,
            "assert_log_p_values": True,
            "temperature": temperature,
            "top_k": TOP_K,
            "top_p": TOP_P,
        }
    )


def _aatps(row: dict) -> float:
    gen_lens = row["gen_seq_lens"]
    if not gen_lens:
        return float("nan")
    gen_tokens = sum(gen_lens[1:])
    denom = max(1, len(gen_lens))
    return gen_tokens / denom


def _mean_accept_len(row: dict) -> float:
    vals = [v for v in row.get("accepted_draft_lens", []) if v is not None]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _run_temperature(prompts: list[dict], temperature: float) -> RunResult:
    worker = _make_worker(temperature)
    rows = []
    for prompt_row in prompts:
        row = worker.process(
            {
                **prompt_row,
                "seed": SEED,
                "method": "ersd_wm",
                "reweight": "ersd",
                "private_key": PRIVATE_KEY,
                "n": N,
                "max_length": MAX_LENGTH,
            }
        )
        rows.append(row)

    del worker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gamma_logp = np.array([row["log_p_values"][0] for row in rows], dtype=np.float64)
    u_logp = np.array([row["log_p_values"][1] for row in rows], dtype=np.float64)
    aatps = np.array([_aatps(row) for row in rows], dtype=np.float64)
    accept_len = np.array([_mean_accept_len(row) for row in rows], dtype=np.float64)
    verify_ops = np.array([row["verify_ops"] for row in rows], dtype=np.float64)

    return RunResult(
        temperature=temperature,
        gamma_tpr=float(np.mean(gamma_logp <= LOG_P_THRESHOLD)),
        u_tpr=float(np.mean(u_logp <= LOG_P_THRESHOLD)),
        gamma_mean_logp=float(np.mean(gamma_logp)),
        u_mean_logp=float(np.mean(u_logp)),
        mean_aatps=float(np.nanmean(aatps)),
        mean_accept_len=float(np.nanmean(accept_len)),
        mean_verify_ops=float(np.nanmean(verify_ops)),
        rows=rows,
    )


def _write_summary(results: list[RunResult], out_path: str):
    lines = [
        "ERSD_WM temperature invariance experiment",
        f"dataset_size={DATASET_SIZE}",
        f"n={N}",
        f"max_length={MAX_LENGTH}",
        f"log_p_threshold={LOG_P_THRESHOLD:.6f} (p=0.01)",
        "",
        "temperature\tgamma_tpr\tu_tpr\tgamma_mean_logp\tu_mean_logp\tmean_aatps\tmean_accept_len\tmean_verify_ops",
    ]
    for result in results:
        lines.append(
            f"{result.temperature:.2f}\t{result.gamma_tpr:.6f}\t{result.u_tpr:.6f}\t"
            f"{result.gamma_mean_logp:.6f}\t{result.u_mean_logp:.6f}\t"
            f"{result.mean_aatps:.6f}\t{result.mean_accept_len:.6f}\t{result.mean_verify_ops:.6f}"
        )
    if len(results) == 2:
        lines.extend(
            [
                "",
                f"delta_mean_aatps\t{results[1].mean_aatps - results[0].mean_aatps:.6f}",
                f"delta_mean_accept_len\t{results[1].mean_accept_len - results[0].mean_accept_len:.6f}",
                f"delta_mean_verify_ops\t{results[1].mean_verify_ops - results[0].mean_verify_ops:.6f}",
                f"delta_gamma_tpr\t{results[1].gamma_tpr - results[0].gamma_tpr:.6f}",
                f"delta_u_tpr\t{results[1].u_tpr - results[0].u_tpr:.6f}",
            ]
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_tpr(results: list[RunResult], out_path: str):
    temps = [r.temperature for r in results]
    gamma = [r.gamma_tpr for r in results]
    u = [r.u_tpr for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(temps, gamma, marker="o", label="ERSD_Aaronson_Gamma TPR")
    ax.plot(temps, u, marker="s", label="ERSD_Aaronson_U TPR")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("TPR @ p < 0.01")
    ax.set_title("ERSD_WM Draft-Temperature Invariance")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    prompts = _load_prompts(DATASET_SIZE)
    results = []
    for temperature in TEMPERATURES:
        results.append(_run_temperature(prompts, temperature))
    _write_summary(
        results,
        os.path.join(LOG_DIR, "ersd_wm_temp_invariance_summary.txt"),
    )
    _plot_tpr(
        results,
        os.path.join(LOG_DIR, "ersd_wm_temp_invariance_tpr.png"),
    )


if __name__ == "__main__":
    main()
