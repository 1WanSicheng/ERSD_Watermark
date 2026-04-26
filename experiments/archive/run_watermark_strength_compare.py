import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import accuwm
import accuwm.pfr as pfr
import unbiased_watermark as uwm
from unbiased_watermark.scores.pfr_aaronson import (
    PFR_Aaronson_Gamma_Score,
    PFR_Aaronson_U_Score,
)
from experiments.tasks import get_summarization_ds
from experiments.worker import Worker


TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


def summarize_step_lengths(step_lens):
    if not step_lens:
        return 0.0
    return float(np.mean(step_lens))


def normalize_meta_array(arr, block_len, dtype=None):
    out = np.array(arr, dtype=dtype if dtype is not None else object)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(1, -1)
    return out[:, :block_len]


def run_worker_method(worker, prompt, method, n, max_length, seed=1):
    d = {
        "prompt": prompt,
        "seed": seed,
        "method": method,
        "reweight": "deltagumbel" if method in ["basic_uwm", "mc_uwm_speed", "mc_uwm_strength"] else "ersd",
        "private_key": b"1234",
        "n": n,
        "max_length": max_length,
    }
    out = worker.process(d)
    row = {
        "method": method,
        "num_tokens": int(sum(out["gen_seq_lens"])),
        "num_steps": int(len(out["gen_seq_lens"])),
        "AATPS": summarize_step_lengths(out["gen_seq_lens"]),
    }
    if method in ["basic_uwm", "mc_uwm_speed", "mc_uwm_strength"]:
        row["detector_u"] = "DeltaGumbel_U"
        row["ANLPPT_U"] = float(-out["log_p_values"][1] / max(row["num_tokens"], 1))
        row["detector_aux"] = "RobustLLR"
        row["ANLPPT_Aux"] = float(-out["log_p_values"][3] / max(row["num_tokens"], 1))
    else:
        raise ValueError(method)
    return row


def run_pfr_method(worker, prompt, method, n, max_length):
    model = worker.model
    ref_model = worker.ref_model
    tokenizer = worker.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    labeler_mode = "prefix" if method == "pfr_prefix" else "context_code"
    gen = pfr.pfr_sample_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        n=n,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode=labeler_mode,
    )

    output_ids = []
    block_lens = []
    source_labels = []
    time_steps = []
    skipped = []
    current_pos = input_ids.shape[1]

    for step_output_ids, _step_output_logprobs, meta in gen:
        block_len = step_output_ids.shape[1]
        remaining = max_length - len(output_ids)
        if remaining <= 0:
            break
        if block_len > remaining:
            step_output_ids = step_output_ids[:, :remaining]
            block_len = remaining
        output_ids.extend(step_output_ids[0].tolist())
        block_lens.append(block_len)
        labels = meta["labels"][:block_len]
        source_labels.extend([label.source_label for label in labels])
        skipped.extend([False] * block_len)
        time_steps.extend(list(range(current_pos + 1, current_pos + block_len + 1)))
        current_pos += block_len
        if len(output_ids) >= max_length:
            break

    ids_tensor = torch.tensor(output_ids, device=model.device, dtype=torch.long).unsqueeze(0)
    source_labels_np = np.array(source_labels, dtype=object)[None, :]
    time_steps_np = np.array(time_steps, dtype=np.int64)[None, :]
    skipped_np = np.array(skipped, dtype=bool)[None, :]

    gamma_score = PFR_Aaronson_Gamma_Score.from_watermarkcode(
        None,
        ids_tensor,
        skipped=skipped_np,
        source_labels=source_labels_np,
        time_steps=time_steps_np,
        private_key=b"1234",
        vocab_size=model.config.vocab_size,
    )
    u_score = PFR_Aaronson_U_Score(
        scores=gamma_score.scores,
        skipped=gamma_score.skipped,
    )
    num_tokens = len(output_ids)
    return {
        "method": method,
        "num_tokens": num_tokens,
        "num_steps": int(len(block_lens)),
        "AATPS": summarize_step_lengths(block_lens),
        "detector_u": "PFR_Aaronson_U",
        "ANLPPT_U": float(-u_score.get_log_p_value() / max(num_tokens, 1)),
        "detector_aux": "PFR_Aaronson_Gamma",
        "ANLPPT_Aux": float(-gamma_score.get_log_p_value() / max(num_tokens, 1)),
    }


def aggregate(rows, method):
    method_rows = [row for row in rows if row["method"] == method]
    return {
        "num_samples": len(method_rows),
        "AATPS_mean": float(np.mean([row["AATPS"] for row in method_rows])),
        "AATPS_std": float(np.std([row["AATPS"] for row in method_rows])),
        "ANLPPT_U_mean": float(np.mean([row["ANLPPT_U"] for row in method_rows])),
        "ANLPPT_U_std": float(np.std([row["ANLPPT_U"] for row in method_rows])),
        "ANLPPT_Aux_mean": float(np.mean([row["ANLPPT_Aux"] for row in method_rows])),
        "ANLPPT_Aux_std": float(np.std([row["ANLPPT_Aux"] for row in method_rows])),
        "detector_u": method_rows[0]["detector_u"],
        "detector_aux": method_rows[0]["detector_aux"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=[
            "basic_uwm",
            "mc_uwm_speed",
            "mc_uwm_strength",
            "pfr_prefix",
            "pfr_cc",
        ],
    )
    args = parser.parse_args()

    worker = Worker(
        {
            "model_str": str(TARGET_MODEL),
            "ref_model_str": str(DRAFT_MODEL),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.0,
            "assert_cch": False,
            "assert_log_p_values": False,
        }
    )

    ds = get_summarization_ds(ds_cut_len=args.samples)
    prompts = [row["prompt"] for row in ds]

    rows = []
    t0 = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        for method in args.methods:
            if method.startswith("pfr_"):
                row = run_pfr_method(worker, prompt, method, args.n, args.max_length)
            else:
                row = run_worker_method(
                    worker,
                    prompt,
                    method,
                    args.n,
                    args.max_length,
                    seed=args.seed,
                )
            row["sample_idx"] = idx
            rows.append(row)
    elapsed = time.perf_counter() - t0

    summary = {method: aggregate(rows, method) for method in args.methods}
    out = {
        "samples": args.samples,
        "n": args.n,
        "max_length": args.max_length,
        "elapsed_sec": elapsed,
        "summary": summary,
        "rows": rows,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
