import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import accuwm
import accuwm.pfr as pfr
import unbiased_watermark as uwm
from experiments.tasks import get_summarization_ds


TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


def load_model(model_path: Path, device: str):
    kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "device_map": device,
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    return transformers.AutoModelForCausalLM.from_pretrained(**kwargs)


def summarize_blocks(block_lens, accepted_counts=None, masked_flags=None, elapsed=None):
    out = {
        "num_steps": len(block_lens),
        "num_tokens": int(sum(block_lens)),
        "AATPS": float(np.mean(block_lens)) if block_lens else 0.0,
    }
    if accepted_counts is not None and accepted_counts:
        out["accepted_draft_mean"] = float(np.mean(accepted_counts))
    if masked_flags is not None and masked_flags:
        vals = [x for arr in masked_flags for x in arr]
        out["masked_ratio"] = float(np.mean(vals)) if vals else 0.0
    if elapsed is not None:
        out["elapsed_sec"] = float(elapsed)
        out["tokens_per_sec"] = float(sum(block_lens) / elapsed) if elapsed > 0 else 0.0
    return out


def run_ersd_method(target, draft, tokenizer, prompt, method_name, lookahead=2, max_length=32):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(target.device)
    kwargs = {
        "model": target,
        "ref_model": draft,
        "input_ids": input_ids,
        "n": lookahead,
        "return_meta": True,
        "process_logits_kwargs": {},
    }
    if method_name == "ersd":
        kwargs["seed_source"] = "prefix"
    elif method_name == "ersd_cc":
        kwargs["seed_source"] = "cc"
        kwargs["cc_extractor"] = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    else:
        raise ValueError(method_name)

    gen = accuwm.ersd.ersd_sample_generator(**kwargs)
    block_lens = []
    accepted_counts = []
    generated = 0
    t0 = time.perf_counter()
    for step_output_ids, _step_output_logprobs, meta in gen:
        block_len = step_output_ids.shape[1]
        if generated + block_len > max_length:
            take = max_length - generated
            if take <= 0:
                break
            block_len = take
        block_lens.append(block_len)
        accepted_counts.append(int(meta["accepted_draft_len"]))
        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - t0
    return summarize_blocks(block_lens, accepted_counts=accepted_counts, elapsed=elapsed)


def run_pfr_method(target, draft, tokenizer, prompt, method_name, lookahead=2, max_length=32):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(target.device)
    labeler_mode = "prefix" if method_name == "pfr_prefix" else "context_code"
    gen = pfr.pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode=labeler_mode,
    )
    block_lens = []
    accepted_counts = []
    masked_flags = []
    generated = 0
    t0 = time.perf_counter()
    for step_output_ids, _step_output_logprobs, meta in gen:
        block_len = step_output_ids.shape[1]
        if generated + block_len > max_length:
            take = max_length - generated
            if take <= 0:
                break
            block_len = take
        block_lens.append(block_len)
        accepted_counts.append(int(meta["accepted_count"]))
        masked_flags.append([label.masked for label in meta["labels"][:block_len]])
        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - t0
    return summarize_blocks(
        block_lens,
        accepted_counts=accepted_counts,
        masked_flags=masked_flags,
        elapsed=elapsed,
    )


def aggregate(rows, method):
    method_rows = [row for row in rows if row["method"] == method]
    summary = {
        "num_samples": len(method_rows),
        "AATPS_mean": float(np.mean([row["AATPS"] for row in method_rows])),
        "AATPS_std": float(np.std([row["AATPS"] for row in method_rows])),
        "accepted_draft_mean": float(
            np.mean([row["accepted_draft_mean"] for row in method_rows])
        ),
        "tokens_per_sec_mean": float(
            np.mean([row["tokens_per_sec"] for row in method_rows])
        ),
    }
    masked_vals = [row.get("masked_ratio") for row in method_rows if "masked_ratio" in row]
    if masked_vals:
        summary["masked_ratio_mean"] = float(np.mean(masked_vals))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = load_model(TARGET_MODEL, device)
    draft = load_model(DRAFT_MODEL, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(TARGET_MODEL))

    ds = get_summarization_ds(ds_cut_len=args.samples)
    prompts = [row["prompt"] for row in ds]
    methods = ["ersd", "ersd_cc", "pfr_prefix", "pfr_cc"]

    rows = []
    for idx, prompt in enumerate(prompts):
        for method in methods:
            if method.startswith("pfr"):
                metrics = run_pfr_method(
                    target, draft, tokenizer, prompt, method, args.lookahead, args.max_length
                )
            else:
                metrics = run_ersd_method(
                    target, draft, tokenizer, prompt, method, args.lookahead, args.max_length
                )
            rows.append(
                {
                    "sample_idx": idx,
                    "method": method,
                    **metrics,
                }
            )

    summary = {method: aggregate(rows, method) for method in methods}
    print(
        json.dumps(
            {
                "device": device,
                "samples": args.samples,
                "lookahead": args.lookahead,
                "max_length": args.max_length,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
