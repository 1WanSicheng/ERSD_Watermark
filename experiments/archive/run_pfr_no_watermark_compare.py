import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import accuwm.pfr as pfr
import accuwm.pfr_no_watermark as pfr_no_watermark
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


def summarize(block_lens, accepted_counts, elapsed):
    return {
        "num_steps": len(block_lens),
        "num_tokens": int(sum(block_lens)),
        "AATPS": float(np.mean(block_lens)) if block_lens else 0.0,
        "accepted_draft_mean": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
        "tokens_per_sec": float(sum(block_lens) / elapsed) if elapsed > 0 else 0.0,
    }


def run_method(target, draft, tokenizer, prompt, method, n, max_length):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(target.device)
    if method == "pfr_no_watermark":
        gen = pfr_no_watermark.pfr_no_watermark_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=n,
            max_length=max_length,
        )
    elif method == "pfr_prefix":
        gen = pfr.pfr_sample_generator(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=n,
            max_length=max_length,
            private_key=b"1234",
            labeler_mode="prefix",
        )
    else:
        raise ValueError(method)

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
        accepted_counts.append(int(meta["accepted_count"]))
        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - t0
    return summarize(block_lens, accepted_counts, elapsed)


def aggregate(rows, method):
    method_rows = [row for row in rows if row["method"] == method]
    return {
        "num_samples": len(method_rows),
        "AATPS_mean": float(np.mean([row["AATPS"] for row in method_rows])),
        "AATPS_std": float(np.std([row["AATPS"] for row in method_rows])),
        "accepted_draft_mean": float(
            np.mean([row["accepted_draft_mean"] for row in method_rows])
        ),
        "tokens_per_sec_mean": float(np.mean([row["tokens_per_sec"] for row in method_rows])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--draft-lengths", type=int, nargs="*", default=[2, 4, 6, 8])
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["pfr_no_watermark", "pfr_prefix"],
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = load_model(TARGET_MODEL, device)
    draft = load_model(DRAFT_MODEL, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(TARGET_MODEL))
    prompts = [row["prompt"] for row in get_summarization_ds(ds_cut_len=args.samples)]

    results = {}
    for n in args.draft_lengths:
        rows = []
        for prompt in prompts:
            for method in args.methods:
                rows.append(
                    {
                        "method": method,
                        "n": n,
                        **run_method(target, draft, tokenizer, prompt, method, n, args.max_length),
                    }
                )
        results[str(n)] = {
            "summary": {method: aggregate(rows, method) for method in args.methods},
            "rows": rows,
        }

    print(
        json.dumps(
            {
                "samples": args.samples,
                "draft_lengths": args.draft_lengths,
                "max_length": args.max_length,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
