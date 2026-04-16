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

import accuwm.pfr as pfr
import unbiased_watermark as uwm
from unbiased_watermark.scores.pfr_aaronson import (
    compute_pfr_aaronson_from_sequence,
)
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


def run_pfr_sequence_score(target, draft, tokenizer, prompt, method, n, max_length):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(target.device)
    labeler_mode = "prefix" if method == "pfr_prefix" else "context_code"
    gen = pfr.pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=n,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode=labeler_mode,
    )

    output_ids = []
    step_lens = []
    t0 = time.perf_counter()
    for step_output_ids, _step_output_logprobs, _meta in gen:
        block_len = step_output_ids.shape[1]
        remaining = max_length - len(output_ids)
        if remaining <= 0:
            break
        if block_len > remaining:
            step_output_ids = step_output_ids[:, :remaining]
            block_len = remaining
        output_ids.extend(step_output_ids[0].tolist())
        step_lens.append(block_len)
        if len(output_ids) >= max_length:
            break
    elapsed = time.perf_counter() - t0

    full_ids = torch.cat(
        [
            input_ids,
            torch.tensor(output_ids, device=input_ids.device, dtype=torch.long).unsqueeze(0),
        ],
        dim=1,
    )
    detect_labeler = pfr.build_default_labeler(
        mode=labeler_mode,
        cc_extractor=uwm.lm.PrevN_ContextCodeExtractor(n=3),
    )
    score = compute_pfr_aaronson_from_sequence(
        full_ids=full_ids,
        prompt_length=input_ids.shape[1],
        labeler=detect_labeler,
        private_key=b"1234",
        vocab_size=target.config.vocab_size,
    )
    return {
        "method": method,
        "num_tokens": len(output_ids),
        "num_steps": len(step_lens),
        "AATPS": float(np.mean(step_lens)) if step_lens else 0.0,
        "tokens_per_sec": float(len(output_ids) / elapsed) if elapsed > 0 else 0.0,
        **score,
    }


def aggregate(rows, method):
    method_rows = [row for row in rows if row["method"] == method]
    keys = [
        "AATPS",
        "ANLPPT_Aaronson",
        "score_mean",
        "masked_ratio",
        "tokens_per_sec",
    ]
    summary = {"num_samples": len(method_rows)}
    for key in keys:
        vals = [row[key] for row in method_rows]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["pfr_prefix", "pfr_cc"],
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target = load_model(TARGET_MODEL, device)
    draft = load_model(DRAFT_MODEL, device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(TARGET_MODEL))
    prompts = [row["prompt"] for row in get_summarization_ds(ds_cut_len=args.samples)]

    rows = []
    t0 = time.perf_counter()
    for sample_idx, prompt in enumerate(prompts):
        for method in args.methods:
            row = run_pfr_sequence_score(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                prompt=prompt,
                method=method,
                n=args.n,
                max_length=args.max_length,
            )
            row["sample_idx"] = sample_idx
            rows.append(row)
    elapsed = time.perf_counter() - t0

    summary = {method: aggregate(rows, method) for method in args.methods}
    print(
        json.dumps(
            {
                "samples": args.samples,
                "n": args.n,
                "max_length": args.max_length,
                "elapsed_sec": elapsed,
                "detector": "PFR_Aaronson_sequence_no_tag",
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
