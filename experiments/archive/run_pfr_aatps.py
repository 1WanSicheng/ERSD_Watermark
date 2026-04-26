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

from accuwm.pfr import pfr_sample_generator
from experiments.archive.run_multi_draft_pfr_aatps import load_model
from experiments.tasks import get_gsm8k_chat_prompts, get_summarization_ds


DEFAULT_TARGET_MODEL = ROOT / "model" / "huggyllama__llama-7b"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "JackFram__llama-68m"


def encode_prompt(tokenizer, prompt, device):
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def run_prompt(target, draft, tokenizer, prompt, lookahead, max_length):
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    gen = pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        private_key=b"1234",
        labeler_mode="prefix",
    )

    block_lens = []
    accepted_counts = []
    draft_lens = []
    generated = 0
    start = time.perf_counter()
    for step_output_ids, _step_output_logprobs, meta in gen:
        block_len = step_output_ids.shape[1]
        if generated + block_len > max_length:
            block_len = max_length - generated
        if block_len <= 0:
            break
        block_lens.append(int(block_len))
        accepted_counts.append(int(meta["accepted_count"]))
        draft_lens.append(int(meta["draft_len"]))
        generated += block_len
        if generated >= max_length:
            break
    elapsed = time.perf_counter() - start

    num_steps = len(block_lens)
    num_tokens = int(sum(block_lens))
    block_efficiency = float(num_tokens / num_steps) if num_steps else 0.0
    token_rate = float(num_tokens / elapsed) if elapsed > 0 else 0.0

    return {
        "num_steps": len(block_lens),
        "num_invocations": num_steps,
        "num_tokens": num_tokens,
        "AATPS": block_efficiency,
        "BE": block_efficiency,
        "block_efficiency": block_efficiency,
        "acceptance_rate": block_efficiency,
        "accepted_draft_mean": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
        "draft_len_mean": float(np.mean(draft_lens)) if draft_lens else 0.0,
        "elapsed_sec": float(elapsed),
        "total_time": float(elapsed),
        "tokens_per_sec": token_rate,
        "token_rate": token_rate,
        "TR": token_rate,
    }


def aggregate(rows):
    be_values = [row["BE"] for row in rows]
    token_rates = [row["token_rate"] for row in rows]
    total_tokens = int(sum(row["num_tokens"] for row in rows))
    total_time = float(sum(row["total_time"] for row in rows))
    global_token_rate = float(total_tokens / total_time) if total_time > 0 else 0.0
    return {
        "num_samples": len(rows),
        "num_tokens_total": total_tokens,
        "AATPS_mean": float(np.mean(be_values)),
        "AATPS_std": float(np.std(be_values)),
        "BE_mean": float(np.mean(be_values)),
        "BE_std": float(np.std(be_values)),
        "block_efficiency_mean": float(np.mean(be_values)),
        "block_efficiency_std": float(np.std(be_values)),
        "acceptance_rate_mean": float(np.mean(be_values)),
        "accepted_draft_mean": float(np.mean([row["accepted_draft_mean"] for row in rows])),
        "draft_len_mean": float(np.mean([row["draft_len_mean"] for row in rows])),
        "num_invocations_mean": float(np.mean([row["num_invocations"] for row in rows])),
        "num_invocations_total": int(sum(row["num_invocations"] for row in rows)),
        "tokens_per_sec_mean": float(np.mean(token_rates)),
        "token_rate_mean": float(np.mean(token_rates)),
        "token_rate_std": float(np.std(token_rates)),
        "TR_mean": float(np.mean(token_rates)),
        "TR_std": float(np.std(token_rates)),
        "TR_global": global_token_rate,
        "elapsed_sec_total": total_time,
        "total_time_total": total_time,
    }


def load_prompts(dataset, samples):
    if dataset == "summarization":
        return [row["prompt"] for row in get_summarization_ds(ds_cut_len=samples)]
    if dataset == "gsm8k":
        return [row["prompt"] for row in get_gsm8k_chat_prompts(ds_cut_len=samples)]
    raise ValueError(f"unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dataset", choices=["summarization", "gsm8k"], default="summarization")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT_MODEL)
    parser.add_argument("--target-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--draft-device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    draft_device = args.draft_device or args.target_device
    target = load_model(args.target_model, args.target_device)
    draft = load_model(args.draft_model, draft_device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.target_model),
        local_files_only=True,
    )

    prompts = load_prompts(args.dataset, args.samples)
    if args.warmup > 0:
        for prompt in prompts[: args.warmup]:
            run_prompt(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                prompt=prompt,
                lookahead=args.lookahead,
                max_length=args.max_length,
            )
    rows = []
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text("", encoding="utf-8")

    for idx, prompt in enumerate(prompts):
        sample_start = time.perf_counter()
        metrics = run_prompt(
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            prompt=prompt,
            lookahead=args.lookahead,
            max_length=args.max_length,
        )
        row = {
            "sample_idx": idx,
            "lookahead": args.lookahead,
            **metrics,
        }
        rows.append(row)
        if args.rows_output is not None:
            with args.rows_output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.progress_every > 0 and (
            (idx + 1) % args.progress_every == 0 or idx + 1 == len(prompts)
        ):
            partial = aggregate(rows)
            print(
                json.dumps(
                    {
                        "progress": {
                            "done": idx + 1,
                            "total": len(prompts),
                            "last_sample_sec": time.perf_counter() - sample_start,
                            "partial_summary": partial,
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "samples": args.samples,
                            "dataset": args.dataset,
                            "warmup": args.warmup,
                            "lookahead": args.lookahead,
                            "max_length": args.max_length,
                            "target_model": str(args.target_model),
                            "draft_model": str(args.draft_model),
                            "target_device": args.target_device,
                            "draft_device": draft_device,
                            "partial": True,
                            "summary": partial,
                            "rows": rows,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    payload = {
        "samples": args.samples,
        "dataset": args.dataset,
        "warmup": args.warmup,
        "lookahead": args.lookahead,
        "max_length": args.max_length,
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "target_device": args.target_device,
        "draft_device": draft_device,
        "summary": aggregate(rows),
        "rows": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
