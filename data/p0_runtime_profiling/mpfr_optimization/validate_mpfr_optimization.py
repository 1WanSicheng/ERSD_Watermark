#!/usr/bin/env python3
"""Record MPFR outputs/block structure and uninstrumented runtime.

The artifact is intended for before/after implementation regression.  Exact
token ids, block lengths, and acceptance metadata are retained so a speed
optimization cannot silently change MPFR's realized sampling path or AATPS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MPFR_spec") not in sys.path:
    sys.path.insert(0, str(ROOT / "MPFR_spec"))

import mpfr_batched_torchgen_cached as mpfr  # noqa: E402
from experiments._shared import (  # noqa: E402
    _maybe_inject_chat_template,
    encode_prompt,
    load_prompts,
)


def sync():
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def run_one(model, draft, input_ids, *, prompt_idx, width, args):
    generator = mpfr.finite_multi_draft_pfr_cached_sample_generator(
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=args.lookahead,
        max_length=args.max_new_tokens,
        num_drafts=width,
        private_key=prompt_idx.to_bytes(8, "big") + b"mpfr-opt-regression",
        process_logits_kwargs={
            "temperature": 1.0,
            "top_k": args.top_k,
            "top_p": 1.0,
        },
        return_meta=True,
        return_logprobs=False,
    )
    chunks = []
    blocks = []
    sync()
    start = time.perf_counter()
    for ids, _logprobs, meta in generator:
        token_list = [int(token) for token in ids[0].detach().cpu().tolist()]
        chunks.extend(token_list)
        blocks.append({
            "tokens": token_list,
            "accepted_count": int(meta["accepted_count"]),
            "draft_tree_size": int(meta["draft_tree_size"]),
            "target_context_count": int(meta["target_context_count"]),
        })
    sync()
    elapsed = time.perf_counter() - start
    return {
        "prompt_idx": prompt_idx,
        "width": width,
        "output_ids": chunks,
        "blocks": blocks,
        "tokens": len(chunks),
        "num_blocks": len(blocks),
        "AATPS": len(chunks) / max(len(blocks), 1),
        "elapsed_sec": elapsed,
        "token_rate": len(chunks) / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    _maybe_inject_chat_template(tokenizer, args.target)
    model = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float16, device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft, torch_dtype=torch.float16, device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    prompts = load_prompts("cnn_dailymail", args.samples)

    # Warm model kernels and allocator outside the timed rows.
    warm_ids = encode_prompt(tokenizer, prompts[0], args.device)
    run_one(model, draft, warm_ids, prompt_idx=10_000, width=args.widths[0], args=args)

    rows = []
    for width in args.widths:
        for prompt_idx, prompt in enumerate(prompts):
            input_ids = encode_prompt(tokenizer, prompt, args.device)
            row = run_one(
                model, draft, input_ids, prompt_idx=prompt_idx,
                width=width, args=args,
            )
            rows.append(row)
            print(
                f"B={width} prompt={prompt_idx} AATPS={row['AATPS']:.3f} "
                f"TR={row['token_rate']:.2f}", flush=True,
            )

    summary = {}
    for width in args.widths:
        group = [row for row in rows if row["width"] == width]
        tokens = sum(row["tokens"] for row in group)
        blocks = sum(row["num_blocks"] for row in group)
        elapsed = sum(row["elapsed_sec"] for row in group)
        summary[str(width)] = {
            "tokens": tokens,
            "blocks": blocks,
            "AATPS": tokens / blocks,
            "token_rate": tokens / elapsed,
            "ms_per_block": 1000.0 * elapsed / blocks,
        }

    payload = {
        "config": vars(args) | {"output": str(args.output)},
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
