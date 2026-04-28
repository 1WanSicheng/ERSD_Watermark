"""
Benchmark the two cache-aware batched MPFR variants side by side on the same
prompts.  Both share the same architecture (draft tree + ONE batched target
forward + KV cache reuse across blocks).  The only difference is the MPFR
sampling primitive:

  ms_pfr_batched_cached:
    accuwm.multi_draft_pfr.ms_pfr_tokens_from_logprobs (per-context
    torch.Generator on GPU).  Top-k applied via accuwm.utils.process_logits
    with a logits_warper.

  mpfr_batched_torchgen_cached:
    MPFR_spec.mpfr_direct_optimized.process_logits_exact (top-k inside the
    sampler) + the same torch.Generator GPU primitive.  RNG layout matches
    the original mpfr_direct_optimized blake2b primitive.

Reports AATPS (tokens/block) and token_rate (tok/s) per (method, B).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MPFR_spec"))

from multi_draft_pfr_batched_cached import (
    multi_draft_pfr_batched_cached_sample_generator as ms_pfr_cached_gen,
)
from mpfr_batched_torchgen_cached import (
    finite_multi_draft_pfr_cached_sample_generator as torchgen_cached_gen,
)


def load_prompts(n: int):
    from datasets import load_dataset
    dset = load_dataset("openai/gsm8k", "main")["train"]
    out = []
    for i in range(n):
        question = dset[i]["question"]
        out.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ])
    return out


def encode_prompt(tokenizer, prompt, device):
    return tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_logits_warper(top_k, top_p, temperature):
    from transformers import (
        LogitsProcessorList, TemperatureLogitsWarper,
        TopKLogitsWarper, TopPLogitsWarper,
    )
    parts = []
    if temperature is not None and float(temperature) != 1.0:
        parts.append(TemperatureLogitsWarper(float(temperature)))
    if top_k is not None and int(top_k) > 0:
        parts.append(TopKLogitsWarper(int(top_k)))
    if top_p is not None and 0 < float(top_p) < 1.0:
        parts.append(TopPLogitsWarper(float(top_p)))
    if not parts:
        return None
    warper = LogitsProcessorList(parts)
    def warp(input_ids, logits):
        return warper(input_ids, logits)
    return warp


METHODS = {
    "ms_pfr_batched_cached": {
        "fn": ms_pfr_cached_gen,
        # accuwm.utils.process_logits expects a logits_warper callable.
        "use_warper": True,
    },
    "mpfr_batched_torchgen_cached": {
        "fn": torchgen_cached_gen,
        # MPFR_spec.process_logits_exact takes raw scalars.
        "use_warper": False,
    },
}


def _run(method_name, target, draft, tokenizer, prompts, lookahead, num_drafts,
         max_new_tokens, top_k, top_p, temperature):
    spec = METHODS[method_name]
    fn = spec["fn"]
    if spec["use_warper"]:
        warper = _build_logits_warper(top_k, top_p, temperature)
        process_logits_kwargs = {"logits_warper": warper} if warper else {}
    else:
        process_logits_kwargs = {
            "top_k": top_k, "top_p": top_p, "temperature": temperature,
        }

    rows = []
    for prompt in prompts:
        input_ids = encode_prompt(tokenizer, prompt, target.device)
        gen = fn(
            model=target,
            ref_model=draft,
            input_ids=input_ids,
            n=lookahead,
            max_length=max_new_tokens,
            num_drafts=num_drafts,
            private_key=b"1234",
            process_logits_kwargs=process_logits_kwargs,
            return_meta=True,
        )
        block_lens: List[int] = []
        accepted_counts: List[int] = []
        emitted = 0
        _sync()
        t0 = time.perf_counter()
        for step_ids, _logp, meta in gen:
            block_len = step_ids.shape[-1]
            if emitted + block_len > max_new_tokens:
                block_len = max_new_tokens - emitted
            if block_len <= 0:
                break
            block_lens.append(int(block_len))
            accepted_counts.append(int(meta.get("accepted_count", max(block_len - 1, 0))))
            emitted += block_len
            if emitted >= max_new_tokens:
                break
        _sync()
        elapsed = time.perf_counter() - t0
        rows.append({
            "tokens": int(emitted),
            "blocks": int(len(block_lens)),
            "AATPS": float(emitted / max(len(block_lens), 1)),
            "accepted_mean": float(np.mean(accepted_counts)) if accepted_counts else 0.0,
            "token_rate": float(emitted / elapsed) if elapsed > 0 else 0.0,
            "elapsed_sec": float(elapsed),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-drafts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--target-model", type=Path,
                        default=ROOT / "model" / "Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", type=Path,
                        default=ROOT / "model" / "Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--methods", nargs="+",
                        default=list(METHODS.keys()),
                        choices=list(METHODS.keys()))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"loading target {args.target_model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_model), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_model),
        device_map=args.device,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    print(f"loading draft  {args.draft_model} on {args.device}")
    draft = AutoModelForCausalLM.from_pretrained(
        str(args.draft_model),
        device_map=args.device,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()

    prompts_all = load_prompts(args.samples + args.warmup)
    warmup = prompts_all[: args.warmup]
    eval_prompts = prompts_all[args.warmup : args.warmup + args.samples]
    print(f"warmup={len(warmup)}  eval={len(eval_prompts)}  "
          f"lookahead={args.lookahead}  max_new_tokens={args.max_new_tokens}  "
          f"top_k={args.top_k}  top_p={args.top_p}  temperature={args.temperature}")

    if warmup:
        for m in args.methods:
            _run(m, target, draft, tokenizer, warmup, args.lookahead,
                 args.num_drafts[0], args.max_new_tokens,
                 args.top_k, args.top_p, args.temperature)

    results = {}
    for B in args.num_drafts:
        for m in args.methods:
            print(f"\n[run] {m} B={B}")
            results[f"{m}_B{B}"] = _run(
                m, target, draft, tokenizer, eval_prompts,
                args.lookahead, B, args.max_new_tokens,
                args.top_k, args.top_p, args.temperature,
            )

    print("\n=== Comparison (mean over {} prompts, draft_len={}, top_k={}) ===".format(
        len(eval_prompts), args.lookahead, args.top_k))
    print(f"{'method':<36s} {'AATPS':>8s} {'TR (tok/s)':>12s} {'tokens/sample':>14s}")
    summary = {}
    for name, rows in results.items():
        aatps_mean = float(np.mean([r["AATPS"] for r in rows]))
        tr_mean = float(np.mean([r["token_rate"] for r in rows]))
        tok_mean = float(np.mean([r["tokens"] for r in rows]))
        summary[name] = {"AATPS": aatps_mean, "token_rate": tr_mean,
                         "tokens_per_sample": tok_mean}
        print(f"{name:<36s} {aatps_mean:>8.3f} {tr_mean:>12.2f} {tok_mean:>14.1f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "summary": summary,
            "rows": results,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
