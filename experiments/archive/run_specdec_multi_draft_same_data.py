import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import DynamicCache


ROOT = Path(__file__).resolve().parents[2]
SPECDEC = ROOT / "SpeculativeDecoding"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SPECDEC) not in sys.path:
    sys.path.insert(0, str(SPECDEC))

import generator as spec_generator
from generator import InvariantGenerator, SpeculativeGenerator
import strategy as spec_strategy
from strategy import InvariantMultiDraftStrategy, SingleDraftStrategy, StrongMultiDraftStrategy

from experiments.archive.run_multi_draft_pfr_aatps import load_model
from experiments.tasks import get_gsm8k_chat_prompts, get_summarization_ds


DEFAULT_TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


class _CacheTensorList:
    def __init__(self, cache, attr):
        self.cache = cache
        self.attr = attr

    def __len__(self):
        return len(self.cache.layers)

    def __getitem__(self, idx):
        return getattr(self.cache.layers[idx], self.attr)

    def __setitem__(self, idx, value):
        setattr(self.cache.layers[idx], self.attr, value)


def install_dynamic_cache_compat():
    if not hasattr(DynamicCache, "key_cache"):
        DynamicCache.key_cache = property(lambda self: _CacheTensorList(self, "keys"))
    if not hasattr(DynamicCache, "value_cache"):
        DynamicCache.value_cache = property(lambda self: _CacheTensorList(self, "values"))


def install_quiet_sampling():
    def quiet_gumbel_sample(logits, noise=None, dim=-1):
        if noise is None:
            noise = torch.zeros_like(logits).uniform_(0, 1)
        return (logits + (-torch.log(-torch.log(noise.clamp_min(1e-20))))).argmax(dim=dim)

    spec_strategy.gumbel_sample = quiet_gumbel_sample
    spec_generator.os = os


def make_strategy(name, target, draft, tokenizer, lookahead, num_drafts):
    if name == "single_draft":
        return SingleDraftStrategy(target, draft, tokenizer, lookahead, 1)
    if name == "invariant_multi_draft":
        return InvariantMultiDraftStrategy(target, draft, tokenizer, lookahead, num_drafts)
    if name == "strong_multi_draft":
        return StrongMultiDraftStrategy(target, draft, tokenizer, lookahead, num_drafts)
    raise ValueError(f"unknown strategy: {name}")


def encode_prompt(tokenizer, prompt, device):
    if isinstance(prompt, list):
        return tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def run_prompt(target, draft, tokenizer, prompt, strategy_name, lookahead, num_drafts, max_length):
    input_ids = encode_prompt(tokenizer, prompt, target.device)
    strategy = make_strategy(
        strategy_name,
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        lookahead=lookahead,
        num_drafts=num_drafts,
    )
    if strategy_name == "single_draft":
        generator = SpeculativeGenerator(strategy)
        temperature = [1.0, 1.0]
    else:
        generator = InvariantGenerator(strategy)
        temperature = [1.0] * (num_drafts + 1)

    start = time.perf_counter()
    outputs = generator(
        input_ids=input_ids,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temperature,
        max_new_tokens=max_length,
    )
    elapsed = time.perf_counter() - start

    num_tokens = int(outputs.sequences.shape[-1] - input_ids.shape[-1])
    be = float(outputs.acceptance_rate)
    tr = float(outputs.token_rate)
    return {
        "num_tokens": num_tokens,
        "num_invocations": int(outputs.num_invocations),
        "BE": be,
        "block_efficiency": be,
        "acceptance_rate": be,
        "TR": tr,
        "token_rate": tr,
        "tokens_per_sec": tr,
        "avg_generation_time": float(outputs.avg_generation_time),
        "avg_verification_time": float(outputs.avg_verification_time),
        "total_time": float(outputs.total_time),
        "elapsed_sec": float(elapsed),
    }


def aggregate(rows):
    be_values = [row["BE"] for row in rows]
    tr_values = [row["TR"] for row in rows]
    total_tokens = int(sum(row["num_tokens"] for row in rows))
    total_time = float(sum(row["total_time"] for row in rows))
    return {
        "num_samples": len(rows),
        "num_tokens_total": total_tokens,
        "BE_mean": float(np.mean(be_values)),
        "BE_std": float(np.std(be_values)),
        "acceptance_rate_mean": float(np.mean(be_values)),
        "TR_mean": float(np.mean(tr_values)),
        "TR_std": float(np.std(tr_values)),
        "TR_global": float(total_tokens / total_time) if total_time > 0 else 0.0,
        "token_rate_mean": float(np.mean(tr_values)),
        "token_rate_std": float(np.std(tr_values)),
        "num_invocations_mean": float(np.mean([row["num_invocations"] for row in rows])),
        "num_invocations_total": int(sum(row["num_invocations"] for row in rows)),
        "avg_generation_time_mean": float(np.mean([row["avg_generation_time"] for row in rows])),
        "avg_verification_time_mean": float(np.mean([row["avg_verification_time"] for row in rows])),
        "total_time_total": total_time,
        "elapsed_sec_total": float(sum(row["elapsed_sec"] for row in rows)),
    }


def load_prompts(dataset, samples):
    if dataset == "summarization":
        return [row["prompt"] for row in get_summarization_ds(ds_cut_len=samples)]
    if dataset == "gsm8k":
        return [row["prompt"] for row in get_gsm8k_chat_prompts(ds_cut_len=samples)]
    raise ValueError(f"unknown dataset: {dataset}")


def main():
    install_dynamic_cache_compat()
    install_quiet_sampling()

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dataset", choices=["summarization", "gsm8k"], default="summarization")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--draft-counts", type=int, nargs="*", default=[2, 4, 6])
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["invariant_multi_draft", "strong_multi_draft"],
        choices=["single_draft", "invariant_multi_draft", "strong_multi_draft"],
    )
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
        warmup_strategy = args.strategies[0]
        warmup_num_drafts = args.draft_counts[0]
        for prompt in prompts[: args.warmup]:
            run_prompt(
                target=target,
                draft=draft,
                tokenizer=tokenizer,
                prompt=prompt,
                strategy_name=warmup_strategy,
                lookahead=args.lookahead,
                num_drafts=warmup_num_drafts,
                max_length=args.max_length,
            )
    results = {}
    if args.rows_output is not None:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.rows_output.write_text("", encoding="utf-8")

    for strategy_name in args.strategies:
        for num_drafts in args.draft_counts:
            key = f"{strategy_name}_B{num_drafts}"
            rows = []
            for idx, prompt in enumerate(prompts):
                sample_start = time.perf_counter()
                metrics = run_prompt(
                    target=target,
                    draft=draft,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    strategy_name=strategy_name,
                    lookahead=args.lookahead,
                    num_drafts=num_drafts,
                    max_length=args.max_length,
                )
                row = {
                    "sample_idx": idx,
                    "strategy": strategy_name,
                    "lookahead": args.lookahead,
                    "num_drafts": num_drafts,
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
                                    "strategy": strategy_name,
                                    "num_drafts": num_drafts,
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
                                    "draft_counts": args.draft_counts,
                                    "strategies": args.strategies,
                                    "max_length": args.max_length,
                                    "target_model": str(args.target_model),
                                    "draft_model": str(args.draft_model),
                                    "target_device": args.target_device,
                                    "draft_device": draft_device,
                                    "partial": True,
                                    "results": {
                                        **results,
                                        key: {
                                            "summary": partial,
                                            "rows": rows,
                                        },
                                    },
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
            results[key] = {
                "summary": aggregate(rows),
                "rows": rows,
            }

    payload = {
        "samples": args.samples,
        "dataset": args.dataset,
        "warmup": args.warmup,
        "lookahead": args.lookahead,
        "draft_counts": args.draft_counts,
        "strategies": args.strategies,
        "max_length": args.max_length,
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "target_device": args.target_device,
        "draft_device": draft_device,
        "metric_note": "BE is SpeculativeDecoding acceptance_rate. TR is SpeculativeDecoding token_rate.",
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
