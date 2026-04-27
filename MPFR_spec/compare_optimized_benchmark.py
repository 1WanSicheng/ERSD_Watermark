#!/usr/bin/env python3
"""
Unified benchmark for optimized direct-MPFR and list-level identical multi-draft speculative decoding.

This script is designed to sit next to:

  finite_multi_draft_pfr_local.py
  list_level_baseline_local.py
  compare_qwen25_benchmark.json

It keeps the MPFR implementation untouched.  For MPFR configs it imports your
MPFR generator.  For baseline configs it imports local generators from
list_level_baseline_local.py.

Main metrics written per prompt:

  token_rate: generated tokens / wall-clock seconds.
  aatps: accepted drafted tokens / speculative block.
  be: block efficiency = all generated tokens / speculative block.
  acceptance_fraction: accepted drafted tokens / attempted draft depth.
  target_contexts_per_token: target forward contexts per generated token.
  draft_tree_nodes_per_token: drafter contexts/nodes per generated token.

For list-level and single-draft baselines, one speculative block uses one target
verification call, so target_context_count=1 per block.  For MPFR, the generator
meta may report more target contexts per block, which is kept in the metric.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_WARMUP = 3


@dataclass
class RunStats:
    num_generated_tokens: int
    total_time: float
    token_rate: float
    num_steps: int
    accepted_tokens_total: int
    attempted_draft_tokens_total: int
    aatps: float
    be: float
    tokens_per_step: float
    acceptance_fraction: float
    normalized_aatps: float
    avg_block_time: float
    target_contexts_total: int
    draft_tree_nodes_total: int
    target_forward_calls_total: int
    draft_forward_calls_total: int
    target_contexts_per_token: float
    draft_tree_nodes_per_token: float
    target_forward_calls_per_token: float
    draft_forward_calls_per_token: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def torch_dtype_from_name(name: str):
    table = {
        "auto": None,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; choose one of {sorted(table)}")
    return table[name]


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def import_function(module_name: str, function_name: str) -> Callable[..., Iterable[Any]]:
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def create_prompt(example: Dict[str, Any], dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name == "openai/gsm8k":
        user = example["question"]
    elif dataset_name == "openai/openai_humaneval":
        user = f"Complete the code:\n{example['prompt']}"
    elif dataset_name == "facebook/natural_reasoning":
        user = example["question"]
    elif dataset_name == "mandarjoshi/trivia_qa":
        user = example["question"]
    elif dataset_name == "google-research-datasets/mbpp":
        user = f"{example['text']} Use Python."
    elif dataset_name == "ucinlp/drop":
        user = f"{example['passage']}\n\n{example['question']}"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]


def load_named_dataset(dataset_name: str):
    if dataset_name == "openai/gsm8k":
        return load_dataset(dataset_name, "main")["train"]
    if dataset_name == "openai/openai_humaneval":
        return load_dataset(dataset_name, "openai_humaneval")["test"]
    if dataset_name == "facebook/natural_reasoning":
        return load_dataset(dataset_name, "default")["train"]
    if dataset_name == "mandarjoshi/trivia_qa":
        return load_dataset(dataset_name, "rc")["train"]
    if dataset_name == "google-research-datasets/mbpp":
        return load_dataset(dataset_name, "full")["train"]
    if dataset_name == "ucinlp/drop":
        return load_dataset(dataset_name, "default")["train"]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def input_ids_from_prompt(tokenizer, prompt: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
    """Return a torch.LongTensor input_ids, never a BatchEncoding."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        encoded = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(encoded, torch.Tensor):
            return encoded.to(device)
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids.to(device)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"].to(device)
        raise TypeError(f"Unexpected apply_chat_template output type: {type(encoded)}")

    text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt) + "\nassistant:"
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids.to(device)


def build_process_logits_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(config.get("process_logits_kwargs", {}))
    for key in ("temperature", "top_p", "top_k"):
        if key in config and key not in kwargs:
            kwargs[key] = config[key]
    kwargs.setdefault("temperature", 1.0)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("top_k", 50)
    return kwargs


def choose_generator(config: Dict[str, Any], args) -> Callable[..., Iterable[Any]]:
    algorithm = str(config.get("algorithm", config.get("strategy", "mpfr")))
    if algorithm in {"mpfr", "mpfr_direct", "mpfr_batched"}:
        return import_function(args.mpfr_module, args.mpfr_function)
    if algorithm in {"list_identical", "strong_multi_draft"}:
        return import_function(args.list_module, "list_level_identical_generator")
    if algorithm == "single_draft":
        return import_function(args.list_module, "single_draft_generator")
    if algorithm in {"basic", "target_only"}:
        return import_function(args.list_module, "target_only_generator")
    raise ValueError(f"Unknown algorithm/strategy: {algorithm}")


def canonical_algorithm(config: Dict[str, Any]) -> str:
    algorithm = str(config.get("algorithm", config.get("strategy", "mpfr")))
    if algorithm == "strong_multi_draft":
        return "list_identical"
    if algorithm == "basic":
        return "target_only"
    if algorithm in {"mpfr_direct", "mpfr_batched"}:
        return "mpfr"
    return algorithm


def run_one_prompt(
    *,
    generator: Callable[..., Iterable[Any]],
    algorithm: str,
    model,
    ref_model,
    tokenizer,
    input_ids: torch.Tensor,
    config: Dict[str, Any],
    gen_length: int,
    private_key: bytes | str,
    default_max_proposals: int,
    default_allow_incomplete: bool,
    default_proposal: str,
) -> RunStats:
    lookahead = int(config.get("max_draft_len", 1))
    num_drafts = int(config.get("max_num_drafts", 1))
    process_logits_kwargs = build_process_logits_kwargs(config)
    proposal = str(config.get("proposal", default_proposal))
    max_proposals = int(config.get("max_proposals", default_max_proposals))
    allow_incomplete = bool(config.get("allow_incomplete", default_allow_incomplete))

    total_tokens = 0
    accepted_total = 0
    attempted_total = 0
    steps = 0
    target_contexts_total = 0
    draft_tree_nodes_total = 0
    target_forward_calls_total = 0
    draft_forward_calls_total = 0

    cuda_synchronize()
    t0 = time.perf_counter()

    common_kwargs = dict(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        lookahead=lookahead,
        num_drafts=num_drafts,
        max_length=gen_length,
        private_key=private_key,
        labeler=None,
        process_logits_kwargs=process_logits_kwargs,
        return_meta=True,
    )
    if algorithm in {"mpfr", "mpfr_direct", "mpfr_batched"}:
        common_kwargs.update(
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            proposal=proposal,
        )

    iterator = generator(**common_kwargs)

    for yielded in iterator:
        if len(yielded) == 3:
            output_ids, _output_logprobs, meta = yielded
        elif len(yielded) == 2:
            output_ids, _output_logprobs = yielded
            meta = {}
        else:
            raise RuntimeError("generator must yield a tuple of length 2 or 3")
        cuda_synchronize()

        emitted = int(output_ids.shape[-1])
        total_tokens += emitted
        accepted = int(meta.get("accepted_count", max(emitted - 1, 0)))
        accepted_total += accepted
        steps += 1

        # Attempted draft depth for AATPS normalization.  Near the end of a prompt
        # the generator may use a shorter draft length to avoid overshooting.
        attempted_total += int(meta.get("attempted_draft_tokens", min(lookahead, max(gen_length - (total_tokens - emitted) - 1, 0))))

        target_contexts_total += int(meta.get("target_context_count", 1))
        draft_tree_nodes_total += int(meta.get("draft_tree_size", lookahead * max(num_drafts, 1)))
        target_forward_calls_total += int(meta.get("target_forward_calls", meta.get("target_context_count", 1)))
        draft_forward_calls_total += int(meta.get("draft_forward_calls", meta.get("draft_tree_size", lookahead * max(num_drafts, 1))))

        if tokenizer.eos_token_id is not None and bool((output_ids == tokenizer.eos_token_id).any().item()):
            break
        if total_tokens >= gen_length:
            break

    cuda_synchronize()
    total_time = time.perf_counter() - t0

    token_rate = total_tokens / total_time if total_time > 0 else float("nan")
    aatps = accepted_total / steps if steps > 0 else float("nan")
    be = total_tokens / steps if steps > 0 else float("nan")
    acceptance_fraction = accepted_total / attempted_total if attempted_total > 0 else float("nan")
    normalized_aatps = aatps / lookahead if lookahead > 0 else float("nan")
    avg_block_time = total_time / steps if steps > 0 else float("nan")

    return RunStats(
        num_generated_tokens=total_tokens,
        total_time=total_time,
        token_rate=token_rate,
        num_steps=steps,
        accepted_tokens_total=accepted_total,
        attempted_draft_tokens_total=attempted_total,
        aatps=aatps,
        be=be,
        tokens_per_step=be,
        acceptance_fraction=acceptance_fraction,
        normalized_aatps=normalized_aatps,
        avg_block_time=avg_block_time,
        target_contexts_total=target_contexts_total,
        draft_tree_nodes_total=draft_tree_nodes_total,
        target_forward_calls_total=target_forward_calls_total,
        draft_forward_calls_total=draft_forward_calls_total,
        target_contexts_per_token=(target_contexts_total / total_tokens if total_tokens > 0 else float("nan")),
        draft_tree_nodes_per_token=(draft_tree_nodes_total / total_tokens if total_tokens > 0 else float("nan")),
        target_forward_calls_per_token=(target_forward_calls_total / total_tokens if total_tokens > 0 else float("nan")),
        draft_forward_calls_per_token=(draft_forward_calls_total / total_tokens if total_tokens > 0 else float("nan")),
    )


def write_summary_csv(raw_csv: Path, summary_csv: Path) -> None:
    import pandas as pd

    df = pd.read_csv(raw_csv)
    metric_cols = [
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be",
        "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_contexts_total", "draft_tree_nodes_total", "target_forward_calls_total",
        "draft_forward_calls_total", "target_contexts_per_token", "draft_tree_nodes_per_token",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
    ]
    summary = df.groupby(["dataset", "algorithm", "config_name"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    # Add speedups when target_only or single_draft baselines are present in the same dataset.
    summary["speedup_vs_target_only_pct"] = np.nan
    summary["speedup_vs_single_draft_pct"] = np.nan
    for dataset, idx in summary.groupby("dataset").groups.items():
        sub = summary.loc[idx]
        base_target = sub.loc[sub["algorithm"].eq("target_only"), "token_rate_mean"]
        base_single = sub.loc[sub["algorithm"].eq("single_draft"), "token_rate_mean"]
        if len(base_target) > 0 and float(base_target.iloc[0]) > 0:
            b = float(base_target.iloc[0])
            summary.loc[idx, "speedup_vs_target_only_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)
        if len(base_single) > 0 and float(base_single.iloc[0]) > 0:
            b = float(base_single.iloc[0])
            summary.loc[idx, "speedup_vs_single_draft_pct"] = 100.0 * (summary.loc[idx, "token_rate_mean"] / b - 1.0)

    summary.to_csv(summary_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--mpfr_module", type=str, default="finite_multi_draft_pfr_local")
    parser.add_argument("--mpfr_function", type=str, default="finite_multi_draft_pfr_generator")
    parser.add_argument("--list_module", type=str, default="list_level_baseline_local")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="outputs_compare")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--default_max_proposals", type=int, default=100_000)
    parser.add_argument("--default_allow_incomplete", action="store_true")
    parser.add_argument("--default_proposal", type=str, default="batched_target_direct_topk",
                        choices=["uniform", "model", "partition_local_uniform", "partition_local_model",
                                 "direct_topk", "batched_target_direct_topk"])
    parser.add_argument("--max_prompts", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    with open(args.config, "r") as f:
        cfg = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raw_csv = output_dir / f"compare_raw_{timestamp}.csv"
    summary_csv = output_dir / f"compare_summary_{timestamp}.csv"

    dtype = torch_dtype_from_name(args.dtype)
    model_kwargs = dict(device_map=args.device_map, trust_remote_code=args.trust_remote_code)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_large = AutoModelForCausalLM.from_pretrained(cfg["target_model"], **model_kwargs)
    model_large.eval()
    model_small = AutoModelForCausalLM.from_pretrained(cfg["draft_model"], **model_kwargs)
    model_small.eval()

    warmup = DEFAULT_WARMUP if args.warmup is None else int(args.warmup)
    gen_length = int(cfg.get("gen_length", 128))
    private_key = cfg.get("private_key", "1234")

    fieldnames = [
        "dataset", "algorithm", "config_name", "test_num", "prompt_index",
        "max_draft_len", "max_num_drafts", "proposal", "max_proposals", "allow_incomplete",
        "temperature", "top_k", "top_p",
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be", "tokens_per_step",
        "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_contexts_total", "draft_tree_nodes_total", "target_forward_calls_total", "draft_forward_calls_total",
        "target_contexts_per_token", "draft_tree_nodes_per_token", "target_forward_calls_per_token", "draft_forward_calls_per_token",
    ]

    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_name in cfg["datasets"]:
            dataset = load_named_dataset(dataset_name)
            num_tests = int(cfg.get("num_test_prompts", 200))
            if args.max_prompts is not None:
                num_tests = min(num_tests, args.max_prompts)
            if num_tests < 0 or num_tests > len(dataset):
                num_tests = len(dataset)

            for config in cfg["configurations"]:
                config_name = str(config["name"])
                algorithm = canonical_algorithm(config)
                generator = choose_generator(config, args)
                desc = f"{dataset_name.split('/')[-1]} | {config_name}"

                for i in tqdm(range(min(warmup, len(dataset))), desc=f"Warmup {desc}"):
                    prompt = create_prompt(dataset[i], dataset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_large.device)
                    _ = run_one_prompt(
                        generator=generator,
                        algorithm=algorithm,
                        model=model_large,
                        ref_model=model_small,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                        private_key=private_key,
                        default_max_proposals=args.default_max_proposals,
                        default_allow_incomplete=args.default_allow_incomplete,
                        default_proposal=args.default_proposal,
                    )

                for i in tqdm(range(num_tests), desc=f"Test {desc}"):
                    prompt = create_prompt(dataset[i], dataset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_large.device)
                    stats = run_one_prompt(
                        generator=generator,
                        algorithm=algorithm,
                        model=model_large,
                        ref_model=model_small,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                        private_key=private_key,
                        default_max_proposals=args.default_max_proposals,
                        default_allow_incomplete=args.default_allow_incomplete,
                        default_proposal=args.default_proposal,
                    )
                    pl_kwargs = build_process_logits_kwargs(config)
                    row = {
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "config_name": config_name,
                        "test_num": i,
                        "prompt_index": i,
                        "max_draft_len": int(config.get("max_draft_len", 1)),
                        "max_num_drafts": int(config.get("max_num_drafts", 1)),
                        "proposal": str(config.get("proposal", args.default_proposal if algorithm == "mpfr" else "")),
                        "max_proposals": int(config.get("max_proposals", args.default_max_proposals if algorithm == "mpfr" else 0)),
                        "allow_incomplete": bool(config.get("allow_incomplete", args.default_allow_incomplete if algorithm == "mpfr" else False)),
                        "temperature": json.dumps(pl_kwargs.get("temperature", 1.0)),
                        "top_k": int(pl_kwargs.get("top_k", 50)),
                        "top_p": float(pl_kwargs.get("top_p", 1.0)),
                        **stats.__dict__,
                    }
                    writer.writerow(row)
                    f.flush()

    write_summary_csv(raw_csv, summary_csv)
    print(f"Wrote raw results to: {raw_csv}")
    print(f"Wrote summary to:     {summary_csv}")


if __name__ == "__main__":
    main()
