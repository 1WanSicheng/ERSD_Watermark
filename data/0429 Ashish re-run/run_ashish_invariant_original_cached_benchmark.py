#!/usr/bin/env python3
"""
Run Ashish/Rowan's original InvariantMultiDraftStrategy with KV cache enabled.

This runner is intentionally narrow: it only evaluates the conditionally
invariant multi-draft baseline from the uploaded SpeculativeDecoding folder.
It calls the original class methods directly:

    strategy.generate_draft(...)
    strategy.verify_draft(...)

The only compatibility change is a monkey patch for recent Hugging Face
DynamicCache objects so that the uploaded original code, which expects
old-style cache fields such as .key_cache/.value_cache and cache[0][0], can
run without rewriting the algorithm.

Outputs use the same raw/summary-style CSV schema as the previous benchmark
scripts, so these rows can replace earlier incorrect Ashish rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def _get_layers(cache):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers
    # Older/alternative cache implementations may store lists directly.
    for name in ("_layers", "cache", "_cache"):
        layers = getattr(cache, name, None)
        if layers is not None:
            return layers
    raise AttributeError("Could not find layers on DynamicCache object")


def _get_layer_tensor(layer, kind: str):
    # Newer DynamicLayer uses .keys/.values; older versions may use variants.
    names = (
        ("keys", "key_states", "key", "key_cache")
        if kind == "key"
        else ("values", "value_states", "value", "value_cache")
    )
    for name in names:
        if hasattr(layer, name):
            x = getattr(layer, name)
            if x is not None:
                return x
    raise AttributeError(f"Could not find {kind} tensor on cache layer {type(layer)}")


def _set_layer_tensor(layer, kind: str, value):
    preferred = ("keys", "key_states", "key", "key_cache") if kind == "key" else ("values", "value_states", "value", "value_cache")
    # Prefer writing back to an attribute that already exists.
    for name in preferred:
        if hasattr(layer, name):
            try:
                setattr(layer, name, value)
                return
            except Exception:
                pass
    # Last resort: use the common modern names.
    setattr(layer, "keys" if kind == "key" else "values", value)


class _CacheTensorList:
    """Mutable list-like proxy exposing DynamicCache layer tensors."""

    def __init__(self, cache, kind: str):
        self.cache = cache
        self.kind = kind

    def __len__(self):
        return len(_get_layers(self.cache))

    def __getitem__(self, idx):
        return _get_layer_tensor(_get_layers(self.cache)[idx], self.kind)

    def __setitem__(self, idx, value):
        _set_layer_tensor(_get_layers(self.cache)[idx], self.kind, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def patch_dynamic_cache_for_original_strategy() -> None:
    """Expose legacy .key_cache/.value_cache/cache[i] API on DynamicCache.

    The uploaded SpeculativeDecoding/strategy.py mutates .key_cache and
    .value_cache directly. Recent Transformers versions wrap them inside
    cache.layers. This patch does not change the algorithm; it only supplies
    the old API expected by the original class methods.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception as e:
        print(f"Warning: could not import DynamicCache for compatibility patch: {e}")
        return

    def _key_cache(self):
        return _CacheTensorList(self, "key")

    def _value_cache(self):
        return _CacheTensorList(self, "value")

    def _getitem(self, idx):
        return (self.key_cache[idx], self.value_cache[idx])

    def _len(self):
        return len(_get_layers(self))

    # Override even if properties exist; the proxy is compatible with assignment.
    DynamicCache.key_cache = property(_key_cache)  # type: ignore[attr-defined]
    DynamicCache.value_cache = property(_value_cache)  # type: ignore[attr-defined]
    DynamicCache.__getitem__ = _getitem  # type: ignore[method-assign]
    DynamicCache.__len__ = _len  # type: ignore[method-assign]


def load_prompt_dataset(name: str):
    if name == "openai/gsm8k":
        return load_dataset(name, "main")["train"]
    if name == "openai/openai_humaneval":
        return load_dataset(name, "openai_humaneval")["test"]
    if name == "facebook/natural_reasoning":
        return load_dataset(name, "default")["train"]
    if name == "google-research-datasets/mbpp":
        return load_dataset(name, "full")["train"]
    if name == "ucinlp/drop":
        return load_dataset(name, "default")["train"]
    raise ValueError(f"Unsupported dataset: {name}")


def create_prompt(row: Dict[str, Any], dset_name: str) -> List[Dict[str, str]]:
    if dset_name == "openai/gsm8k":
        user = row["question"]
    elif dset_name == "openai/openai_humaneval":
        user = f"Complete the code:\n{row['prompt']}"
    elif dset_name == "facebook/natural_reasoning":
        user = row["question"]
    elif dset_name == "google-research-datasets/mbpp":
        user = f"{row['text']} Use Python."
    elif dset_name == "ucinlp/drop":
        user = f"{row['passage']}\n\n{row['question']}"
    else:
        raise ValueError(f"Unsupported dataset: {dset_name}")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]


def input_ids_from_prompt(tokenizer, prompt: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        encoded = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        if isinstance(encoded, torch.Tensor):
            return encoded.to(device)
        if hasattr(encoded, "input_ids"):
            return encoded.input_ids.to(device)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"].to(device)
        raise TypeError(f"Unexpected apply_chat_template output type: {type(encoded)}")
    text = "\n".join(f"{m['role']}: {m['content']}" for m in prompt) + "\nassistant:"
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


@torch.no_grad()
def run_one_prompt_original_cached(
    *,
    strategy,
    tokenizer,
    input_ids: torch.Tensor,
    config: Dict[str, Any],
    gen_length: int,
    LogitsProcessor,
) -> Dict[str, Any]:
    device = model_device(strategy.target)
    input_ids = input_ids.to(device)
    input_len = int(input_ids.size(-1))

    L = int(config["max_draft_len"])
    B = int(config["max_num_drafts"])
    temperature = config.get("temperature", 1.0)
    top_k = int(config.get("top_k", 50))
    top_p = float(config.get("top_p", 1.0))

    if hasattr(temperature, "__len__") and not isinstance(temperature, (str, bytes)):
        target_temp = torch.tensor(temperature[0], device=device).reshape((1, 1, 1))
        draft_temp = torch.tensor(temperature[1:], device=device).reshape((-1, 1, 1))
        target_logits_processor = LogitsProcessor(target_temp, top_p, top_k)
        draft_logits_processor = LogitsProcessor(draft_temp, top_p, top_k)
    else:
        temp = torch.tensor(float(temperature), device=device).reshape((1, 1, 1))
        target_logits_processor = LogitsProcessor(temp, top_p, top_k)
        draft_logits_processor = LogitsProcessor(temp, top_p, top_k)

    randomness = torch.empty((gen_length + L + 1, B, strategy.vocab_size), device=device)
    randomness.uniform_()

    target_past_key_values = None
    draft_past_key_values = None
    num_steps = 0
    accepted_tokens_total = 0
    attempted_draft_tokens_total = 0
    target_forward_calls_total = 0
    draft_forward_calls_total = 0
    total_generation_time = 0.0
    total_verification_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t0 = perf_counter()
    while input_ids.size(-1) < input_len + gen_length:
        position = int(input_ids.size(-1) - input_len)
        remaining = int(input_len + gen_length - input_ids.size(-1))
        if remaining <= 0:
            break

        # Original class assumes fixed max_draft_len. Stop when fewer than L+1
        # target positions remain to avoid changing the original methods.
        # This matches the common 128-token case unless the final block would
        # overshoot. If it overshoots, generation can exceed gen_length slightly;
        # truncate accounting below if needed.
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tg0 = perf_counter()
        draft_outputs = strategy.generate_draft(
            input_ids=input_ids,
            past_key_values=draft_past_key_values,
            logits_processor=draft_logits_processor,
            position=position,
            randomness=randomness,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tg1 = perf_counter()
        draft_past_key_values = draft_outputs.draft_past_key_values

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tv0 = perf_counter()
        verify_outputs = strategy.verify_draft(
            input_ids=draft_outputs.sequences,
            target_past_key_values=target_past_key_values,
            draft_past_key_values=draft_past_key_values,
            draft_probs=draft_outputs.draft_probs,
            logits_processor=target_logits_processor,
            position=position,
            randomness=randomness,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tv1 = perf_counter()

        old_len = int(input_ids.size(-1))
        input_ids = verify_outputs.sequences
        emitted = int(input_ids.size(-1) - old_len)
        draft_past_key_values = verify_outputs.draft_past_key_values
        target_past_key_values = verify_outputs.target_past_key_values

        accepted_draft = max(0, int(verify_outputs.accept_count) - 1)
        num_steps += 1
        accepted_tokens_total += accepted_draft
        attempted_draft_tokens_total += L * B
        target_forward_calls_total += 1
        draft_forward_calls_total += L
        total_generation_time += tg1 - tg0
        total_verification_time += tv1 - tv0

        eos = tokenizer.eos_token_id
        if eos is not None and emitted > 0 and bool((input_ids[:, -emitted:] == int(eos)).any().item()):
            break

        # Safety guard if an unexpected cache issue causes no progress.
        if emitted <= 0:
            raise RuntimeError("Original invariant strategy emitted no tokens; aborting to avoid infinite loop")

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t1 = perf_counter()

    num_generated_tokens = int(input_ids.size(-1) - input_len)
    total_time = float(t1 - t0)
    token_rate = num_generated_tokens / total_time if total_time > 0 else float("nan")
    be = num_generated_tokens / num_steps if num_steps else float("nan")
    aatps = accepted_tokens_total / num_steps if num_steps else float("nan")
    acceptance_fraction = accepted_tokens_total / attempted_draft_tokens_total if attempted_draft_tokens_total else float("nan")
    normalized_aatps = aatps / L if L else float("nan")

    return {
        "num_generated_tokens": num_generated_tokens,
        "total_time": total_time,
        "token_rate": token_rate,
        "num_steps": num_steps,
        "accepted_tokens_total": float(accepted_tokens_total),
        "attempted_draft_tokens_total": float(attempted_draft_tokens_total),
        "aatps": aatps,
        "be": be,
        "tokens_per_step": be,
        "acceptance_fraction": acceptance_fraction,
        "normalized_aatps": normalized_aatps,
        "avg_block_time": total_time / num_steps if num_steps else float("nan"),
        "target_forward_calls_total": float(target_forward_calls_total),
        "draft_forward_calls_total": float(draft_forward_calls_total),
        "target_forward_calls_per_token": target_forward_calls_total / num_generated_tokens if num_generated_tokens else float("nan"),
        "draft_forward_calls_per_token": draft_forward_calls_total / num_generated_tokens if num_generated_tokens else float("nan"),
        "avg_generation_time": total_generation_time / num_steps if num_steps else float("nan"),
        "avg_verification_time": total_verification_time / num_steps if num_steps else float("nan"),
    }


def summarize(raw_path: Path, summary_path: Path) -> None:
    df = pd.read_csv(raw_path)
    metric_cols = [
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be",
        "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_forward_calls_total", "draft_forward_calls_total",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
        "avg_generation_time", "avg_verification_time",
    ]
    grouped = df.groupby(["seed", "dataset", "method", "config_name"], dropna=False)[metric_cols]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(summary_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", default="outputs_ashish_invariant_original_cached")
    parser.add_argument("--speculative_dir", default="SpeculativeDecoding")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    spec_dir = Path(args.speculative_dir).resolve()
    if not spec_dir.exists():
        raise FileNotFoundError(f"SpeculativeDecoding folder not found: {spec_dir}")
    sys.path.insert(0, str(spec_dir))

    patch_dynamic_cache_for_original_strategy()

    # Import the original class and utilities after sys.path and cache patch.
    from strategy import InvariantMultiDraftStrategy  # type: ignore
    from utils import LogitsProcessor  # type: ignore

    seed_everything(args.seed)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    configs = [c for c in cfg["configurations"] if c.get("method") in {"ashish_invariant", "ashish_invariant_official", "ashish_invariant_original_cached"}]
    if not configs:
        raise ValueError("No Ashish invariant configurations found in JSON")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["target_model"], device_map="auto", torch_dtype=dtype).eval()
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["draft_model"], device_map="auto", torch_dtype=dtype).eval()

    gen_length = int(cfg.get("gen_length", 128))
    warmup = int(args.warmup if args.warmup is not None else cfg.get("warmup", 10))
    max_prompts = int(args.max_prompts if args.max_prompts is not None else cfg.get("num_test_prompts", 200))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raw_path = out_dir / f"ashish_invariant_original_cached_raw_seed{args.seed}_{stamp}.csv"
    summary_path = out_dir / f"ashish_invariant_original_cached_summary_seed{args.seed}_{stamp}.csv"

    fieldnames = [
        "seed", "dataset", "method", "config_name", "test_num", "prompt_index",
        "max_draft_len", "max_num_drafts", "temperature", "top_k", "top_p",
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be",
        "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_forward_calls_total", "draft_forward_calls_total",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
        "avg_generation_time", "avg_verification_time",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dset_name in cfg["datasets"]:
            dset = load_prompt_dataset(dset_name)
            n_test = min(max_prompts, len(dset))
            n_warm = min(warmup, len(dset))

            for config0 in configs:
                config = dict(config0)
                name = config["name"]
                top_k = int(config.get("top_k", cfg.get("top_k", 50)))
                top_p = float(config.get("top_p", cfg.get("top_p", 1.0)))
                config["top_k"] = top_k
                config["top_p"] = top_p
                L = int(config["max_draft_len"])
                B = int(config["max_num_drafts"])

                # Recreate original strategy per config. This matches the original usage.
                strategy = InvariantMultiDraftStrategy(model, ref_model, tokenizer, L, B)

                for i in tqdm(range(n_warm), total=n_warm, desc=f"Warmup {dset_name.split('/')[-1]} | {name}"):
                    prompt = create_prompt(dset[i], dset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_device(model))
                    _ = run_one_prompt_original_cached(
                        strategy=strategy,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                        LogitsProcessor=LogitsProcessor,
                    )

                for i in tqdm(range(n_test), total=n_test, desc=f"Test {dset_name.split('/')[-1]} | {name}"):
                    prompt = create_prompt(dset[i], dset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_device(model))
                    stats = run_one_prompt_original_cached(
                        strategy=strategy,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                        LogitsProcessor=LogitsProcessor,
                    )
                    row = {
                        "seed": args.seed,
                        "dataset": dset_name,
                        "method": "ashish_invariant_original_cached",
                        "config_name": name.replace("ashish_invariant_official", "ashish_invariant_original_cached").replace("ashish_invariant", "ashish_invariant_original_cached"),
                        "test_num": i,
                        "prompt_index": i,
                        "max_draft_len": L,
                        "max_num_drafts": B,
                        "temperature": json.dumps(config.get("temperature", 1.0)),
                        "top_k": top_k,
                        "top_p": top_p,
                    }
                    row.update(stats)
                    writer.writerow(row)
                    f.flush()

    summarize(raw_path, summary_path)
    print(f"Wrote raw: {raw_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
