#!/usr/bin/env python3
"""
Run the official Ashish/Rowan list-level invariant multi-draft baseline only.

This runner imports InvariantMultiDraftStrategy from the uploaded
SpeculativeDecoding/strategy.py and uses the same active-set invariant logic.
The only compatibility change is that generate_draft/verify_draft are executed
without Hugging Face KV-cache reuse, because the uploaded repository assumes an
older tuple-style cache API while recent Transformers uses DynamicCache.

Outputs use the same raw/summary schema as run_three_method_json_benchmark.py,
so they can be compared with existing MPFR CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# These are imported after adding SpeculativeDecoding/ to sys.path in main().
OfficialInvariantMultiDraftStrategy = None
DraftOutput = None
VerifyOutput = None
LogitsProcessor = None
_gumbel_sample = None


@dataclass
class PromptRecord:
    prompt: List[Dict[str, str]]
    index: int


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
        question = row["question"]
        user = f"{question}"
    elif dset_name == "openai/openai_humaneval":
        question = row["prompt"]
        user = f"Complete the code:\n{question}"
    elif dset_name == "facebook/natural_reasoning":
        question = row["question"]
        user = f"{question}"
    elif dset_name == "google-research-datasets/mbpp":
        question = row["text"]
        user = f"{question} Use Python."
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


class CompatibleOfficialInvariantStrategy:
    """Use the official InvariantMultiDraftStrategy identity and logic, with cache disabled.

    The uploaded class is written for an older Transformers cache API. This wrapper
    instantiates the official class and reuses its attributes. The draft/verify
    code below is the same active-set invariant algorithm: at target verification
    depth s, the Gumbel noise is maxed only over the currently active drafts.
    """

    def __init__(self, target, drafter, tokenizer, max_draft_len: int, max_num_drafts: int):
        # Instantiate the official strategy class so the experiment is tied to the
        # uploaded SpeculativeDecoding/strategy.py implementation and parameters.
        if OfficialInvariantMultiDraftStrategy is None:
            raise RuntimeError("OfficialInvariantMultiDraftStrategy was not imported")
        self.official = OfficialInvariantMultiDraftStrategy(
            target, drafter, tokenizer, max_draft_len, max_num_drafts
        )
        self.target = self.official.target
        self.drafter = self.official.drafter
        self.max_draft_len = self.official.max_draft_len
        self.max_num_drafts = self.official.max_num_drafts
        self.vocab_size = self.official.vocab_size

    @torch.no_grad()
    def generate_draft(self, input_ids, logits_processor, position: int, randomness: torch.Tensor):
        batch_size = self.max_num_drafts
        rn = randomness[position : (position + self.max_draft_len + 1), :, :]
        draft_device = model_device(self.drafter)

        input_ids = input_ids.to(draft_device).repeat((batch_size, 1))

        # Same draft generation rule as the official method, but no KV cache.
        for i in range(self.max_draft_len):
            outputs = self.drafter(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            logits = outputs.logits[..., : self.vocab_size]
            logits = logits_processor(logits[:, -1:])
            draft_ids = _gumbel_sample(
                logits,
                rn[i, :, :].unsqueeze(0).permute(1, 0, 2).to(draft_device),
            )
            input_ids = torch.cat((input_ids, draft_ids), dim=-1)

        return DraftOutput(sequences=input_ids, draft_probs=None, draft_past_key_values=None)

    @torch.no_grad()
    def verify_draft(self, input_ids, logits_processor, position: int, randomness: torch.Tensor):
        batch_size, input_len = input_ids.shape
        device = model_device(self.target)
        input_ids = input_ids.to(device)
        rn = randomness[position : (position + self.max_draft_len + 1), :, :].to(device)

        outputs = self.target(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = outputs.logits[..., : self.vocab_size]
        logits = logits_processor(logits[:, (-self.max_draft_len - 1) :])
        draft_ids = input_ids[:, -self.max_draft_len :]
        assert logits.size(1) == draft_ids.size(1) + 1

        rejected_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        alive_group_id = 0
        depth = 0

        for depth in range(self.max_draft_len):
            # This is the defining official invariant rule: max over active drafts.
            target_noise, _ = torch.max(rn[depth, ~rejected_mask, :], dim=0)
            target_ids = _gumbel_sample(logits[:, depth], target_noise)

            for k in range(batch_size):
                if rejected_mask[k]:
                    continue
                if draft_ids[k, depth] == target_ids[k]:
                    alive_group_id = k
                    rejected_mask |= torch.ne(draft_ids[:, depth], draft_ids[k, depth])
                    break
                rejected_mask[k] = True

            if torch.all(rejected_mask):
                break
        else:
            depth = self.max_draft_len

        if not torch.all(rejected_mask):
            depth = self.max_draft_len

        # Match the uploaded official InvariantMultiDraftStrategy: the final token
        # is sampled from logits[alive_group_id, depth] with fresh Gumbel noise.
        last_token = _gumbel_sample(logits[alive_group_id, depth]).unsqueeze(0)
        output_prefix = input_ids[alive_group_id, : (input_len - self.max_draft_len + depth)]
        new_input_ids = torch.cat((output_prefix, last_token)).unsqueeze(0)

        return VerifyOutput(
            sequences=new_input_ids,
            target_past_key_values=None,
            draft_past_key_values=None,
            accept_count=depth + 1,
        )


@torch.no_grad()
def run_one_prompt_official_invariant(
    *,
    model,
    ref_model,
    tokenizer,
    input_ids: torch.Tensor,
    config: Dict[str, Any],
    gen_length: int,
) -> Dict[str, Any]:
    device = model_device(model)
    input_ids = input_ids.to(device)
    input_len = int(input_ids.size(-1))

    L = int(config["max_draft_len"])
    B = int(config["max_num_drafts"])
    temperature = config.get("temperature", 1.0)
    top_k = int(config.get("top_k", 50))
    top_p = float(config.get("top_p", 1.0))

    strategy = CompatibleOfficialInvariantStrategy(model, ref_model, tokenizer, L, B)

    if hasattr(temperature, "__len__") and not isinstance(temperature, (str, bytes)):
        target_temp = torch.tensor(temperature[0], device=device).reshape((1, 1, 1))
        draft_temp = torch.tensor(temperature[1:], device=device).reshape((-1, 1, 1))
        target_logits_processor = LogitsProcessor(target_temp, top_p, top_k)
        draft_logits_processor = LogitsProcessor(draft_temp, top_p, top_k)
    else:
        temp = torch.tensor(float(temperature), device=device).reshape((1, 1, 1))
        target_logits_processor = LogitsProcessor(temp, top_p, top_k)
        draft_logits_processor = LogitsProcessor(temp, top_p, top_k)

    # Same randomness shape as official InvariantGenerator.
    randomness = torch.empty((gen_length + L + 1, B, strategy.vocab_size), device=device)
    randomness.uniform_()

    num_steps = 0
    accepted_tokens_total = 0
    attempted_draft_tokens_total = 0
    target_forward_calls_total = 0
    draft_forward_calls_total = 0
    total_generation_time = 0.0
    total_verification_time = 0.0

    t0 = perf_counter()
    while input_ids.size(-1) < input_len + gen_length:
        position = int(input_ids.size(-1) - input_len)
        remaining = int(input_len + gen_length - input_ids.size(-1))
        if remaining <= 0:
            break

        # If only one token remains, this block still works, but keep L bounded.
        block_L = min(L, max(1, remaining - 1)) if remaining > 1 else 1
        if block_L != L:
            # The official class has fixed L. For final short remainder, just use
            # a one-step target sample by running the same strategy with L=1.
            old_L = strategy.max_draft_len
            strategy.max_draft_len = block_L
        else:
            old_L = L

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tg0 = perf_counter()
        draft_outputs = strategy.generate_draft(
            input_ids=input_ids,
            logits_processor=draft_logits_processor,
            position=position,
            randomness=randomness,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tg1 = perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tv0 = perf_counter()
        verify_outputs = strategy.verify_draft(
            input_ids=draft_outputs.sequences,
            logits_processor=target_logits_processor,
            position=position,
            randomness=randomness,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        tv1 = perf_counter()

        if old_L != strategy.max_draft_len:
            strategy.max_draft_len = old_L

        emitted = int(verify_outputs.sequences.size(-1) - input_ids.size(-1))
        input_ids = verify_outputs.sequences
        accepted_draft = max(0, int(verify_outputs.accept_count) - 1)

        num_steps += 1
        accepted_tokens_total += accepted_draft
        attempted_draft_tokens_total += block_L * B
        target_forward_calls_total += 1
        draft_forward_calls_total += block_L
        total_generation_time += tg1 - tg0
        total_verification_time += tv1 - tv0

        eos = tokenizer.eos_token_id
        if eos is not None and bool((input_ids[:, -emitted:] == int(eos)).any().item()):
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    t1 = perf_counter()

    num_generated_tokens = int(input_ids.size(-1) - input_len)
    total_time = float(t1 - t0)
    token_rate = num_generated_tokens / total_time if total_time > 0 else float("nan")
    be = num_generated_tokens / num_steps if num_steps else float("nan")
    aatps = accepted_tokens_total / num_steps if num_steps else float("nan")
    acceptance_fraction = (
        accepted_tokens_total / attempted_draft_tokens_total if attempted_draft_tokens_total else float("nan")
    )
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
        "num_generated_tokens",
        "total_time",
        "token_rate",
        "num_steps",
        "accepted_tokens_total",
        "attempted_draft_tokens_total",
        "aatps",
        "be",
        "tokens_per_step",
        "acceptance_fraction",
        "normalized_aatps",
        "avg_block_time",
        "target_forward_calls_total",
        "draft_forward_calls_total",
        "target_forward_calls_per_token",
        "draft_forward_calls_per_token",
        "avg_generation_time",
        "avg_verification_time",
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
    parser.add_argument("--output_dir", default="outputs_ashish_invariant_official")
    parser.add_argument("--speculative_dir", default="SpeculativeDecoding")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    spec_dir = Path(args.speculative_dir).resolve()
    if not spec_dir.exists():
        raise FileNotFoundError(f"SpeculativeDecoding folder not found: {spec_dir}")
    sys.path.insert(0, str(spec_dir))

    global OfficialInvariantMultiDraftStrategy, DraftOutput, VerifyOutput, LogitsProcessor, _gumbel_sample
    from strategy import InvariantMultiDraftStrategy as _OfficialInvariantMultiDraftStrategy  # type: ignore
    from utils import DraftOutput as _DraftOutput, VerifyOutput as _VerifyOutput  # type: ignore
    from utils import LogitsProcessor as _LogitsProcessor, gumbel_sample as __gumbel_sample  # type: ignore

    OfficialInvariantMultiDraftStrategy = _OfficialInvariantMultiDraftStrategy
    DraftOutput = _DraftOutput
    VerifyOutput = _VerifyOutput
    LogitsProcessor = _LogitsProcessor
    _gumbel_sample = __gumbel_sample

    seed_everything(args.seed)

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # Only run official Ashish invariant configs. This lets you use either a
    # dedicated Ashish-only JSON or your larger three-method JSON.
    configs = [c for c in cfg["configurations"] if c.get("method") in {"ashish_invariant", "ashish_invariant_official"}]
    if not configs:
        raise ValueError("No configurations with method='ashish_invariant' found in JSON")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["target_model"], device_map="auto", torch_dtype=dtype)
    model.eval()
    ref_model = AutoModelForCausalLM.from_pretrained(cfg["draft_model"], device_map="auto", torch_dtype=dtype)
    ref_model.eval()

    gen_length = int(cfg.get("gen_length", 128))
    warmup = int(args.warmup if args.warmup is not None else cfg.get("warmup", 10))
    max_prompts = args.max_prompts
    if max_prompts is None:
        max_prompts = int(cfg.get("num_test_prompts", 200))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raw_path = out_dir / f"ashish_invariant_official_raw_seed{args.seed}_{stamp}.csv"
    summary_path = out_dir / f"ashish_invariant_official_summary_seed{args.seed}_{stamp}.csv"

    fieldnames = [
        "seed",
        "dataset",
        "method",
        "config_name",
        "test_num",
        "prompt_index",
        "max_draft_len",
        "max_num_drafts",
        "temperature",
        "top_k",
        "top_p",
        "num_generated_tokens",
        "total_time",
        "token_rate",
        "num_steps",
        "accepted_tokens_total",
        "attempted_draft_tokens_total",
        "aatps",
        "be",
        "tokens_per_step",
        "acceptance_fraction",
        "normalized_aatps",
        "avg_block_time",
        "target_forward_calls_total",
        "draft_forward_calls_total",
        "target_forward_calls_per_token",
        "draft_forward_calls_per_token",
        "avg_generation_time",
        "avg_verification_time",
    ]

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for dset_name in cfg["datasets"]:
            dset = load_prompt_dataset(dset_name)
            n_test = min(max_prompts, len(dset))
            n_warm = min(warmup, len(dset))

            for config in configs:
                name = config["name"]
                method = "ashish_invariant_official"
                top_k = int(config.get("top_k", cfg.get("top_k", 50)))
                top_p = float(config.get("top_p", cfg.get("top_p", 1.0)))
                config = dict(config)
                config["top_k"] = top_k
                config["top_p"] = top_p

                for i in tqdm(range(n_warm), total=n_warm, desc=f"Warmup {dset_name.split('/')[-1]} | {name}"):
                    prompt = create_prompt(dset[i], dset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_device(model))
                    _ = run_one_prompt_official_invariant(
                        model=model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                    )

                for i in tqdm(range(n_test), total=n_test, desc=f"Test {dset_name.split('/')[-1]} | {name}"):
                    prompt = create_prompt(dset[i], dset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_device(model))
                    stats = run_one_prompt_official_invariant(
                        model=model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        config=config,
                        gen_length=gen_length,
                    )
                    row = {
                        "seed": args.seed,
                        "dataset": dset_name,
                        "method": method,
                        "config_name": name.replace("ashish_invariant", "ashish_invariant_official"),
                        "test_num": i,
                        "prompt_index": i,
                        "max_draft_len": int(config["max_draft_len"]),
                        "max_num_drafts": int(config["max_num_drafts"]),
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
