"""
JSON-driven benchmark for three speculative decoding methods:
  1. Ashish/List-level strong multi-draft implementation from SpeculativeDecoding/
  2. MPFR torch-generator cached implementation from MPFR_spec/mpfr_batched_torchgen_cached.py
  3. MPFR multi-sample PFR cached implementation from MPFR_spec/multi_draft_pfr_batched_cached.py

Put this file at the parent folder containing both SpeculativeDecoding/ and MPFR_spec/.
Example root layout:
  ~/MPFR/
    SpeculativeDecoding/
    MPFR_spec/
    run_three_method_json_benchmark.py
    three_method_qwen25.json

The script writes one raw CSV and one summary CSV.  The raw CSV contains a seed
column, so later cleaning/combining across seeds is much easier.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SPEC_DIR = ROOT / "SpeculativeDecoding"
MPFR_DIR = ROOT / "MPFR_spec"
for p in [str(ROOT), str(SPEC_DIR), str(MPFR_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Ashish / list-level implementation from SpeculativeDecoding.
from generator import InvariantGenerator  # type: ignore
from strategy import StrongMultiDraftStrategy  # type: ignore

# New MPFR cached implementations.
from mpfr_batched_torchgen_cached import (  # type: ignore
    finite_multi_draft_pfr_cached_sample_generator,
)
from multi_draft_pfr_batched_cached import (  # type: ignore
    multi_draft_pfr_batched_cached_sample_generator,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def torch_dtype_from_name(name: str):
    name = str(name).lower()
    if name in {"auto", "none"}:
        return None
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype: {name}")


def load_named_dataset(name: str):
    if name == "openai/gsm8k":
        return load_dataset(name, "main")["train"]
    if name == "openai/openai_humaneval":
        return load_dataset(name, "openai_humaneval")["test"]
    if name == "facebook/natural_reasoning":
        return load_dataset(name, "default")["train"]
    if name == "ucinlp/drop":
        return load_dataset(name, "default")["train"]
    if name == "google-research-datasets/mbpp":
        return load_dataset(name, "full")["train"]
    raise ValueError(f"unsupported dataset: {name}")


def create_prompt(row: Dict[str, Any], dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name == "openai/gsm8k":
        user = row["question"]
    elif dataset_name == "openai/openai_humaneval":
        user = "Complete the code:\n" + row["prompt"]
    elif dataset_name == "facebook/natural_reasoning":
        user = row["question"]
    elif dataset_name == "ucinlp/drop":
        user = row["passage"] + "\n\n" + row["question"]
    elif dataset_name == "google-research-datasets/mbpp":
        user = row["text"] + " Use Python."
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]


def input_ids_from_prompt(tokenizer, prompt: List[Dict[str, str]], device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(encoded, torch.Tensor):
        return encoded.to(device)
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids.to(device)
    if isinstance(encoded, dict) and "input_ids" in encoded:
        return encoded["input_ids"].to(device)
    raise TypeError(f"unexpected apply_chat_template output type: {type(encoded)}")


def normalize_temperature(temp: Any, num_drafts: int) -> Any:
    # Ashish code expects length B+1 when a list is used.  MPFR code can also
    # read target/draft temperatures from the same list convention.
    if isinstance(temp, list):
        return temp
    return [float(temp)] * (int(num_drafts) + 1)


def build_mpfr_process_logits_kwargs(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    temp = config.get("temperature", 1.0)
    top_k = int(config.get("top_k", 50))
    top_p = float(config.get("top_p", 1.0))

    if method == "multi_draft_pfr_batched_cached":
        # This variant uses accuwm.utils.process_logits(logits_warper=...).
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        target_temp = temp[0] if isinstance(temp, list) else temp
        parts = []
        if target_temp is not None and float(target_temp) != 1.0:
            parts.append(TemperatureLogitsWarper(float(target_temp)))
        if top_k is not None and int(top_k) > 0:
            parts.append(TopKLogitsWarper(int(top_k)))
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            parts.append(TopPLogitsWarper(float(top_p)))
        if not parts:
            return {}
        warper = LogitsProcessorList(parts)
        return {"logits_warper": lambda input_ids, logits: warper(input_ids, logits)}

    # torchgen cached variant uses scalar config and its own exact processor.
    return {"temperature": temp, "top_k": top_k, "top_p": top_p}


@dataclass
class RunStats:
    num_generated_tokens: int
    total_time: float
    token_rate: float
    num_steps: int
    accepted_tokens_total: float
    attempted_draft_tokens_total: float
    aatps: float
    be: float
    tokens_per_step: float
    acceptance_fraction: float
    normalized_aatps: float
    avg_block_time: float
    target_forward_calls_total: float
    draft_forward_calls_total: float
    target_forward_calls_per_token: float
    draft_forward_calls_per_token: float


def run_ashish_strong(
    *,
    model,
    ref_model,
    tokenizer,
    input_ids: torch.Tensor,
    config: Dict[str, Any],
    gen_length: int,
) -> RunStats:
    L = int(config.get("max_draft_len", 4))
    B = int(config.get("max_num_drafts", 2))
    top_k = int(config.get("top_k", 50))
    top_p = float(config.get("top_p", 1.0))
    temp = normalize_temperature(config.get("temperature", 1.0), B)

    strategy = StrongMultiDraftStrategy(model, ref_model, tokenizer, L, B)
    generator = InvariantGenerator(strategy)

    cuda_synchronize()
    t0 = time.perf_counter()
    out = generator(
        input_ids=input_ids,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=gen_length,
    )
    cuda_synchronize()
    elapsed = time.perf_counter() - t0

    input_len = int(input_ids.shape[-1])
    num_tokens = int(out.sequences.shape[-1] - input_len)
    steps = int(out.num_invocations)
    be = num_tokens / max(steps, 1)

    # In the original Ashish generator, output.acceptance_rate is actually
    # mean accept_count per invocation; accept_count includes the extra emitted
    # target/residual token.  Thus AATPS ~= acceptance_rate - 1.
    accepted_plus_one = float(out.acceptance_rate)
    aatps = max(accepted_plus_one - 1.0, 0.0)
    accepted_total = aatps * steps
    attempted_total = min(L * steps, max(num_tokens, 0) + max(steps, 0) * (L - 1))
    acceptance_fraction = accepted_total / attempted_total if attempted_total > 0 else float("nan")

    target_calls = float(steps)
    draft_calls = float(steps * L)

    return RunStats(
        num_generated_tokens=num_tokens,
        total_time=float(elapsed),
        token_rate=float(num_tokens / elapsed) if elapsed > 0 else float("nan"),
        num_steps=steps,
        accepted_tokens_total=float(accepted_total),
        attempted_draft_tokens_total=float(attempted_total),
        aatps=float(aatps),
        be=float(be),
        tokens_per_step=float(be),
        acceptance_fraction=float(acceptance_fraction),
        normalized_aatps=float(aatps / L) if L > 0 else float("nan"),
        avg_block_time=float(elapsed / max(steps, 1)),
        target_forward_calls_total=target_calls,
        draft_forward_calls_total=draft_calls,
        target_forward_calls_per_token=float(target_calls / num_tokens) if num_tokens > 0 else float("nan"),
        draft_forward_calls_per_token=float(draft_calls / num_tokens) if num_tokens > 0 else float("nan"),
    )


def run_mpfr_generator(
    *,
    method: str,
    model,
    ref_model,
    tokenizer,
    input_ids: torch.Tensor,
    config: Dict[str, Any],
    gen_length: int,
    private_key: str | bytes,
) -> RunStats:
    L = int(config.get("max_draft_len", 4))
    B = int(config.get("max_num_drafts", 2))
    if method == "mpfr_batched_torchgen_cached":
        fn = finite_multi_draft_pfr_cached_sample_generator
    elif method == "multi_draft_pfr_batched_cached":
        fn = multi_draft_pfr_batched_cached_sample_generator
    else:
        raise ValueError(f"unknown MPFR method: {method}")

    process_logits_kwargs = build_mpfr_process_logits_kwargs(config, method)
    total_tokens = 0
    accepted_total = 0
    attempted_total = 0
    steps = 0
    target_calls = 0
    draft_calls = 0

    gen = fn(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        n=L,
        max_length=gen_length,
        private_key=private_key,
        num_drafts=B,
        process_logits_kwargs=process_logits_kwargs,
        return_meta=True,
    )

    cuda_synchronize()
    t0 = time.perf_counter()
    for yielded in gen:
        if len(yielded) == 3:
            output_ids, _logp, meta = yielded
        else:
            output_ids, _logp = yielded
            meta = {}
        cuda_synchronize()
        emitted = int(output_ids.shape[-1])
        total_tokens += emitted
        steps += 1
        accepted_total += int(meta.get("accepted_count", max(emitted - 1, 0)))
        attempted_total += int(meta.get("attempted_draft_tokens", min(L, max(gen_length - (total_tokens - emitted) - 1, 0))))
        target_calls += int(meta.get("target_forward_calls", 1))
        draft_calls += int(meta.get("draft_forward_calls", L))
        if tokenizer.eos_token_id is not None and bool((output_ids == tokenizer.eos_token_id).any().item()):
            break
        if total_tokens >= gen_length:
            break
    cuda_synchronize()
    elapsed = time.perf_counter() - t0

    be = total_tokens / max(steps, 1)
    aatps = accepted_total / max(steps, 1)
    return RunStats(
        num_generated_tokens=int(total_tokens),
        total_time=float(elapsed),
        token_rate=float(total_tokens / elapsed) if elapsed > 0 else float("nan"),
        num_steps=int(steps),
        accepted_tokens_total=float(accepted_total),
        attempted_draft_tokens_total=float(attempted_total),
        aatps=float(aatps),
        be=float(be),
        tokens_per_step=float(be),
        acceptance_fraction=float(accepted_total / attempted_total) if attempted_total > 0 else float("nan"),
        normalized_aatps=float(aatps / L) if L > 0 else float("nan"),
        avg_block_time=float(elapsed / max(steps, 1)),
        target_forward_calls_total=float(target_calls),
        draft_forward_calls_total=float(draft_calls),
        target_forward_calls_per_token=float(target_calls / total_tokens) if total_tokens > 0 else float("nan"),
        draft_forward_calls_per_token=float(draft_calls / total_tokens) if total_tokens > 0 else float("nan"),
    )


def summarize(raw_csv: Path, summary_csv: Path) -> None:
    import pandas as pd
    df = pd.read_csv(raw_csv)
    metric_cols = [
        "num_generated_tokens", "total_time", "token_rate", "num_steps",
        "accepted_tokens_total", "attempted_draft_tokens_total", "aatps", "be",
        "tokens_per_step", "acceptance_fraction", "normalized_aatps", "avg_block_time",
        "target_forward_calls_total", "draft_forward_calls_total",
        "target_forward_calls_per_token", "draft_forward_calls_per_token",
    ]
    summary = df.groupby(["seed", "dataset", "method", "config_name"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(summary_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output_dir", default="outputs_three_method")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = json.loads(Path(args.config).read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    raw_csv = out_dir / f"three_method_raw_seed{args.seed}_{stamp}.csv"
    summary_csv = out_dir / f"three_method_summary_seed{args.seed}_{stamp}.csv"

    dtype = torch_dtype_from_name(args.dtype)
    model_kwargs = dict(device_map=args.device_map, trust_remote_code=args.trust_remote_code)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_large = AutoModelForCausalLM.from_pretrained(cfg["target_model"], **model_kwargs).eval()
    model_small = AutoModelForCausalLM.from_pretrained(cfg["draft_model"], **model_kwargs).eval()

    gen_length = int(cfg.get("gen_length", 128))
    private_key = cfg.get("private_key", "1234")
    warmup = int(cfg.get("warmup", 10) if args.warmup is None else args.warmup)

    fields = [
        "seed", "dataset", "method", "config_name", "test_num", "prompt_index",
        "max_draft_len", "max_num_drafts", "temperature", "top_k", "top_p",
        *RunStats.__dataclass_fields__.keys(),
    ]

    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset_name in cfg["datasets"]:
            dataset = load_named_dataset(dataset_name)
            n = int(cfg.get("num_test_prompts", 200))
            if args.max_prompts is not None:
                n = min(n, int(args.max_prompts))
            if n < 0 or n > len(dataset):
                n = len(dataset)

            for config in cfg["configurations"]:
                method = str(config["method"])
                config_name = str(config["name"])
                desc = f"{dataset_name.split('/')[-1]} | {config_name}"

                for i in tqdm(range(min(warmup, len(dataset))), desc=f"Warmup {desc}"):
                    prompt = create_prompt(dataset[i], dataset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_large.device)
                    if method == "ashish_strong":
                        _ = run_ashish_strong(model=model_large, ref_model=model_small, tokenizer=tokenizer, input_ids=input_ids, config=config, gen_length=gen_length)
                    else:
                        _ = run_mpfr_generator(method=method, model=model_large, ref_model=model_small, tokenizer=tokenizer, input_ids=input_ids, config=config, gen_length=gen_length, private_key=private_key)

                for i in tqdm(range(n), desc=f"Test {desc}"):
                    prompt = create_prompt(dataset[i], dataset_name)
                    input_ids = input_ids_from_prompt(tokenizer, prompt, model_large.device)
                    if method == "ashish_strong":
                        stats = run_ashish_strong(model=model_large, ref_model=model_small, tokenizer=tokenizer, input_ids=input_ids, config=config, gen_length=gen_length)
                    elif method in {"mpfr_batched_torchgen_cached", "multi_draft_pfr_batched_cached"}:
                        stats = run_mpfr_generator(method=method, model=model_large, ref_model=model_small, tokenizer=tokenizer, input_ids=input_ids, config=config, gen_length=gen_length, private_key=private_key)
                    else:
                        raise ValueError(f"unknown method: {method}")

                    row = {
                        "seed": args.seed,
                        "dataset": dataset_name,
                        "method": method,
                        "config_name": config_name,
                        "test_num": i,
                        "prompt_index": i,
                        "max_draft_len": int(config.get("max_draft_len", 4)),
                        "max_num_drafts": int(config.get("max_num_drafts", 2)),
                        "temperature": json.dumps(config.get("temperature", 1.0)),
                        "top_k": int(config.get("top_k", cfg.get("top_k", 50))),
                        "top_p": float(config.get("top_p", cfg.get("top_p", 1.0))),
                        **asdict(stats),
                    }
                    writer.writerow(row)
                    f.flush()

    summarize(raw_csv, summary_csv)
    print(f"Wrote raw:     {raw_csv}")
    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()
