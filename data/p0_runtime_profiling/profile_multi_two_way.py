#!/usr/bin/env python3
"""Two-way MPFR/INVARIANT runtime breakdown for rebuttal diagnostics.

End-to-end passes are uninstrumented. A second synchronized pass attributes
runtime to model forwards, logits processing, and each method's sampling
primitive. The synchronized pass is diagnostic and must not be used as the
headline throughput measurement.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MPFR_spec") not in sys.path:
    sys.path.insert(0, str(ROOT / "MPFR_spec"))

import accuwm.invariant_multi as invariant_core  # noqa: E402
import accuwm.pfr as pfr_core  # noqa: E402
import mpfr_batched_torchgen_cached as mpfr_cached  # noqa: E402
from experiments._shared import _maybe_inject_chat_template, load_prompts  # noqa: E402
from profile_mpfr_runtime import (  # noqa: E402
    PROMPTS,
    PhaseTotals,
    sync,
    wrap_forward,
)


EXCLUSIVE_PHASES = (
    "target_forward",
    "draft_forward",
    "mpfr_logits_processing",
    "pfr_arrival_sampling",
    "invariant_logits_processing",
    "invariant_gumbel_sampling",
    "mpfr_cache_gather",
    "mpfr_cache_repeat",
    "mpfr_target_cache_select",
    "mpfr_draft_cache_select",
)
SUBSET_PHASES = ("keyed_uniform_rng",)


def summarize(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def encode_any_prompt(tokenizer, prompt, device):
    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            text = "\n".join(
                message["content"] for message in prompt
                if message["role"] == "user"
            )
            ids = tokenizer(text, return_tensors="pt").input_ids
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids.to(device)


def reset_and_snapshot_memory():
    allocated = []
    for index in range(torch.cuda.device_count()):
        allocated.append(int(torch.cuda.memory_allocated(index)))
        torch.cuda.reset_peak_memory_stats(index)
    return allocated


def peak_memory_rows(allocated_before):
    peaks = [
        int(torch.cuda.max_memory_allocated(index))
        for index in range(torch.cuda.device_count())
    ]
    return {
        "peak_allocated_bytes_per_gpu": peaks,
        "incremental_peak_allocated_bytes_per_gpu": [
            max(peak - baseline, 0)
            for peak, baseline in zip(peaks, allocated_before)
        ],
    }


def serializable_device_map(model):
    mapping = getattr(model, "hf_device_map", None)
    if mapping is None:
        return None
    return {name: str(device) for name, device in mapping.items()}


def drain_mpfr(
    model, draft, input_ids, lookahead, num_drafts, max_new_tokens, top_k
):
    generator = mpfr_cached.finite_multi_draft_pfr_cached_sample_generator(
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_new_tokens,
        num_drafts=num_drafts,
        private_key=b"p0-profile-key",
        process_logits_kwargs={
            "temperature": 1.0,
            "top_k": top_k,
            "top_p": 1.0,
        },
        return_meta=True,
        return_logprobs=False,
    )
    stats = {
        "tokens": 0,
        "blocks": 0,
        "accepted": 0,
        "draft_tree_size": 0,
        "target_context_count": 0,
        "target_forward_calls_meta": 0,
        "draft_forward_calls_meta": 0,
    }
    for ids, _logprobs, meta in generator:
        stats["tokens"] += int(ids.shape[-1])
        stats["blocks"] += 1
        stats["accepted"] += int(meta["accepted_count"])
        stats["draft_tree_size"] += int(meta["draft_tree_size"])
        stats["target_context_count"] += int(meta["target_context_count"])
        stats["target_forward_calls_meta"] += int(
            meta["target_forward_calls"]
        )
        stats["draft_forward_calls_meta"] += int(
            meta["draft_forward_calls"]
        )
    return stats


def make_invariant_runner(
    model, draft, lookahead, num_drafts, max_new_tokens, top_k
):
    effective_vocab = min(
        int(model.config.vocab_size), int(draft.config.vocab_size)
    )

    class StubTokenizer:
        vocab_size = effective_vocab

    strategy = invariant_core.InvariantMultiDraftStrategy(
        target=model,
        drafter=draft,
        tokenizer=StubTokenizer(),
        max_draft_len=lookahead,
        max_num_drafts=num_drafts,
    )
    generator = invariant_core.InvariantGenerator(strategy)
    eos_token_id = int(model.config.eos_token_id)

    def run(input_ids):
        output = generator(
            input_ids=input_ids,
            eos_token_id=eos_token_id,
            temperature=1.0,
            top_k=(top_k if top_k > 0 else effective_vocab),
            max_new_tokens=max_new_tokens,
        )
        return {
            "tokens": int(
                output.sequences.shape[-1] - input_ids.shape[-1]
            ),
            "blocks": int(output.num_invocations),
            "accepted": float(
                output.acceptance_rate * output.num_invocations
            ),
        }

    return run


def install_component_wrappers(totals):
    originals = {
        "mpfr_logits": mpfr_cached.process_logits_exact,
        "mpfr_sampler": mpfr_cached.ms_pfr_tokens_from_logprobs,
        "uniform": pfr_core.SharedPFRSource.uniform_noise,
        "invariant_logits": invariant_core.LogitsProcessor.__call__,
        "invariant_gumbel": invariant_core.gumbel_sample,
        "cache_gather": mpfr_cached._gather_cache_rows,
        "cache_repeat": mpfr_cached._repeat_cache,
        "target_cache_select": mpfr_cached._select_and_truncate_cache,
        "draft_cache_select": mpfr_cached._select_cache_row,
    }

    def mpfr_logits(*args, **kwargs):
        with totals.phase("mpfr_logits_processing"):
            return originals["mpfr_logits"](*args, **kwargs)

    def mpfr_sampler(*args, **kwargs):
        with totals.phase("pfr_arrival_sampling"):
            return originals["mpfr_sampler"](*args, **kwargs)

    def uniform(self, *args, **kwargs):
        with totals.phase("keyed_uniform_rng"):
            return originals["uniform"](self, *args, **kwargs)

    def invariant_logits(self, *args, **kwargs):
        with totals.phase("invariant_logits_processing"):
            return originals["invariant_logits"](self, *args, **kwargs)

    def invariant_gumbel(*args, **kwargs):
        with totals.phase("invariant_gumbel_sampling"):
            return originals["invariant_gumbel"](*args, **kwargs)

    def cache_gather(*args, **kwargs):
        with totals.phase("mpfr_cache_gather"):
            return originals["cache_gather"](*args, **kwargs)

    def cache_repeat(*args, **kwargs):
        with totals.phase("mpfr_cache_repeat"):
            return originals["cache_repeat"](*args, **kwargs)

    def target_cache_select(*args, **kwargs):
        with totals.phase("mpfr_target_cache_select"):
            return originals["target_cache_select"](*args, **kwargs)

    def draft_cache_select(*args, **kwargs):
        with totals.phase("mpfr_draft_cache_select"):
            return originals["draft_cache_select"](*args, **kwargs)

    mpfr_cached.process_logits_exact = mpfr_logits
    mpfr_cached.ms_pfr_tokens_from_logprobs = mpfr_sampler
    pfr_core.SharedPFRSource.uniform_noise = uniform
    invariant_core.LogitsProcessor.__call__ = invariant_logits
    invariant_core.gumbel_sample = invariant_gumbel
    mpfr_cached._gather_cache_rows = cache_gather
    mpfr_cached._repeat_cache = cache_repeat
    mpfr_cached._select_and_truncate_cache = target_cache_select
    mpfr_cached._select_cache_row = draft_cache_select
    return originals


def restore_component_wrappers(originals):
    mpfr_cached.process_logits_exact = originals["mpfr_logits"]
    mpfr_cached.ms_pfr_tokens_from_logprobs = originals["mpfr_sampler"]
    pfr_core.SharedPFRSource.uniform_noise = originals["uniform"]
    invariant_core.LogitsProcessor.__call__ = originals["invariant_logits"]
    invariant_core.gumbel_sample = originals["invariant_gumbel"]
    mpfr_cached._gather_cache_rows = originals["cache_gather"]
    mpfr_cached._repeat_cache = originals["cache_repeat"]
    mpfr_cached._select_and_truncate_cache = originals["target_cache_select"]
    mpfr_cached._select_cache_row = originals["draft_cache_select"]


def benchmark_method(
    *, name, run, model, tokenizer, prompts, profile_prompts, warmup,
    totals, seed_offset
):
    def run_one(prompt, seed):
        input_ids = encode_any_prompt(tokenizer, prompt, model.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        sync()
        allocated_before = reset_and_snapshot_memory()
        start = time.perf_counter()
        stats = run(input_ids)
        sync()
        elapsed = time.perf_counter() - start
        return {
            **stats,
            "elapsed_sec": elapsed,
            "token_rate": stats["tokens"] / elapsed,
            "ms_per_block": 1000.0 * elapsed / stats["blocks"],
            **peak_memory_rows(allocated_before),
        }

    totals.enabled = False
    for i in range(warmup):
        run_one(prompts[i % len(prompts)], seed_offset + i)

    rows = [
        run_one(prompt, seed_offset + 1_000 + i)
        for i, prompt in enumerate(prompts)
    ]

    totals.reset()
    totals.enabled = True
    profile_rows = [
        run_one(prompt, seed_offset + 1_000 + i)
        for i, prompt in enumerate(profile_prompts)
    ]
    totals.enabled = False

    tokens = sum(row["tokens"] for row in rows)
    blocks = sum(row["blocks"] for row in rows)
    elapsed = sum(row["elapsed_sec"] for row in rows)
    prof_tokens = sum(row["tokens"] for row in profile_rows)
    prof_blocks = sum(row["blocks"] for row in profile_rows)
    prof_elapsed = sum(row["elapsed_sec"] for row in profile_rows)
    phase_sec = {
        phase: float(totals.seconds.get(phase, 0.0))
        for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
    }
    exclusive_sum = sum(phase_sec[p] for p in EXCLUSIVE_PHASES)
    remainder = max(prof_elapsed - exclusive_sum, 0.0)

    return {
        "method": name,
        "end_to_end": {
            "tokens": tokens,
            "blocks": blocks,
            "elapsed_sec": elapsed,
            "token_rate_global": tokens / elapsed,
            "token_rate_per_prompt": summarize(
                [row["token_rate"] for row in rows]
            ),
            "tokens_per_block": tokens / blocks,
            "ms_per_block": 1000.0 * elapsed / blocks,
            "ms_per_block_per_prompt": summarize(
                [row["ms_per_block"] for row in rows]
            ),
            "peak_allocated_gib_per_gpu": [
                max(row["peak_allocated_bytes_per_gpu"][index] for row in rows)
                / (1024**3)
                for index in range(torch.cuda.device_count())
            ],
            "incremental_peak_allocated_mib_per_gpu": [
                max(
                    row["incremental_peak_allocated_bytes_per_gpu"][index]
                    for row in rows
                ) / (1024**2)
                for index in range(torch.cuda.device_count())
            ],
            "rows": rows,
        },
        "instrumented": {
            "tokens": prof_tokens,
            "blocks": prof_blocks,
            "elapsed_sec": prof_elapsed,
            "tokens_per_block": prof_tokens / prof_blocks,
            "ms_per_block": 1000.0 * prof_elapsed / prof_blocks,
            "phase_sec": phase_sec,
            "phase_calls": {
                phase: int(totals.calls.get(phase, 0))
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "phase_ms_per_block": {
                phase: 1000.0 * phase_sec[phase] / prof_blocks
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "phase_fraction": {
                phase: phase_sec[phase] / prof_elapsed
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "remainder_sec": remainder,
            "remainder_ms_per_block": 1000.0 * remainder / prof_blocks,
            "remainder_fraction": remainder / prof_elapsed,
            "rows": profile_rows,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target-device-map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
    )
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16"], default="float16"
    )
    parser.add_argument(
        "--dataset",
        choices=("cnn_dailymail", "eli5", "fixed"),
        default="cnn_dailymail",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--profile-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--draft-counts", type=int, nargs="+", default=[2, 4, 6, 8]
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("mpfr", "invariant"),
        default=["mpfr", "invariant"],
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Use 0 to disable top-k truncation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = getattr(torch, args.dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.target)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target, use_fast=False
        )
    _maybe_inject_chat_template(tokenizer, args.target)
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=dtype,
        device_map=args.target_device_map or args.device,
        low_cpu_mem_usage=True,
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft,
        torch_dtype=dtype,
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()

    profile_samples = (
        args.samples if args.profile_samples is None else args.profile_samples
    )
    if profile_samples <= 0:
        raise ValueError("--profile-samples must be positive")
    prompt_count = max(args.samples, profile_samples)
    if args.dataset == "fixed":
        all_prompts = [
            PROMPTS[i % len(PROMPTS)] for i in range(prompt_count)
        ]
    else:
        all_prompts = load_prompts(args.dataset, prompt_count)
    prompts = all_prompts[:args.samples]
    profile_prompts = all_prompts[:profile_samples]

    totals = PhaseTotals()
    target_forward = wrap_forward(model, "target_forward", totals)
    draft_forward = wrap_forward(draft, "draft_forward", totals)
    originals = install_component_wrappers(totals)
    results = []
    try:
        for b in args.draft_counts:
            runners = {}
            if "mpfr" in args.methods:
                runners["mpfr"] = lambda input_ids, b=b: drain_mpfr(
                    model, draft, input_ids, args.lookahead, b,
                    args.max_new_tokens, args.top_k,
                )
            if "invariant" in args.methods:
                runners["invariant"] = make_invariant_runner(
                    model, draft, args.lookahead, b, args.max_new_tokens,
                    args.top_k,
                )
            for method, run in runners.items():
                print(
                    f"[multi] dataset={args.dataset} B={b} method={method}",
                    flush=True,
                )
                result = benchmark_method(
                    name=method,
                    run=run,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    profile_prompts=profile_prompts,
                    warmup=args.warmup,
                    totals=totals,
                    seed_offset=40_000 + 10_000 * b,
                )
                result["num_drafts"] = b
                results.append(result)
                e2e = result["end_to_end"]
                print(
                    f"  TR={e2e['token_rate_global']:.2f} "
                    f"AATPS={e2e['tokens_per_block']:.3f} "
                    f"block={e2e['ms_per_block']:.2f}ms "
                    f"max-peak+="
                    f"{max(e2e['incremental_peak_allocated_mib_per_gpu']):.1f}MiB",
                    flush=True,
                )
    finally:
        totals.enabled = False
        model.forward = target_forward
        draft.forward = draft_forward
        restore_component_wrappers(originals)

    payload = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "torch": torch.__version__,
        },
        "config": {
            "target": args.target,
            "draft": args.draft,
            "device": args.device,
            "target_device_map_requested": args.target_device_map,
            "target_device_map_actual": serializable_device_map(model),
            "draft_device_map_actual": serializable_device_map(draft),
            "dtype": args.dtype,
            "dataset": args.dataset,
            "samples": args.samples,
            "profile_samples": profile_samples,
            "warmup": args.warmup,
            "lookahead": args.lookahead,
            "max_new_tokens": args.max_new_tokens,
            "draft_counts": args.draft_counts,
            "methods": args.methods,
            "top_k": args.top_k,
            "top_p": 1.0,
            "temperature": 1.0,
            "seed_protocol": (
                "one paired seed per prompt; identical across methods and "
                "end-to-end/instrumented passes"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
