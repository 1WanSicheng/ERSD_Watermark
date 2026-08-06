#!/usr/bin/env python3
"""First-pass synchronized runtime breakdown for MPFR.

This script monkey-patches narrow runtime boundaries instead of changing the
production decoder. Timers synchronize CUDA before and after each measured
region, so phase totals are attributable but slower than normal execution.
An uninstrumented pass is reported separately for end-to-end throughput.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MPFR_spec") not in sys.path:
    sys.path.insert(0, str(ROOT / "MPFR_spec"))

import accuwm.multi_draft_utils as multi_utils  # noqa: E402
import accuwm.pfr as pfr_core  # noqa: E402
import mpfr_batched_torchgen_cached as mpfr_cached  # noqa: E402
from accuwm.invariant_multi import (  # noqa: E402
    InvariantGenerator,
    InvariantMultiDraftStrategy,
)


PROMPTS = [
    "Summarize the role of renewable energy in reducing carbon emissions.",
    "Explain why the sky appears blue during the day.",
    "Describe the main causes and consequences of inflation.",
    "Summarize how vaccines train the immune system.",
    "Explain the difference between weather and climate.",
    "Describe how a computer stores information in memory.",
    "Summarize the benefits and risks of artificial intelligence.",
    "Explain why ocean tides occur.",
]


def sync() -> None:
    # A dispatched large model can have outstanding kernels on several GPUs.
    # Synchronizing only the input device under-counts model-parallel latency.
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


class PhaseTotals:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.enabled = False

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            sync()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()


def wrap_forward(model, phase_name: str, totals: PhaseTotals):
    original = model.forward

    def wrapped(*args, **kwargs):
        with totals.phase(phase_name):
            return original(*args, **kwargs)

    model.forward = wrapped
    return original


def install_sampling_wrappers(totals: PhaseTotals):
    original_sampler = mpfr_cached.ms_pfr_tokens_from_logprobs
    original_uniform = pfr_core.SharedPFRSource.uniform_noise

    def wrapped_sampler(*args, **kwargs):
        with totals.phase("pfr_arrival_sampling"):
            return original_sampler(*args, **kwargs)

    def wrapped_uniform(self, *args, **kwargs):
        with totals.phase("keyed_uniform_rng"):
            return original_uniform(self, *args, **kwargs)

    # The cached MPFR module imports the sampler by name, so patch that alias.
    mpfr_cached.ms_pfr_tokens_from_logprobs = wrapped_sampler
    # Patch the class method used by every SharedPFRSource instance.
    pfr_core.SharedPFRSource.uniform_noise = wrapped_uniform
    # Keep the utility alias aligned in case another path reaches it.
    multi_utils.ms_pfr_tokens_from_logprobs = wrapped_sampler
    return original_sampler, original_uniform


def restore_sampling_wrappers(original_sampler, original_uniform) -> None:
    mpfr_cached.ms_pfr_tokens_from_logprobs = original_sampler
    multi_utils.ms_pfr_tokens_from_logprobs = original_sampler
    pfr_core.SharedPFRSource.uniform_noise = original_uniform


def encode_prompt(tokenizer, text: str, device) -> torch.LongTensor:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        ids = tokenizer(text, return_tensors="pt").input_ids
    return ids.to(device)


def drain_mpfr(
    *,
    model,
    draft,
    input_ids,
    lookahead: int,
    num_drafts: int,
    max_new_tokens: int,
) -> int:
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
            "top_k": 50,
            "top_p": 1.0,
        },
        return_meta=True,
        return_logprobs=False,
    )
    generated = 0
    for ids, _logprobs, _meta in generator:
        generated += int(ids.shape[-1])
        if generated >= max_new_tokens:
            break
    return generated


def run_once(
    *,
    model,
    draft,
    tokenizer,
    prompt: str,
    lookahead: int,
    num_drafts: int,
    max_new_tokens: int,
) -> tuple[int, float, int]:
    input_ids = encode_prompt(tokenizer, prompt, model.device)
    torch.cuda.reset_peak_memory_stats()
    sync()
    start = time.perf_counter()
    tokens = drain_mpfr(
        model=model,
        draft=draft,
        input_ids=input_ids,
        lookahead=lookahead,
        num_drafts=num_drafts,
        max_new_tokens=max_new_tokens,
    )
    sync()
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated())
    return tokens, elapsed, peak


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def benchmark_plain_method(
    *,
    name: str,
    run: Callable[[torch.LongTensor], int],
    model,
    tokenizer,
    prompts: list[str],
    warmup: int,
) -> dict:
    for i in range(warmup):
        input_ids = encode_prompt(tokenizer, prompts[i % len(prompts)], model.device)
        run(input_ids)

    rows = []
    for i, prompt in enumerate(prompts):
        input_ids = encode_prompt(tokenizer, prompt, model.device)
        torch.manual_seed(10_000 + i)
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        tokens = run(input_ids)
        sync()
        elapsed = time.perf_counter() - start
        rows.append({
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "token_rate": tokens / elapsed,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        })

    total_tokens = sum(row["tokens"] for row in rows)
    total_time = sum(row["elapsed_sec"] for row in rows)
    return {
        "method": name,
        "samples": len(prompts),
        "tokens": total_tokens,
        "elapsed_sec": total_time,
        "token_rate_global": total_tokens / total_time,
        "token_rate": summarize([row["token_rate"] for row in rows]),
        "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows)
        / (1024**3),
        "rows": rows,
    }


def benchmark_target_only(
    *, model, tokenizer, prompts, warmup: int, max_new_tokens: int,
) -> dict:
    def run(input_ids):
        out = model.generate(
            input_ids,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        return int(out.shape[-1] - input_ids.shape[-1])

    return benchmark_plain_method(
        name="target_only",
        run=run,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        warmup=warmup,
    )


def benchmark_invariant(
    *,
    model,
    draft,
    tokenizer,
    prompts,
    warmup: int,
    lookahead: int,
    num_drafts: int,
    max_new_tokens: int,
) -> dict:
    effective_vocab = min(int(model.config.vocab_size), int(draft.config.vocab_size))

    class StubTokenizer:
        vocab_size = effective_vocab

    strategy = InvariantMultiDraftStrategy(
        target=model,
        drafter=draft,
        tokenizer=StubTokenizer(),
        max_draft_len=lookahead,
        max_num_drafts=num_drafts,
    )
    generator = InvariantGenerator(strategy)
    eos_token_id = int(model.config.eos_token_id)

    def run(input_ids):
        out = generator(
            input_ids=input_ids,
            eos_token_id=eos_token_id,
            temperature=1.0,
            top_k=50,
            max_new_tokens=max_new_tokens,
        )
        return int(out.sequences.shape[-1] - input_ids.shape[-1])

    return benchmark_plain_method(
        name=f"invariant_B{num_drafts}",
        run=run,
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        warmup=warmup,
    )


def benchmark_b(
    *,
    model,
    draft,
    tokenizer,
    prompts: list[str],
    warmup: int,
    lookahead: int,
    num_drafts: int,
    max_new_tokens: int,
    totals: PhaseTotals,
) -> dict:
    totals.enabled = False
    for i in range(warmup):
        run_once(
            model=model,
            draft=draft,
            tokenizer=tokenizer,
            prompt=prompts[i % len(prompts)],
            lookahead=lookahead,
            num_drafts=num_drafts,
            max_new_tokens=max_new_tokens,
        )

    # Normal end-to-end pass: no phase synchronization overhead.
    e2e_rows = []
    totals.enabled = False
    for prompt in prompts:
        tokens, elapsed, peak = run_once(
            model=model,
            draft=draft,
            tokenizer=tokenizer,
            prompt=prompt,
            lookahead=lookahead,
            num_drafts=num_drafts,
            max_new_tokens=max_new_tokens,
        )
        e2e_rows.append({
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "token_rate": tokens / elapsed,
            "peak_allocated_bytes": peak,
        })

    # Instrumented pass: synchronized phase attribution.
    totals.reset()
    totals.enabled = True
    profile_tokens = 0
    profile_elapsed = 0.0
    profile_peak = 0
    for prompt in prompts:
        tokens, elapsed, peak = run_once(
            model=model,
            draft=draft,
            tokenizer=tokenizer,
            prompt=prompt,
            lookahead=lookahead,
            num_drafts=num_drafts,
            max_new_tokens=max_new_tokens,
        )
        profile_tokens += tokens
        profile_elapsed += elapsed
        profile_peak = max(profile_peak, peak)
    totals.enabled = False

    target = totals.seconds.get("target_forward", 0.0)
    draft_time = totals.seconds.get("draft_forward", 0.0)
    sampling = totals.seconds.get("pfr_arrival_sampling", 0.0)
    rng = totals.seconds.get("keyed_uniform_rng", 0.0)
    # Sampling includes RNG, so do not add RNG again.
    accounted = target + draft_time + sampling
    remainder = max(profile_elapsed - accounted, 0.0)

    e2e_total_tokens = sum(r["tokens"] for r in e2e_rows)
    e2e_total_time = sum(r["elapsed_sec"] for r in e2e_rows)
    return {
        "num_drafts": num_drafts,
        "lookahead": lookahead,
        "samples": len(prompts),
        "max_new_tokens": max_new_tokens,
        "end_to_end": {
            "tokens": e2e_total_tokens,
            "elapsed_sec": e2e_total_time,
            "token_rate_global": e2e_total_tokens / e2e_total_time,
            "token_rate": summarize([r["token_rate"] for r in e2e_rows]),
            "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in e2e_rows)
            / (1024**3),
            "rows": e2e_rows,
        },
        "instrumented": {
            "tokens": profile_tokens,
            "elapsed_sec": profile_elapsed,
            "token_rate": profile_tokens / profile_elapsed,
            "peak_allocated_gib": profile_peak / (1024**3),
            "phase_sec": dict(totals.seconds),
            "phase_calls": dict(totals.calls),
            "derived_sec": {
                "target_forward": target,
                "draft_forward": draft_time,
                "pfr_arrival_sampling_including_rng": sampling,
                "keyed_uniform_rng_subset": rng,
                "remainder_tree_cache_python": remainder,
            },
            "derived_fraction": {
                "target_forward": target / profile_elapsed,
                "draft_forward": draft_time / profile_elapsed,
                "pfr_arrival_sampling_including_rng": sampling / profile_elapsed,
                "keyed_uniform_rng_subset": rng / profile_elapsed,
                "remainder_tree_cache_python": remainder / profile_elapsed,
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-counts", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--output", type=Path, default=Path("p0_first_pass.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiling script")

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    model = AutoModelForCausalLM.from_pretrained(
        args.target,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    ).eval()

    totals = PhaseTotals()
    target_forward = wrap_forward(model, "target_forward", totals)
    draft_forward = wrap_forward(draft, "draft_forward", totals)
    original_sampler, original_uniform = install_sampling_wrappers(totals)

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.samples)]
    totals.enabled = False
    print("[P0] benchmarking target-only baseline", flush=True)
    target_only = benchmark_target_only(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"  e2e={target_only['token_rate_global']:.2f} tok/s",
        flush=True,
    )

    results = []
    invariant_results = []
    try:
        for b in args.draft_counts:
            print(f"[P0] profiling B={b}", flush=True)
            result = benchmark_b(
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                prompts=prompts,
                warmup=args.warmup,
                lookahead=args.lookahead,
                num_drafts=b,
                max_new_tokens=args.max_new_tokens,
                totals=totals,
            )
            results.append(result)
            fractions = result["instrumented"]["derived_fraction"]
            print(
                f"  e2e={result['end_to_end']['token_rate_global']:.2f} tok/s "
                f"target={100*fractions['target_forward']:.1f}% "
                f"draft={100*fractions['draft_forward']:.1f}% "
                f"sampling={100*fractions['pfr_arrival_sampling_including_rng']:.1f}% "
                f"remainder={100*fractions['remainder_tree_cache_python']:.1f}%",
                flush=True,
            )
            print(f"[P0] benchmarking INVARIANT B={b}", flush=True)
            invariant = benchmark_invariant(
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                prompts=prompts,
                warmup=args.warmup,
                lookahead=args.lookahead,
                num_drafts=b,
                max_new_tokens=args.max_new_tokens,
            )
            invariant_results.append(invariant)
            print(
                f"  e2e={invariant['token_rate_global']:.2f} tok/s",
                flush=True,
            )
    finally:
        totals.enabled = False
        model.forward = target_forward
        draft.forward = draft_forward
        restore_sampling_wrappers(original_sampler, original_uniform)

    payload = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "torch": torch.__version__,
        },
        "config": {
            "target": args.target,
            "draft": args.draft,
            "device": args.device,
            "samples": args.samples,
            "warmup": args.warmup,
            "lookahead": args.lookahead,
            "max_new_tokens": args.max_new_tokens,
            "draft_counts": args.draft_counts,
            "top_k": 50,
            "temperature": 1.0,
        },
        "baselines": {
            "target_only": target_only,
            "invariant": invariant_results,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
