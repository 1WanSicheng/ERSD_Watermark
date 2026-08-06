#!/usr/bin/env python3
"""Single-draft PFR/PFR-NOWM runtime breakdown for rebuttal diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TopKLogitsWarper,
)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "MPFR_spec") not in sys.path:
    sys.path.insert(0, str(ROOT / "MPFR_spec"))

import accuwm.pfr as pfr_core  # noqa: E402
from accuwm.mc import mc_sample_generator  # noqa: E402
from profile_mpfr_runtime import (  # noqa: E402
    PROMPTS,
    PhaseTotals,
    benchmark_plain_method,
    benchmark_target_only,
    encode_prompt,
    install_sampling_wrappers,
    restore_sampling_wrappers,
    sync,
    wrap_forward,
)


_TOP_K_WARPER = LogitsProcessorList([TopKLogitsWarper(50)])


def _warp_top_k(input_ids, logits):
    return _TOP_K_WARPER(input_ids, logits)


PROCESS_LOGITS = {
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    # Single-draft pfr.py and mc.py consume the callable warper; the scalar
    # fields above are consumed only by the multi-draft implementation.
    "logits_warper": _warp_top_k,
}


def drain_single_pfr(
    *,
    model,
    draft,
    input_ids,
    lookahead: int,
    max_new_tokens: int,
    watermark: bool,
) -> int:
    labeler = pfr_core.build_default_labeler(mode="context_code")
    generator = pfr_core.pfr_cached_sample_generator(
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_new_tokens,
        private_key=b"p0-profile-key",
        watermark=watermark,
        labeler=labeler,
        process_logits_kwargs=PROCESS_LOGITS,
        return_meta=True,
    )
    generated = 0
    for ids, _logprobs, _meta in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        if generated >= max_new_tokens:
            break
    return generated


def drain_vsps(
    *,
    model,
    draft,
    input_ids,
    lookahead: int,
    max_new_tokens: int,
) -> int:
    generator = mc_sample_generator(
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        process_logits_kwargs=PROCESS_LOGITS,
    )
    generated = 0
    for ids, _logprobs in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        if generated >= max_new_tokens:
            break
    return generated


def benchmark_pfr_method(
    *,
    name: str,
    watermark: bool,
    model,
    draft,
    tokenizer,
    prompts,
    warmup: int,
    lookahead: int,
    max_new_tokens: int,
    totals: PhaseTotals,
) -> dict:
    def run_one(prompt: str, seed: int) -> tuple[int, float, int]:
        input_ids = encode_prompt(tokenizer, prompt, model.device)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        tokens = drain_single_pfr(
            model=model,
            draft=draft,
            input_ids=input_ids,
            lookahead=lookahead,
            max_new_tokens=max_new_tokens,
            watermark=watermark,
        )
        sync()
        elapsed = time.perf_counter() - start
        return tokens, elapsed, int(torch.cuda.max_memory_allocated())

    totals.enabled = False
    for i in range(warmup):
        run_one(prompts[i % len(prompts)], 20_000 + i)

    rows = []
    for i, prompt in enumerate(prompts):
        tokens, elapsed, peak = run_one(prompt, 21_000 + i)
        rows.append({
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "token_rate": tokens / elapsed,
            "peak_allocated_bytes": peak,
        })

    totals.reset()
    totals.enabled = True
    profile_tokens = 0
    profile_elapsed = 0.0
    profile_peak = 0
    for i, prompt in enumerate(prompts):
        tokens, elapsed, peak = run_one(prompt, 22_000 + i)
        profile_tokens += tokens
        profile_elapsed += elapsed
        profile_peak = max(profile_peak, peak)
    totals.enabled = False

    target = totals.seconds.get("target_forward", 0.0)
    draft_time = totals.seconds.get("draft_forward", 0.0)
    sampling = totals.seconds.get("pfr_arrival_sampling", 0.0)
    keyed_rng = totals.seconds.get("keyed_uniform_rng", 0.0)
    fresh_rng = totals.seconds.get("fresh_uniform_rng", 0.0)
    remainder = max(profile_elapsed - target - draft_time - sampling, 0.0)
    total_tokens = sum(row["tokens"] for row in rows)
    total_time = sum(row["elapsed_sec"] for row in rows)

    return {
        "method": name,
        "samples": len(prompts),
        "lookahead": lookahead,
        "max_new_tokens": max_new_tokens,
        "end_to_end": {
            "tokens": total_tokens,
            "elapsed_sec": total_time,
            "token_rate_global": total_tokens / total_time,
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows)
            / (1024**3),
            "rows": rows,
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
                "keyed_uniform_rng_subset": keyed_rng,
                "fresh_uniform_rng_subset": fresh_rng,
                "remainder_cache_python": remainder,
            },
            "derived_fraction": {
                "target_forward": target / profile_elapsed,
                "draft_forward": draft_time / profile_elapsed,
                "pfr_arrival_sampling_including_rng": sampling / profile_elapsed,
                "keyed_uniform_rng_subset": keyed_rng / profile_elapsed,
                "fresh_uniform_rng_subset": fresh_rng / profile_elapsed,
                "remainder_cache_python": remainder / profile_elapsed,
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
    parser.add_argument(
        "--output", type=Path, default=Path("p0_single_first_pass.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    original_target_forward = wrap_forward(model, "target_forward", totals)
    original_draft_forward = wrap_forward(draft, "draft_forward", totals)
    original_sampler, original_uniform = install_sampling_wrappers(totals)
    original_fresh_uniform = pfr_core.FreshNoiseSource.uniform_noise

    def wrapped_fresh_uniform(self, *inner_args, **inner_kwargs):
        with totals.phase("fresh_uniform_rng"):
            return original_fresh_uniform(self, *inner_args, **inner_kwargs)

    pfr_core.FreshNoiseSource.uniform_noise = wrapped_fresh_uniform
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.samples)]

    try:
        totals.enabled = False
        print("[P0-single] target-only", flush=True)
        target_only = benchmark_target_only(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            warmup=args.warmup,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"  e2e={target_only['token_rate_global']:.2f} tok/s", flush=True)

        def run_vsps(input_ids):
            return drain_vsps(
                model=model,
                draft=draft,
                input_ids=input_ids,
                lookahead=args.lookahead,
                max_new_tokens=args.max_new_tokens,
            )

        print("[P0-single] VSPS", flush=True)
        vsps = benchmark_plain_method(
            name="vsps",
            run=run_vsps,
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            warmup=args.warmup,
        )
        print(f"  e2e={vsps['token_rate_global']:.2f} tok/s", flush=True)

        methods = []
        for name, watermark in (("pfr_nowm", False), ("pfr", True)):
            print(f"[P0-single] {name}", flush=True)
            result = benchmark_pfr_method(
                name=name,
                watermark=watermark,
                model=model,
                draft=draft,
                tokenizer=tokenizer,
                prompts=prompts,
                warmup=args.warmup,
                lookahead=args.lookahead,
                max_new_tokens=args.max_new_tokens,
                totals=totals,
            )
            methods.append(result)
            fractions = result["instrumented"]["derived_fraction"]
            print(
                f"  e2e={result['end_to_end']['token_rate_global']:.2f} tok/s "
                f"target={100*fractions['target_forward']:.1f}% "
                f"draft={100*fractions['draft_forward']:.1f}% "
                f"sampling={100*fractions['pfr_arrival_sampling_including_rng']:.1f}% "
                f"remainder={100*fractions['remainder_cache_python']:.1f}%",
                flush=True,
            )
    finally:
        totals.enabled = False
        model.forward = original_target_forward
        draft.forward = original_draft_forward
        pfr_core.FreshNoiseSource.uniform_noise = original_fresh_uniform
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
            "top_k": 50,
            "temperature": 1.0,
        },
        "baselines": {
            "target_only": target_only,
            "vsps": vsps,
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
