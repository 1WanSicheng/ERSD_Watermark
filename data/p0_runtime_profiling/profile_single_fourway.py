#!/usr/bin/env python3
"""Four-way single-draft runtime breakdown for rebuttal diagnostics.

Compares VSPS, MSE, PFR-NOWM, and PFR under one model pair and one effective
top-k configuration. End-to-end passes run without phase instrumentation.
Instrumented passes synchronize narrow phase boundaries and are used only for
bottleneck attribution.
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

import accuwm.basic as basic_core  # noqa: E402
import accuwm.basic_watermark as basic_watermark  # noqa: E402
import accuwm.mc as mc_core  # noqa: E402
import accuwm.mc_watermark as mc_watermark  # noqa: E402
import accuwm.pfr as pfr_core  # noqa: E402
import unbiased_watermark as uwm  # noqa: E402
from profile_mpfr_runtime import (  # noqa: E402
    PROMPTS,
    PhaseTotals,
    encode_prompt,
    install_sampling_wrappers,
    restore_sampling_wrappers,
    sync,
    wrap_forward,
)
from experiments._shared import (  # noqa: E402
    _maybe_inject_chat_template,
    build_process_logits_kwargs,
    load_prompts,
)


EXCLUSIVE_PHASES = (
    "target_forward",
    "draft_forward",
    "logits_processing",
    "pfr_arrival_sampling",
    "mse_watermark_step",
    "mc_accept_residual_sampling",
    "basic_sampling",
)
SUBSET_PHASES = (
    "keyed_uniform_rng",
    "fresh_uniform_rng",
    "mse_code_cpu_numpy",
    "mse_code_to_device",
    "mse_reweight_gpu",
)


def drain_pfr(
    model, draft, input_ids, lookahead, max_new_tokens, watermark,
    process_logits_kwargs,
):
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
        process_logits_kwargs=process_logits_kwargs,
        return_meta=True,
    )
    generated = 0
    blocks = 0
    accepted = 0
    for ids, _logprobs, meta in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        blocks += 1
        accepted += min(int(meta.get("accepted_count", 0)), take)
        if generated >= max_new_tokens:
            break
    return {"tokens": generated, "blocks": blocks, "accepted": accepted}


def drain_vsps(
    model, draft, input_ids, lookahead, max_new_tokens,
    process_logits_kwargs,
):
    generator = mc_core.mc_sample_generator(
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        process_logits_kwargs=process_logits_kwargs,
    )
    generated = 0
    blocks = 0
    for ids, _logprobs in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        blocks += 1
        if generated >= max_new_tokens:
            break
    return {"tokens": generated, "blocks": blocks}


def drain_basic_uwm(
    model, input_ids, max_new_tokens, process_logits_kwargs,
):
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.ContextCodeHistory(batch_shape=(1,))
    generator = basic_watermark.basic_uwm_generator(
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=b"p0-profile-key",
        model=model,
        input_ids=input_ids,
        n=1,
        process_logits_kwargs=process_logits_kwargs,
    )
    generated = 0
    blocks = 0
    for ids, _logprobs in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        blocks += 1
        if generated >= max_new_tokens:
            break
    return {"tokens": generated, "blocks": blocks}


def drain_basic(
    model, input_ids, max_new_tokens, process_logits_kwargs,
):
    """Plain autoregressive sampling under the same logits processing."""
    generator = basic_core.basic_generator(
        model=model,
        input_ids=input_ids,
        n=1,
        process_logits_kwargs=process_logits_kwargs,
    )
    generated = 0
    blocks = 0
    for ids, _logprobs in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        blocks += 1
        if generated >= max_new_tokens:
            break
    return {"tokens": generated, "blocks": blocks}


def drain_gumbel_sd(
    model, draft, input_ids, lookahead, max_new_tokens, strength,
    process_logits_kwargs,
):
    # strength=False is MSE; strength=True is MWS in the paper.
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.ContextCodeHistory(batch_shape=(1,))
    generator = mc_watermark.mc_uwm_sample_generator(
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=b"p0-profile-key",
        reweight_in_mc=strength,
        model=model,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        temperature=1.0,
        process_logits_kwargs=process_logits_kwargs,
    )
    generated = 0
    blocks = 0
    for ids, _logprobs in generator:
        take = min(int(ids.shape[-1]), max_new_tokens - generated)
        generated += take
        blocks += 1
        if generated >= max_new_tokens:
            break
    return {"tokens": generated, "blocks": blocks}


def summarize(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def encode_any_prompt(tokenizer, prompt, device):
    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            ids = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            user_messages = [
                message["content"] for message in prompt
                if message.get("role") == "user"
            ]
            text = user_messages[-1] if user_messages else "\n".join(
                message.get("content", "") for message in prompt
            )
            ids = tokenizer(text, return_tensors="pt").input_ids
        return ids.to(device)
    return encode_prompt(tokenizer, prompt, device)


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


def benchmark_method(
    *,
    name,
    run,
    model,
    tokenizer,
    prompts,
    profile_prompts,
    warmup,
    totals,
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
            **peak_memory_rows(allocated_before),
        }

    totals.enabled = False
    for i in range(warmup):
        run_one(prompts[i % len(prompts)], 30_000 + i)

    rows = [
        run_one(prompt, 31_000 + i)
        for i, prompt in enumerate(prompts)
    ]

    totals.reset()
    totals.enabled = True
    instrumented_rows = [
        run_one(prompt, 31_000 + i)
        for i, prompt in enumerate(profile_prompts)
    ]
    totals.enabled = False

    e2e_tokens = sum(row["tokens"] for row in rows)
    e2e_time = sum(row["elapsed_sec"] for row in rows)
    prof_tokens = sum(row["tokens"] for row in instrumented_rows)
    prof_time = sum(row["elapsed_sec"] for row in instrumented_rows)
    phase_sec = {
        phase: float(totals.seconds.get(phase, 0.0))
        for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
    }
    exclusive_sum = sum(phase_sec[phase] for phase in EXCLUSIVE_PHASES)
    remainder = max(prof_time - exclusive_sum, 0.0)

    return {
        "method": name,
        "end_to_end": {
            "tokens": e2e_tokens,
            "blocks": sum(row["blocks"] for row in rows),
            "elapsed_sec": e2e_time,
            "token_rate_global": e2e_tokens / e2e_time,
            "token_rate_per_prompt": summarize(
                [row["token_rate"] for row in rows]
            ),
            "tokens_per_block": e2e_tokens
            / max(sum(row["blocks"] for row in rows), 1),
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
            "blocks": sum(row["blocks"] for row in instrumented_rows),
            "elapsed_sec": prof_time,
            "token_rate": prof_tokens / prof_time,
            "phase_sec": phase_sec,
            "phase_calls": {
                phase: int(totals.calls.get(phase, 0))
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "phase_ms_per_token": {
                phase: 1000.0 * phase_sec[phase] / prof_tokens
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "phase_fraction": {
                phase: phase_sec[phase] / prof_time
                for phase in EXCLUSIVE_PHASES + SUBSET_PHASES
            },
            "remainder_sec": remainder,
            "remainder_ms_per_token": 1000.0 * remainder / prof_tokens,
            "remainder_fraction": remainder / prof_time,
        },
    }


def patch_phase(module, attribute, phase, totals, patches):
    original = getattr(module, attribute)

    def wrapped(*args, **kwargs):
        with totals.phase(phase):
            return original(*args, **kwargs)

    setattr(module, attribute, wrapped)
    patches.append((module, attribute, original))


def patch_classmethod(cls, attribute, phase, totals, patches):
    original_descriptor = cls.__dict__[attribute]
    original_bound = getattr(cls, attribute)

    def wrapped(inner_cls, *args, **kwargs):
        del inner_cls
        with totals.phase(phase):
            return original_bound(*args, **kwargs)

    setattr(cls, attribute, classmethod(wrapped))
    patches.append((cls, attribute, original_descriptor))


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
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--profile-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lookaheads", type=int, nargs="+", default=[4])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dataset",
        choices=[
            "fixed", "cnn_dailymail", "cnn_paper_summarization",
            "cnn_dailymail_basefmt", "eli5",
        ],
        default="fixed",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "basic", "basic_uwm", "vsps", "mse", "mws",
            "pfr_nowm", "pfr",
        ],
        default=["vsps", "mse", "mws", "pfr_nowm", "pfr"],
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Use 0 to disable top-k truncation.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("p0_single_fourway.json")
    )
    return parser.parse_args()


def main():
    args = parse_args()
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

    totals = PhaseTotals()
    original_target_forward = wrap_forward(model, "target_forward", totals)
    original_draft_forward = wrap_forward(draft, "draft_forward", totals)
    original_sampler, original_uniform = install_sampling_wrappers(totals)
    original_fresh_uniform = pfr_core.FreshNoiseSource.uniform_noise
    patches = []

    def wrapped_fresh_uniform(self, *inner_args, **inner_kwargs):
        with totals.phase("fresh_uniform_rng"):
            return original_fresh_uniform(self, *inner_args, **inner_kwargs)

    pfr_core.FreshNoiseSource.uniform_noise = wrapped_fresh_uniform

    for module in (pfr_core, basic_core, basic_watermark, mc_core, mc_watermark):
        patch_phase(module, "process_logits", "logits_processing", totals, patches)
    for module in (basic_core, basic_watermark, mc_core, mc_watermark):
        patch_phase(module, "basic_sample", "basic_sampling", totals, patches)
    for module in (mc_core, mc_watermark):
        patch_phase(
            module,
            "mc_sample",
            "mc_accept_residual_sampling",
            totals,
            patches,
        )
    for module in (basic_watermark, mc_watermark):
        patch_phase(
            module,
            "step_watermark",
            "mse_watermark_step",
            totals,
            patches,
        )
    patch_classmethod(
        uwm.DeltaGumbel_WatermarkCode,
        "from_random_",
        "mse_code_cpu_numpy",
        totals,
        patches,
    )
    patch_phase(
        uwm.DeltaGumbel_WatermarkCode,
        "tensor_shape_map",
        "mse_code_to_device",
        totals,
        patches,
    )
    patch_phase(
        uwm.DeltaGumbel_Reweight,
        "reweight_logits",
        "mse_reweight_gpu",
        totals,
        patches,
    )

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
    process_logits_kwargs = build_process_logits_kwargs({
        "temperature": 1.0,
        "top_k": args.top_k,
        "top_p": 1.0,
    })

    results = []
    try:
        for lookahead in args.lookaheads:
            runners = {
                "basic": lambda ids, L=lookahead: drain_basic(
                    model, ids, args.max_new_tokens,
                    process_logits_kwargs,
                ),
                "basic_uwm": lambda ids, L=lookahead: drain_basic_uwm(
                    model, ids, args.max_new_tokens,
                    process_logits_kwargs,
                ),
                "vsps": lambda ids, L=lookahead: drain_vsps(
                    model, draft, ids, L, args.max_new_tokens,
                    process_logits_kwargs,
                ),
                "mse": lambda ids, L=lookahead: drain_gumbel_sd(
                    model, draft, ids, L, args.max_new_tokens, False,
                    process_logits_kwargs,
                ),
                "mws": lambda ids, L=lookahead: drain_gumbel_sd(
                    model, draft, ids, L, args.max_new_tokens, True,
                    process_logits_kwargs,
                ),
                "pfr_nowm": lambda ids, L=lookahead: drain_pfr(
                    model, draft, ids, L, args.max_new_tokens, False,
                    process_logits_kwargs,
                ),
                "pfr": lambda ids, L=lookahead: drain_pfr(
                    model, draft, ids, L, args.max_new_tokens, True,
                    process_logits_kwargs,
                ),
            }
            for name in args.methods:
                run = runners[name]
                print(
                    f"[P0-single-fourway] L={lookahead} {name}",
                    flush=True,
                )
                result = benchmark_method(
                    name=name,
                    run=run,
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    profile_prompts=profile_prompts,
                    warmup=args.warmup,
                    totals=totals,
                )
                result["lookahead"] = lookahead
                results.append(result)
                inst = result["instrumented"]
                print(
                    f"  e2e={result['end_to_end']['token_rate_global']:.2f} tok/s "
                    f"tok/block={result['end_to_end']['tokens_per_block']:.2f} "
                    f"target={100*inst['phase_fraction']['target_forward']:.1f}% "
                    f"draft={100*inst['phase_fraction']['draft_forward']:.1f}% "
                    f"pfr={100*inst['phase_fraction']['pfr_arrival_sampling']:.1f}% "
                    f"mse-wm={100*inst['phase_fraction']['mse_watermark_step']:.1f}% "
                    f"remainder={100*inst['remainder_fraction']:.1f}%",
                    flush=True,
                )
    finally:
        totals.enabled = False
        model.forward = original_target_forward
        draft.forward = original_draft_forward
        pfr_core.FreshNoiseSource.uniform_noise = original_fresh_uniform
        restore_sampling_wrappers(original_sampler, original_uniform)
        for module, attribute, original in reversed(patches):
            setattr(module, attribute, original)

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
            "samples": args.samples,
            "profile_samples": profile_samples,
            "warmup": args.warmup,
            "lookaheads": args.lookaheads,
            "max_new_tokens": args.max_new_tokens,
            "dataset": args.dataset,
            "top_k": args.top_k,
            "temperature": 1.0,
            "seed_protocol": (
                "one paired seed per prompt; identical across methods and "
                "end-to-end/instrumented passes"
            ),
            "mse_mapping": "mc_uwm_speed (reweight_in_mc=False)",
            "mws_mapping": "mc_uwm_strength (reweight_in_mc=True)",
            "methods": args.methods,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
