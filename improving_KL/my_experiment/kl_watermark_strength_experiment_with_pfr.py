#!/usr/bin/env python3
"""
KL watermark-strength + AATPS experiment with an added single-draft PFR method.

Place this file at:

    improving_KL/my_experiment/kl_watermark_strength_experiment_with_pfr.py

and place/copy your uploaded MPFR implementation at either:

    improving_KL/mpfr_direct_optimized.py

or anywhere importable via PYTHONPATH.

Then run from the improving_KL project root, for example:

    PYTHONPATH=. python -u my_experiment/kl_watermark_strength_experiment_with_pfr.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --ref-model Qwen/Qwen2.5-0.5B-Instruct \
      --dataset manual \
      --samples 2 \
      --max-length 32 \
      --lookahead 4 \
      --top-k 100 \
      --methods basic_uwm mc_uwm_strength mc_uwm_speed pfr_uwm \
      --seed 1 \
      --kl-num-keys 1 \
      --output outputs/kl_aatps_with_pfr.json

The new method name is:

    pfr_uwm

It implements your simplified single-draft PFR speculative decoder by calling
finite_multi_draft_pfr_sample_generator(..., B=1).  The target and draft share
one keyed finite-support Poisson source at each context.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Make the project root importable when this file is run from my_experiment/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the original experiment harness instead of duplicating it.
import my_experiment.kl_watermark_strength_experiment as base
from unbiased_watermark.scores.kl_watermark_strength import (
    next_token_logits_from_full_sequence,
    summarize_tokenwise_kl,
)

try:
    import mpfr_direct_optimized as mpfr
except ImportError as exc:  # pragma: no cover, user-facing setup error
    raise ImportError(
        "Could not import mpfr_direct_optimized.py. Copy your uploaded "
        "mpfr_direct_optimized.py into the improving_KL project root, or add "
        "its directory to PYTHONPATH."
    ) from exc


_BASE_COLLECT_GENERATION = base.collect_generation
_BASE_ESTIMATE_KL = base.estimate_kl
_BASE_AGGREGATE = base.aggregate
_BASE_BUILD_PARSER = base.build_parser


def seed_bytes(seed: int, suffix: str | bytes = b"") -> bytes:
    """Safe fixed-width seed encoding.

    Do not use bytes(seed): bytes(2155929800) allocates about 2.1GB of zeros.
    """
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be nonnegative")
    prefix = (seed_int & ((1 << 64) - 1)).to_bytes(8, "little", signed=False)
    if isinstance(suffix, str):
        suffix = suffix.encode("utf-8")
    return prefix + bytes(suffix)


def pfr_top_k(args: argparse.Namespace) -> int:
    value = getattr(args, "pfr_top_k", None)
    return int(args.top_k if value is None else value)


def pfr_process_logits_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Processing used by the uploaded direct finite-support MPFR implementation.

    mpfr_direct_optimized.py intentionally supports temperature/top-k/top-p.
    It does not use the original accuwm logits_processor interface, so we pass
    these fields explicitly for PFR.
    """
    return {
        "temperature": (float(args.temperature), float(args.temperature)),
        "top_k": pfr_top_k(args),
        "top_p": float(getattr(args, "top_p", 1.0)),
    }


def maybe_cuda_synchronize(model) -> None:
    if torch.cuda.is_available():
        try:
            device = getattr(model, "device", None)
            if device is not None and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device=torch.device(device))
            else:
                torch.cuda.synchronize()
        except Exception:
            torch.cuda.synchronize()


@torch.no_grad()
def collect_generation(
    method: str,
    target,
    draft,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    process_logits_kwargs: dict[str, Any],
) -> tuple[torch.LongTensor, torch.LongTensor, dict[str, Any]]:
    """Collect one generated continuation and metadata.

    For all existing methods, delegate to the original file.  For pfr_uwm, use
    the uploaded exact direct MPFR code with B=1, i.e. single-draft PFR.
    """
    if method != "pfr_uwm":
        maybe_cuda_synchronize(target)
        t0 = time.perf_counter()
        input_ids, out_ids, meta = _BASE_COLLECT_GENERATION(
            method,
            target,
            draft,
            tokenizer,
            prompt,
            args,
            process_logits_kwargs,
        )
        maybe_cuda_synchronize(target)
        elapsed = time.perf_counter() - t0
        meta = dict(meta)
        meta["generation_elapsed_sec"] = elapsed
        meta["token_rate"] = int(out_ids.numel()) / max(elapsed, 1e-12)
        return input_ids, out_ids, meta

    input_ids = base.encode_prompt(tokenizer, args.model, prompt, target.device)
    private_key = seed_bytes(args.seed, args.private_key)

    gen = mpfr.finite_multi_draft_pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=int(args.lookahead),
        max_length=int(args.max_length),
        private_key=private_key,
        num_drafts=1,
        B=1,
        labeler=None,
        process_logits_kwargs=pfr_process_logits_kwargs(args),
        return_meta=True,
        proposal=str(getattr(args, "pfr_proposal", "batched_target_direct_topk")),
    )

    chunks: list[torch.LongTensor] = []
    chunk_lengths: list[int] = []
    block_meta: list[dict[str, Any]] = []
    generated = 0

    maybe_cuda_synchronize(target)
    t0 = time.perf_counter()
    for output_ids, _output_logprobs, meta in gen:
        remaining = int(args.max_length) - generated
        if remaining <= 0:
            break
        if output_ids.shape[1] > remaining:
            output_ids = output_ids[:, :remaining]
        chunks.append(output_ids.detach())
        chunk_lengths.append(int(output_ids.shape[1]))
        block_meta.append(dict(meta))
        generated += int(output_ids.shape[1])
        if generated >= int(args.max_length):
            break
    maybe_cuda_synchronize(target)
    elapsed = time.perf_counter() - t0

    if chunks:
        out_ids = torch.cat(chunks, dim=1)
    else:
        out_ids = torch.empty((1, 0), dtype=torch.long, device=input_ids.device)

    meta_out: dict[str, Any] = {
        "chunk_lengths": chunk_lengths,
        "generation_elapsed_sec": elapsed,
        "token_rate": int(out_ids.numel()) / max(elapsed, 1e-12),
        "pfr_B": 1,
        "pfr_num_drafts": 1,
        "pfr_top_k": pfr_top_k(args),
        "pfr_proposal": str(getattr(args, "pfr_proposal", "batched_target_direct_topk")),
        "pfr_block_meta": block_meta,
    }

    # Aggregate MPFR block counters so they are easy to inspect in the JSON rows.
    for key in (
        "accepted_count",
        "attempted_draft_tokens",
        "draft_tree_size",
        "target_context_count",
        "target_forward_calls",
        "draft_forward_calls",
    ):
        meta_out[key + "_sum"] = int(sum(int(m.get(key, 0)) for m in block_meta))

    return input_ids, out_ids, meta_out


@torch.no_grad()
def compute_pfr_delta_kl_from_sequence(
    p_logits: torch.FloatTensor,
    full_ids: torch.LongTensor,
    prompt_length: int,
    private_key: bytes,
    *,
    temperature: float,
    top_k: int,
    baseline: str = "full",
) -> dict[str, Any]:
    """Compute Definition-3.1-style KL for keyed target-side PFR.

    At a fixed prefix c and key zeta, direct finite PFR chooses one deterministic
    token y_zeta(c).  Hence P_zeta(.|c) is a point mass and

        D_KL(P_zeta(.|c) || P(.|c)) = -log P(y_zeta(c)|c).

    The PFR winner is drawn from the processed top-k/temperature distribution,
    exactly as in mpfr_direct_optimized.py.

    baseline="full": compare the point mass to the full target distribution
        softmax(raw_logits / temperature).  This follows the uploaded markdown's
        convention that the denominator is full target entropy whenever possible.

    baseline="processed": compare the point mass to the processed top-k target
        distribution actually used by the finite-support PFR sampler.  This is
        the clean decoder-level baseline and its key-averaged ratio is near 1.
    """
    if p_logits.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    if baseline not in {"full", "processed"}:
        raise ValueError("baseline must be 'full' or 'processed'")
    if prompt_length <= 0 or prompt_length > full_ids.shape[1]:
        raise ValueError("invalid prompt_length")

    source_factory = mpfr.KeyedPoissonSource(private_key)
    per_token_kl = []
    per_token_entropy = []

    # p_logits has shape [1, number_of_output_tokens, vocab_size].
    for offset in range(p_logits.shape[1]):
        pos = prompt_length + offset
        context = tuple(int(x) for x in full_ids[0, :pos].detach().cpu().tolist())
        raw_logits = p_logits[0, offset, :].float()

        processed_logits = mpfr.process_logits_exact(
            raw_logits,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=1.0,
        )
        processed_logprobs = F.log_softmax(processed_logits.float(), dim=-1)

        source = source_factory.for_context(context)
        y = int(mpfr.direct_finite_mpfr_tokens_from_logprobs(processed_logprobs, source, 1)[0].item())

        if baseline == "processed":
            base_logprobs = processed_logprobs
            kind = "pfr_delta_processed_topk"
        else:
            base_logprobs = F.log_softmax(raw_logits / float(temperature), dim=-1)
            kind = "pfr_delta_full_p"

        kl_value = -base_logprobs[y]
        base_probs = base_logprobs.exp()
        entropy_value = -torch.where(
            base_probs > 0,
            base_probs * base_logprobs,
            torch.zeros_like(base_probs),
        ).sum()
        per_token_kl.append(kl_value)
        per_token_entropy.append(entropy_value)

    if not per_token_kl:
        return summarize_tokenwise_kl(torch.empty((1, 0), device=p_logits.device))

    result = summarize_tokenwise_kl(
        torch.stack(per_token_kl).view(1, -1),
        skipped=None,
        per_token_entropy=torch.stack(per_token_entropy).view(1, -1),
    )
    result["KL_WS_kind"] = kind
    result["KL_WS_pfr_top_k"] = int(top_k)
    result["KL_WS_pfr_baseline"] = baseline
    return result


@torch.no_grad()
def estimate_kl(
    method: str,
    target,
    draft,
    input_ids: torch.LongTensor,
    out_ids: torch.LongTensor,
    args: argparse.Namespace,
    process_logits_kwargs: dict[str, Any],
    generation_meta: dict[str, Any] | None = None,
    kl_key_index: int = 0,
) -> dict[str, Any]:
    """Estimate KL watermark strength for all methods, including pfr_uwm."""
    if method != "pfr_uwm":
        return _BASE_ESTIMATE_KL(
            method,
            target,
            draft,
            input_ids,
            out_ids,
            args,
            process_logits_kwargs,
            generation_meta=generation_meta,
            kl_key_index=kl_key_index,
        )

    if out_ids.numel() == 0:
        return {"KL_WS_mean": 0.0, "KL_WS_sum": 0.0, "KL_WS_count": 0, "KL_WS_kind": "empty"}

    full_ids = torch.cat([input_ids, out_ids], dim=1)
    prompt_length = input_ids.shape[1]
    p_logits = next_token_logits_from_full_sequence(
        target,
        full_ids,
        prompt_length,
        process_logits_kwargs=process_logits_kwargs,
    )
    private_key = seed_bytes(int(args.seed) + int(args.kl_key_offset) + int(kl_key_index), args.private_key)
    return compute_pfr_delta_kl_from_sequence(
        p_logits,
        full_ids,
        prompt_length,
        private_key,
        temperature=float(args.temperature),
        top_k=pfr_top_k(args),
        baseline=str(getattr(args, "pfr_kl_baseline", "full")),
    )


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Original aggregate plus generation token-rate fields."""
    result = _BASE_AGGREGATE(rows, method)
    method_rows = [row for row in rows if row["method"] == method]
    elapsed = [float(row.get("generation_elapsed_sec", 0.0)) for row in method_rows]
    total_elapsed = float(np.sum(elapsed)) if elapsed else 0.0
    total_tokens = int(result.get("num_tokens", 0))
    token_rates = [
        float(row.get("num_tokens", 0)) / max(float(row.get("generation_elapsed_sec", 0.0)), 1e-12)
        for row in method_rows
        if float(row.get("generation_elapsed_sec", 0.0)) > 0
    ]
    result["generation_elapsed_sec"] = total_elapsed
    result["token_rate"] = total_tokens / max(total_elapsed, 1e-12)
    result["token_rate_sample_mean"] = float(np.mean(token_rates)) if token_rates else 0.0
    result["token_rate_sample_std"] = float(np.std(token_rates)) if token_rates else 0.0
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = _BASE_BUILD_PARSER()
    parser.add_argument(
        "--pfr-proposal",
        default="batched_target_direct_topk",
        choices=["batched_target_direct_topk", "direct_topk"],
        help="PFR implementation variant from mpfr_direct_optimized.py.",
    )
    parser.add_argument(
        "--pfr-top-k",
        type=int,
        default=None,
        help="Top-k support used by PFR. Defaults to --top-k.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Accepted for API symmetry. The uploaded MPFR code currently ignores top-p.",
    )
    parser.add_argument(
        "--pfr-kl-baseline",
        default="full",
        choices=["full", "processed"],
        help=(
            "full: KL(delta_y || full softmax target), matching the markdown's full-P convention. "
            "processed: KL(delta_y || top-k/temperature target actually sampled by PFR)."
        ),
    )
    return parser


def main() -> None:
    # Patch the original module's globals so base.run_experiment uses the new functions.
    base.collect_generation = collect_generation
    base.estimate_kl = estimate_kl
    base.aggregate = aggregate
    base.build_parser = build_parser

    parser = build_parser()
    cli_args = parser.parse_args()
    runs = base.load_configured_runs(parser, cli_args)
    results = []
    for index, args in enumerate(runs):
        if len(runs) > 1:
            print(json.dumps({"run_index": index, "output": args.output, "methods": args.methods}, ensure_ascii=False))
        results.append(base.run_experiment(args))
    if len(results) > 1:
        print(json.dumps({"num_runs": len(results), "outputs": [run["args"].get("output") for run in results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
