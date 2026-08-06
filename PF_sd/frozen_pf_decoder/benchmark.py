#!/usr/bin/env python3
"""Small matched benchmark for max-order PF, MPFR, and List-Level INVARIANT."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import _shared as S  # noqa: E402
from PF_sd.frozen_pf_decoder.core import (  # noqa: E402
    counter_draft_select_support,
    counter_latin_draft_select_support,
    counter_latin_target_select_support,
    counter_latin_target_traverse_support,
    counter_target_select_support,
    keyed_counter_uniforms_on_support,
    max_order_pf_generator,
    pf_li_score,
    pf_power_law_score,
    recover_max_order_pivots,
    speculative_max_order_pf_generator,
)
from accuwm.pfr import PFRSourceFactory  # noqa: E402
from PF_sd.frozen_pf_decoder.decoder import (  # noqa: E402
    speculative_tree_free_latin_pf_generator,
)
from unbiased_watermark.scores.pfr_aaronson import (  # noqa: E402
    _log_gamma_tail_p_value,
)


def _drain(generator: Iterable, max_length: int, *, collect_pivots: bool = True):
    chunks: List[torch.Tensor] = []
    block_lens: List[int] = []
    labels: List[bytes] = []
    pivots: List[torch.Tensor] = []
    metas: List[dict] = []
    generated = 0
    for ids, _logprobs, meta in generator:
        take = min(int(ids.shape[-1]), int(max_length) - generated)
        if take <= 0:
            break
        chunks.append(ids[:, :take].detach().cpu())
        block_lens.append(take)
        labels.extend(meta.get("source_labels", [])[:take])
        pivot = meta.get("aggregate_pivots")
        if pivot is not None and collect_pivots:
            pivots.append(torch.as_tensor(pivot).reshape(-1)[:take].cpu())
        metas.append(meta)
        generated += take
        if generated >= int(max_length):
            break
    output = (
        torch.cat(chunks, dim=1)
        if chunks
        else torch.empty((1, 0), dtype=torch.long)
    )
    all_pivots = torch.cat(pivots) if pivots else torch.empty(0)
    return output, block_lens, labels, all_pivots, metas


def _pf_metrics(pivots: torch.Tensor, rho: float, eps: float) -> dict:
    if pivots.numel() == 0:
        return {
            "PF_ANLPPT_original": 0.0,
            "PF_Li_score": 0.0,
            "PF_PL_score": 0.0,
        }
    pivots = pivots.float().clamp(1e-10, 1.0)
    n = int(pivots.numel())
    original = float((-torch.log(pivots)).sum())
    log_p = _log_gamma_tail_p_value(n, original)
    return {
        "PF_ANLPPT_original": max(-float(log_p) / n, 0.0),
        "PF_Li_score": float(pf_li_score(pivots, rho).mean()),
        "PF_PL_score": float(pf_power_law_score(pivots, eps).mean()),
        "PF_mean_pivot": float(pivots.mean()),
    }


def _aggregate(rows: List[dict]) -> dict:
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        for width in sorted({row["width"] for row in rows if row["method"] == method}):
            group = [
                row for row in rows
                if row["method"] == method and row["width"] == width
            ]
            key = f"{method}_B{width}"
            total_tokens = sum(row["tokens"] for row in group)
            total_blocks = sum(row["blocks"] for row in group)
            total_time = sum(row["elapsed_sec"] for row in group)
            summary[key] = {
                "n_prompts": len(group),
                "tokens": total_tokens,
                "blocks": total_blocks,
                "AATPS": total_tokens / max(total_blocks, 1),
                "token_rate": total_tokens / max(total_time, 1e-12),
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in group),
            }
            for field in (
                "PF_ANLPPT_original",
                "PF_Li_score",
                "PF_PL_score",
                "target_contexts_per_block",
                "draft_contexts_per_block",
            ):
                values = [float(row[field]) for row in group if field in row]
                if values:
                    summary[key][field] = float(np.mean(values))
    return summary


def _load_prompts(config: dict) -> List[List[dict]]:
    """Load prompts through the shared loader or directly from local Arrow."""
    n = int(config["samples"])
    arrow_path = config.get("dataset_arrow")
    if not arrow_path:
        return S.load_prompts(config.get("dataset", "cnn_dailymail"), n)
    ds = Dataset.from_file(str(arrow_path)).shuffle(seed=42)
    ds = ds.filter(lambda row: len(row["article"]) < 3000).select(range(n))
    return [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": (
                    "Summarize the following article in 3-5 sentences:\n\n"
                    f"{row['article'][:1500]}"
                ),
            },
        ]
        for row in ds
    ]


def run(config: dict) -> dict:
    target, draft, tokenizer, device = S.load_models_and_tokenizer(config)
    prompts = _load_prompts(config)
    methods = list(config.get("methods", ["target_pf", "max_order_pf", "mpfr", "invariant"]))
    widths = [int(width) for width in config.get("widths", [2, 4])]
    lookahead = int(config.get("lookahead", 4))
    max_length = int(config.get("max_new_tokens", 128))
    temperature = float(config.get("temperature", 1.0))
    top_k = int(config.get("top_k", 50) or 0)
    top_p = float(config.get("top_p", 1.0))
    use_chat = bool(config.get("use_chat_template", True))
    base_key = S.private_key_from_str(config.get("private_key", "max-order-pf"))
    rho = float(config.get("pf_detector_rho", 0.5))
    eps = float(config.get("pf_detector_eps", 0.05))
    full_generation_warmup = bool(config.get("full_generation_warmup", True))
    rotate_method_order = bool(config.get("rotate_method_order", True))
    rotate_width_order = bool(config.get("rotate_width_order", True))
    vocab_size = min(int(target.config.vocab_size), int(draft.config.vocab_size))
    plk = S.build_process_logits_kwargs(
        {"temperature": temperature, "top_k": top_k, "top_p": top_p}
    )
    rows: List[dict] = []

    if any("counter" in method for method in methods):
        if top_k <= 0:
            raise ValueError("counter benchmark currently requires top_k > 0")
        keyed_counter_uniforms_on_support(
            source_factory=PFRSourceFactory(base_key),
            context_label=b"counter-kernel-warmup",
            fields=range(max(widths)),
            support=torch.arange(min(top_k, vocab_size), device=device),
            vocab_size=vocab_size,
            device=device,
        )
        warm_dtype = next(target.parameters()).dtype
        warm_logits = torch.zeros(vocab_size, device=device, dtype=warm_dtype)
        warm_support = torch.arange(min(top_k, vocab_size), device=device)
        warm_factory = PFRSourceFactory(base_key)
        for n_fields in range(1, max(widths) + 1):
            counter_draft_select_support(
                processed_logits=warm_logits,
                support=warm_support,
                fields=range(n_fields),
                source_factory=warm_factory,
                context_label=b"counter-select-warmup",
            )
            counter_target_select_support(
                processed_logits=warm_logits,
                support=warm_support,
                fields=range(n_fields),
                source_factory=warm_factory,
                context_label=b"counter-select-warmup",
            )
            warm_compact = torch.zeros(
                int(warm_support.numel()), device=device, dtype=warm_dtype
            )
            counter_draft_select_support(
                processed_logits=warm_compact,
                support=warm_support,
                fields=range(n_fields),
                source_factory=warm_factory,
                context_label=b"counter-select-warmup",
                compact_logits=True,
                vocab_size=vocab_size,
            )
            counter_target_select_support(
                processed_logits=warm_compact,
                support=warm_support,
                fields=range(n_fields),
                source_factory=warm_factory,
                context_label=b"counter-select-warmup",
                compact_logits=True,
                vocab_size=vocab_size,
            )
        if any(
            method in {"latin_pf_counter_fused", "latin_pf_counter_tree_free"}
            for method in methods
        ):
            for latin_width in widths:
                for active_count in range(1, int(latin_width) + 1):
                    counter_latin_draft_select_support(
                        processed_logits=warm_compact,
                        support=warm_support,
                        fields=range(active_count),
                        width=latin_width,
                        source_factory=warm_factory,
                        context_label=b"counter-latin-active-warmup",
                        compact_logits=True,
                        vocab_size=vocab_size,
                    )
                counter_latin_target_select_support(
                    processed_logits=warm_compact,
                    support=warm_support,
                    width=latin_width,
                    source_factory=warm_factory,
                    context_label=b"counter-latin-warmup",
                    exact_pivot=False,
                    compact_logits=True,
                    vocab_size=vocab_size,
                )
                counter_latin_target_traverse_support(
                    processed_logits=warm_compact.unsqueeze(0),
                    supports=warm_support.unsqueeze(0),
                    context_seeds=[1],
                    child_tokens=torch.full(
                        (1, latin_width), -1, dtype=torch.int64
                    ),
                    child_rows=torch.full(
                        (1, latin_width), -1, dtype=torch.int64
                    ),
                    width=latin_width,
                    vocab_size=vocab_size,
                    draft_steps=lookahead,
                    max_outputs=lookahead + 1,
                    eos_token=-1,
                )
        S._sync()

    # Primitive warm-up above removes counter/Triton compilation from the
    # timing window.  A full excluded generation pass is still needed to warm
    # model forwards, KV-cache allocation, and method-specific decoder paths.
    prompt_runs = []
    if full_generation_warmup and prompts:
        prompt_runs.append((-1, prompts[0], True))
    prompt_runs.extend((idx, prompt, False) for idx, prompt in enumerate(prompts))

    target_reference_methods = {
        "target_pf", "target_pf_counter", "target_anchor_pf",
        "target_latin_pf_counter", "target_reverse_latin_pf_counter",
    }

    for prompt_idx, prompt, is_warmup in prompt_runs:
        input_ids = S.encode_prompt(
            tokenizer, prompt, device, use_chat_template=use_chat
        )
        # Reuse prompt 0's seed for the excluded warm-up.  Every measured run
        # resets the RNG again, so warm-up cannot change generated samples.
        seed = int(config.get("seed", 7)) + max(prompt_idx, 0)
        private_key = S.seeded_private_key(seed, base_key)
        expected_by_width = {}
        expected_counter_by_width = {}
        expected_anchor_by_width = {}
        expected_latin_by_width = {}
        expected_reverse_latin_by_width = {}
        references = [m for m in methods if m in target_reference_methods]
        measured_methods = [m for m in methods if m not in target_reference_methods]
        if rotate_method_order and not is_warmup and measured_methods:
            shift = prompt_idx % len(measured_methods)
            measured_methods = measured_methods[shift:] + measured_methods[:shift]
        # Target-only references must remain first because some speculative
        # methods compare their token path against the corresponding reference.
        order = references + measured_methods

        for method in order:
            run_widths = list(widths)
            if rotate_width_order and not is_warmup and run_widths:
                width_shift = prompt_idx % len(run_widths)
                run_widths = run_widths[width_shift:] + run_widths[:width_shift]
            for width in run_widths:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                S._sync()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()

                if method == "target_pf":
                    result = _drain(
                        max_order_pf_generator(
                            target,
                            input_ids.clone(),
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            max_vocab_size=vocab_size,
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                    expected_by_width[width] = out_ids.reshape(-1).tolist()
                elif method == "target_anchor_pf":
                    result = _drain(
                        max_order_pf_generator(
                            target,
                            input_ids.clone(),
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            max_vocab_size=vocab_size,
                            target_coupling="random_anchor",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                    expected_anchor_by_width[width] = out_ids.reshape(-1).tolist()
                elif method == "target_pf_counter":
                    result = _drain(
                        max_order_pf_generator(
                            target,
                            input_ids.clone(),
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            max_vocab_size=vocab_size,
                            rng_backend="counter_philox",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                    expected_counter_by_width[width] = out_ids.reshape(-1).tolist()
                elif method == "target_latin_pf_counter":
                    result = _drain(
                        max_order_pf_generator(
                            target,
                            input_ids.clone(),
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            max_vocab_size=vocab_size,
                            target_coupling="latin_hypercube",
                            rng_backend="counter_philox",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                    expected_latin_by_width[width] = out_ids.reshape(-1).tolist()
                elif method == "target_reverse_latin_pf_counter":
                    result = _drain(
                        max_order_pf_generator(
                            target,
                            input_ids.clone(),
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            max_vocab_size=vocab_size,
                            target_coupling="latin_reverse",
                            rng_backend="counter_philox",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                    expected_reverse_latin_by_width[width] = (
                        out_ids.reshape(-1).tolist()
                    )
                elif method == "max_order_pf":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "random_anchor_pf":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            target_coupling="random_anchor",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "max_order_pf_counter":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            rng_backend="counter_philox",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "latin_pf_counter":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            target_coupling="latin_hypercube",
                            rng_backend="counter_philox",
                            fuse_latin_sampling=False,
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "latin_pf_counter_fused":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            target_coupling="latin_hypercube",
                            rng_backend="counter_philox",
                            batch_target_selection=False,
                            fuse_latin_sampling=True,
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "latin_pf_counter_tree_free":
                    result = _drain(
                        speculative_tree_free_latin_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "reverse_latin_pf_counter":
                    result = _drain(
                        speculative_max_order_pf_generator(
                            target,
                            draft,
                            input_ids.clone(),
                            lookahead=lookahead,
                            width=width,
                            max_length=max_length,
                            private_key=private_key,
                            process_logits_kwargs=plk,
                            return_meta=True,
                            return_logprobs=False,
                            record_pivots=False,
                            target_coupling="latin_reverse",
                            rng_backend="counter_philox",
                        ),
                        max_length,
                        collect_pivots=False,
                    )
                    out_ids, block_lens, labels, pivots, metas = result
                elif method == "mpfr":
                    out_ids, block_lens, *_ = S._run_mpfr_torchgen_cached(
                        target,
                        draft,
                        input_ids.clone(),
                        lookahead=lookahead,
                        max_length=max_length,
                        seed=seed,
                        base_key=base_key,
                        plk=plk,
                        num_drafts=width,
                    )
                    labels, pivots, metas = [], torch.empty(0), []
                elif method == "invariant":
                    out_ids, block_lens, *_ = S._run_invariant_multi(
                        target,
                        draft,
                        input_ids.clone(),
                        lookahead=lookahead,
                        max_length=max_length,
                        seed=seed,
                        base_key=base_key,
                        plk=plk,
                        num_drafts=width,
                    )
                    # The legacy INVARIANT generator emits its entire final
                    # verification block and can overshoot max_length by up to
                    # lookahead tokens.  All other runners return exactly the
                    # requested output budget, so crop before computing AATPS
                    # and TR.  The final invocation still counts as one block.
                    out_ids = out_ids[:, :max_length]
                    labels, pivots, metas = [], torch.empty(0), []
                else:
                    raise ValueError(f"unknown method: {method}")

                S._sync()
                elapsed = time.perf_counter() - start
                peak_gib = (
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                n_tokens = int(out_ids.shape[-1])
                if is_warmup:
                    print(
                        f"[warmup {method} B={width}] "
                        f"tokens={n_tokens} elapsed={elapsed:.3f}s",
                        flush=True,
                    )
                    continue
                row = {
                    "method": method,
                    "width": width,
                    "prompt_idx": prompt_idx,
                    "tokens": n_tokens,
                    "blocks": max(len(block_lens), 1),
                    "AATPS": n_tokens / max(len(block_lens), 1),
                    "token_rate": n_tokens / max(elapsed, 1e-12),
                    "elapsed_sec": elapsed,
                    "peak_allocated_gib": peak_gib,
                }

                if method in {
                    "target_pf", "max_order_pf",
                    "target_anchor_pf", "random_anchor_pf",
                    "target_pf_counter", "max_order_pf_counter",
                    "target_latin_pf_counter", "latin_pf_counter",
                    "latin_pf_counter_fused", "latin_pf_counter_tree_free",
                    "target_reverse_latin_pf_counter", "reverse_latin_pf_counter",
                }:
                    coupling = (
                        "random_anchor"
                        if method in {"target_anchor_pf", "random_anchor_pf"}
                        else (
                            "latin_hypercube"
                            if method in {
                                "target_latin_pf_counter",
                                "latin_pf_counter",
                                "latin_pf_counter_fused",
                                "latin_pf_counter_tree_free",
                            }
                            else (
                                "latin_reverse"
                                if method in {
                                    "target_reverse_latin_pf_counter",
                                    "reverse_latin_pf_counter",
                                }
                                else "max_order"
                            )
                        )
                    )
                    rng_backend = (
                        "counter_philox"
                        if method in {
                            "target_pf_counter", "max_order_pf_counter",
                            "target_latin_pf_counter", "latin_pf_counter",
                            "latin_pf_counter_fused", "latin_pf_counter_tree_free",
                            "target_reverse_latin_pf_counter",
                            "reverse_latin_pf_counter",
                        }
                        else "torch_dense"
                    )
                    recovered = recover_max_order_pivots(
                        out_ids=out_ids,
                        context_labels=labels,
                        width=width,
                        private_key=private_key,
                        vocab_size=vocab_size,
                        device=device,
                        target_coupling=coupling,
                        rng_backend=rng_backend,
                    ).cpu()
                    if pivots.numel() and not torch.equal(recovered, pivots):
                        raise RuntimeError("post-hoc aggregate pivots do not match generation")
                    pivots = recovered
                    row.update(_pf_metrics(recovered, rho=rho, eps=eps))
                if method == "max_order_pf":
                    row["exact_target_path"] = float(
                        out_ids.reshape(-1).tolist() == expected_by_width[width]
                    )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if method == "random_anchor_pf":
                    row["exact_target_path"] = float(
                        out_ids.reshape(-1).tolist()
                        == expected_anchor_by_width[width]
                    )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if method == "max_order_pf_counter":
                    if width in expected_counter_by_width:
                        row["exact_target_path"] = float(
                            out_ids.reshape(-1).tolist()
                            == expected_counter_by_width[width]
                        )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if method == "latin_pf_counter":
                    if width in expected_latin_by_width:
                        row["exact_target_path"] = float(
                            out_ids.reshape(-1).tolist()
                            == expected_latin_by_width[width]
                        )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if method == "latin_pf_counter_fused":
                    if width in expected_latin_by_width:
                        row["exact_target_path"] = float(
                            out_ids.reshape(-1).tolist()
                            == expected_latin_by_width[width]
                        )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if method == "latin_pf_counter_tree_free":
                    if width in expected_latin_by_width:
                        row["exact_target_path"] = float(
                            out_ids.reshape(-1).tolist()
                            == expected_latin_by_width[width]
                        )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                if config.get("store_output_tokens", False):
                    row["output_token_ids"] = out_ids.reshape(-1).tolist()
                if method == "reverse_latin_pf_counter":
                    if width in expected_reverse_latin_by_width:
                        row["exact_target_path"] = float(
                            out_ids.reshape(-1).tolist()
                            == expected_reverse_latin_by_width[width]
                        )
                    row["target_contexts_per_block"] = float(
                        np.mean([meta["target_context_count"] for meta in metas])
                    )
                    row["draft_contexts_per_block"] = float(
                        np.mean([meta["draft_tree_size"] for meta in metas])
                    )
                rows.append(row)
                print(
                    f"[{method} B={width}] prompt={prompt_idx} "
                    f"AATPS={row['AATPS']:.3f} TR={row['token_rate']:.2f}",
                    flush=True,
                )
    return {"config": config, "summary": _aggregate(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--lookahead", type=int)
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.samples is not None:
        config["samples"] = args.samples
    if args.device is not None:
        config["device"] = args.device
    if args.methods is not None:
        config["methods"] = args.methods
    if args.lookahead is not None:
        config["lookahead"] = args.lookahead
    if args.widths is not None:
        config["widths"] = args.widths
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
