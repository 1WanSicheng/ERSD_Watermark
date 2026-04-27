#!/usr/bin/env python3
"""
Self-contained baselines for Qwen multi-draft speculative decoding experiments.

This file contains a clean local implementation of the algorithms needed for
comparison with the MPFR benchmark:

  * target_only_generator: ordinary target-model sampling.
  * single_draft_generator: standard single-draft speculative sampling.
  * list_level_identical_generator: the identical/homogeneous multi-draft
    list-level coupling baseline, matching the "strong_multi_draft" style in
    the uploaded List-Level Distribution Coupling code.

The public generator functions intentionally match the MPFR generator interface:

    generator(model, ref_model, input_ids, lookahead, num_drafts, max_length,
              private_key=None, process_logits_kwargs=None, return_meta=True, ...)

and yield either

    (output_ids, output_logprobs, meta)

or, if return_meta=False,

    (output_ids, output_logprobs).

Metric convention in meta:

    accepted_count      = accepted drafted tokens in this block.
    emitted_count       = number of generated tokens emitted in this block.
    target_context_count= number of target-model verification calls in this block.
    draft_tree_size     = number of drafter forward contexts in this block.

For list-level/single-draft speculative decoding, one block corresponds to one
large-model verification call, so BE = emitted_count averaged over blocks, while
AATPS = accepted_count averaged over blocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import math
import random

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from transformers import DynamicCache


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _safe_log(x: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(x.clamp(min=eps))


def _gumbel_noise(u: Tensor) -> Tensor:
    return -_safe_log(-_safe_log(u))


def gumbel_sample(logits: Tensor, noise: Optional[Tensor] = None, dim: int = -1) -> Tensor:
    if noise is None:
        noise = torch.empty_like(logits).uniform_(0.0, 1.0)
    return (logits + _gumbel_noise(noise)).argmax(dim=dim)


class LogitsProcessor:
    """Minimal top-k + temperature processor, matching the uploaded code style."""

    def __init__(self, temperature: Any = 1.0, top_p: float = 1.0, top_k: int = 50):
        self.temperature = temperature
        self.top_p = top_p  # kept for API compatibility; original code ignores top_p.
        self.top_k = int(top_k)

    def __call__(self, logits: Tensor) -> Tensor:
        if self.top_k is not None and self.top_k > 0 and self.top_k < logits.shape[-1]:
            val, ind = torch.topk(logits, self.top_k, sorted=False, dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(-1, ind, val)
        else:
            filtered = logits

        temp = self.temperature
        if not torch.is_tensor(temp):
            temp = torch.tensor(temp, device=filtered.device, dtype=filtered.dtype)
        else:
            temp = temp.to(device=filtered.device, dtype=filtered.dtype)
        return filtered / temp


def _model_device(model) -> torch.device:
    # Works for normal HF models and most device_map='auto' single-GPU cases.
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def _cache_len(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        try:
            return int(cache.get_seq_length())
        except TypeError:
            pass
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        return int(cache.key_cache[0].shape[-2])
    try:
        return int(cache[0][0].shape[-2])
    except Exception:
        return 0


def _new_cache_or_none() -> Any:
    # Passing None also works in modern Transformers, but DynamicCache is useful
    # for compatibility with the uploaded code.
    try:
        return DynamicCache()
    except Exception:
        return None


def _repeat_cache_inplace(cache: Any, batch_size: int) -> Any:
    if cache is None:
        return None
    if batch_size == 1:
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) == 0:
            return cache
        current_bsz = int(cache.key_cache[0].shape[0])
        if current_bsz == batch_size:
            return cache
        if current_bsz != 1:
            raise RuntimeError(f"cannot repeat cache with batch size {current_bsz} to {batch_size}")
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].repeat((batch_size, 1, 1, 1))
            cache.value_cache[i] = cache.value_cache[i].repeat((batch_size, 1, 1, 1))
        return cache
    # Tuple-style past_key_values fallback.
    repeated = []
    for layer in cache:
        k, v = layer[:2]
        repeated.append((k.repeat((batch_size, 1, 1, 1)), v.repeat((batch_size, 1, 1, 1))) + tuple(layer[2:]))
    return tuple(repeated)


def _select_and_truncate_cache_inplace(cache: Any, batch_index: int, seq_len: int) -> Any:
    if cache is None:
        return None
    seq_len = max(int(seq_len), 0)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][batch_index, :, :seq_len, :].unsqueeze(0).contiguous()
            cache.value_cache[i] = cache.value_cache[i][batch_index, :, :seq_len, :].unsqueeze(0).contiguous()
        return cache
    selected = []
    for layer in cache:
        k, v = layer[:2]
        selected.append((
            k[batch_index, :, :seq_len, :].unsqueeze(0).contiguous(),
            v[batch_index, :, :seq_len, :].unsqueeze(0).contiguous(),
        ) + tuple(layer[2:]))
    return tuple(selected)


def _split_temperatures(temperature: Any, num_drafts: int, device: torch.device) -> Tuple[Any, Any]:
    if isinstance(temperature, (list, tuple)):
        target_temp = torch.tensor(temperature[0], device=device).reshape((1, 1, 1))
        draft_values = list(temperature[1:])
        if len(draft_values) == 0:
            draft_values = [temperature[0]] * num_drafts
        if len(draft_values) == 1 and num_drafts > 1:
            draft_values = draft_values * num_drafts
        draft_temp = torch.tensor(draft_values[:num_drafts], device=device).reshape((-1, 1, 1))
        return target_temp, draft_temp
    temp = torch.tensor(float(temperature), device=device).reshape((1, 1, 1))
    return temp, temp


def _make_processors(process_logits_kwargs: Optional[Dict[str, Any]], num_drafts: int, device: torch.device):
    kwargs = dict(process_logits_kwargs or {})
    temperature = kwargs.get("temperature", 1.0)
    top_p = float(kwargs.get("top_p", 1.0))
    top_k = int(kwargs.get("top_k", 50))
    target_temp, draft_temp = _split_temperatures(temperature, num_drafts, device)
    return LogitsProcessor(target_temp, top_p, top_k), LogitsProcessor(draft_temp, top_p, top_k)


def _maybe_yield(output_ids: LongTensor, output_logprobs: Optional[Tensor], meta: Dict[str, Any], return_meta: bool):
    if return_meta:
        return output_ids, output_logprobs, meta
    return output_ids, output_logprobs


# -----------------------------------------------------------------------------
# Target-only baseline
# -----------------------------------------------------------------------------


@torch.no_grad()
def target_only_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes | str | None = None,
    labeler: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    **_: Any,
):
    del ref_model, lookahead, num_drafts, private_key, labeler
    model.eval()
    device = _model_device(model)
    input_ids = input_ids.to(device)
    target_processor, _draft_processor = _make_processors(process_logits_kwargs, 1, device)
    past_key_values = None
    generated = 0

    while generated < max_length:
        outputs = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        past_key_values = outputs.past_key_values
        logits = target_processor(outputs.logits[:, -1:, :])
        token = gumbel_sample(logits).reshape(1, 1)
        output_ids = token.to(device=device, dtype=torch.long)
        logprobs = F.log_softmax(logits, dim=-1)
        meta = {
            "accepted_count": 0,
            "emitted_count": 1,
            "target_context_count": 1,
            "draft_tree_size": 0,
            "draft_forward_calls": 0,
            "target_forward_calls": 1,
        }
        yield _maybe_yield(output_ids, logprobs, meta, return_meta)
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        generated += 1
        if getattr(model.config, "eos_token_id", None) is not None and int(output_ids[0, 0]) == int(model.config.eos_token_id):
            break


# -----------------------------------------------------------------------------
# Standard single-draft speculative sampling baseline
# -----------------------------------------------------------------------------


@torch.no_grad()
def _single_draft_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    target_past_key_values: Any,
    draft_past_key_values: Any,
    draft_len: int,
    target_processor: LogitsProcessor,
    draft_processor: LogitsProcessor,
    vocab_size: int,
):
    device = _model_device(model)
    draft_device = _model_device(ref_model)
    prefix_len = int(input_ids.shape[-1])

    # Draft generation, one path.
    draft_input_ids = input_ids.to(draft_device)
    draft_probs = []
    for _ in range(draft_len):
        outputs = ref_model(
            input_ids=draft_input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        draft_past_key_values = outputs.past_key_values
        logits = outputs.logits[..., :vocab_size]
        logits = draft_processor(logits[:, -1:, :])
        probs = F.softmax(logits, dim=-1).squeeze(0).squeeze(0)
        draft_probs.append(probs.to(device))
        next_id = gumbel_sample(logits).reshape(1, 1)
        draft_input_ids = torch.cat([draft_input_ids, next_id], dim=1)

    full_ids = draft_input_ids.to(device)

    # Target verification in one large-model call.
    outputs = model(
        input_ids=full_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    target_past_key_values = outputs.past_key_values
    logits = outputs.logits[..., :vocab_size]
    logits = target_processor(logits[:, -(draft_len + 1):, :])
    target_probs = F.softmax(logits, dim=-1)
    draft_ids = full_ids[:, -draft_len:]

    depth = 0
    residual = None
    for depth in range(draft_len):
        p = target_probs[0, depth, :]
        q = draft_probs[depth]
        x = draft_ids[0, depth]
        h = p[x] / q[x].clamp_min(1e-20)
        if random.random() >= float(torch.minimum(h, torch.ones_like(h)).item()):
            residual = p - q
            residual = torch.maximum(residual, torch.zeros_like(residual))
            if float(residual.sum().item()) <= 0.0:
                residual = p
            else:
                residual = residual / residual.sum()
            break
    else:
        depth = draft_len
        residual = target_probs[0, -1, :]

    last_token = torch.multinomial(residual, num_samples=1).to(device)
    new_prefix = full_ids[0, :prefix_len + depth]
    new_input_ids = torch.cat([new_prefix, last_token]).unsqueeze(0)
    output_ids = new_input_ids[:, prefix_len:]

    # Keep caches only up to the accepted draft prefix. The sampled last_token is
    # not in either cache yet and will be fed in the next block.
    cache_keep_len = prefix_len + depth
    target_past_key_values = _select_and_truncate_cache_inplace(target_past_key_values, 0, cache_keep_len)
    draft_past_key_values = _select_and_truncate_cache_inplace(draft_past_key_values, 0, cache_keep_len)

    meta = {
        "accepted_count": int(depth),
        "emitted_count": int(output_ids.shape[-1]),
        "target_context_count": 1,
        "draft_tree_size": int(draft_len),
        "draft_forward_calls": int(draft_len),
        "target_forward_calls": 1,
    }
    return output_ids, logits, meta, target_past_key_values, draft_past_key_values


@torch.no_grad()
def single_draft_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes | str | None = None,
    labeler: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    **_: Any,
):
    del num_drafts, private_key, labeler
    model.eval(); ref_model.eval()
    device = _model_device(model)
    input_ids = input_ids.to(device)
    vocab_size = min(int(getattr(model.config, "vocab_size", 10**9)), int(getattr(ref_model.config, "vocab_size", 10**9)))
    target_processor, draft_processor = _make_processors(process_logits_kwargs, 1, device)
    target_past = None
    draft_past = None
    generated = 0

    while generated < max_length:
        remaining = max_length - generated
        if remaining <= 1:
            for y in target_only_generator(model, ref_model, input_ids, 1, 1, 1, process_logits_kwargs=process_logits_kwargs, return_meta=True):
                output_ids, logprobs, meta = y
            yield _maybe_yield(output_ids, logprobs, meta, return_meta)
            input_ids = torch.cat([input_ids, output_ids], dim=1)
            generated += int(output_ids.shape[-1])
            break

        draft_len = min(int(lookahead), remaining - 1)
        output_ids, logprobs, meta, target_past, draft_past = _single_draft_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            target_past_key_values=target_past,
            draft_past_key_values=draft_past,
            draft_len=draft_len,
            target_processor=target_processor,
            draft_processor=draft_processor,
            vocab_size=vocab_size,
        )
        yield _maybe_yield(output_ids, logprobs, meta, return_meta)
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        generated += int(output_ids.shape[-1])
        if getattr(model.config, "eos_token_id", None) is not None and bool((output_ids == model.config.eos_token_id).any().item()):
            break


# -----------------------------------------------------------------------------
# Identical/homogeneous list-level coupling baseline
# -----------------------------------------------------------------------------


@torch.no_grad()
def _list_level_identical_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    target_past_key_values: Any,
    draft_past_key_values: Any,
    draft_len: int,
    num_drafts: int,
    target_processor: LogitsProcessor,
    draft_processor: LogitsProcessor,
    randomness: Tensor,
    position: int,
    vocab_size: int,
):
    device = _model_device(model)
    draft_device = _model_device(ref_model)
    prefix_len = int(input_ids.shape[-1])
    rn = randomness[position:(position + draft_len + 1), :, :]

    # Generate B homogeneous drafts in a batch.
    draft_input_ids = input_ids.to(draft_device).repeat((num_drafts, 1))
    draft_past_key_values = _repeat_cache_inplace(draft_past_key_values, num_drafts)

    for depth in range(draft_len):
        outputs = ref_model(
            input_ids=draft_input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        draft_past_key_values = outputs.past_key_values
        logits = outputs.logits[..., :vocab_size]
        logits = draft_processor(logits[:, -1:, :])
        noise = rn[depth, :, :].unsqueeze(1).to(draft_device)
        draft_ids = gumbel_sample(logits, noise=noise).reshape(num_drafts, 1)
        draft_input_ids = torch.cat([draft_input_ids, draft_ids], dim=1)

    full_ids = draft_input_ids.to(device)

    # Verify all drafts with one target forward pass.
    target_past_key_values = _repeat_cache_inplace(target_past_key_values, num_drafts)
    outputs = model(
        input_ids=full_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    target_past_key_values = outputs.past_key_values
    logits = outputs.logits[..., :vocab_size]
    logits = target_processor(logits[:, -(draft_len + 1):, :])
    draft_ids = full_ids[:, -draft_len:]

    rejected_mask = torch.zeros(num_drafts, dtype=torch.bool, device=device)
    alive_group_id = 0
    depth = 0

    for depth in range(draft_len):
        # Strong/identical list coupling: target uses the max of all draft noises,
        # matching the uploaded StrongMultiDraftStrategy.
        target_noise, _ = torch.max(rn[depth, :, :].to(device), dim=0)
        target_ids = gumbel_sample(logits[:, depth, :], noise=target_noise)

        accepted_this_depth = False
        for k in range(num_drafts):
            if bool(rejected_mask[k].item()):
                continue
            if int(draft_ids[k, depth].item()) == int(target_ids[k].item()):
                alive_group_id = k
                rejected_mask |= torch.ne(draft_ids[:, depth], draft_ids[k, depth])
                accepted_this_depth = True
                break
            rejected_mask[k] = True

        if torch.all(rejected_mask):
            break
        if not accepted_this_depth:
            break
    else:
        depth = draft_len

    # If all draft_len tokens were accepted, depth=draft_len and the next target
    # token is drawn from the depth=draft_len target distribution.
    target_noise, _ = torch.max(rn[depth, :, :].to(device), dim=0)
    last_token = gumbel_sample(logits[alive_group_id, depth, :], noise=target_noise).reshape(1).to(device)

    new_prefix = full_ids[alive_group_id, :prefix_len + depth]
    new_input_ids = torch.cat([new_prefix, last_token]).unsqueeze(0)
    output_ids = new_input_ids[:, prefix_len:]

    # Truncate caches to prefix + accepted draft tokens. The final sampled token
    # will be fed on the next block.
    cache_keep_len = prefix_len + depth
    target_past_key_values = _select_and_truncate_cache_inplace(target_past_key_values, alive_group_id, cache_keep_len)
    draft_past_key_values = _select_and_truncate_cache_inplace(draft_past_key_values, alive_group_id, cache_keep_len)

    meta = {
        "accepted_count": int(depth),
        "emitted_count": int(output_ids.shape[-1]),
        "target_context_count": 1,
        "draft_tree_size": int(draft_len * num_drafts),
        "draft_forward_calls": int(draft_len),
        "target_forward_calls": 1,
        "alive_group_id": int(alive_group_id),
    }
    return output_ids, logits, meta, target_past_key_values, draft_past_key_values


@torch.no_grad()
def list_level_identical_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes | str | None = None,
    labeler: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    **_: Any,
):
    del private_key, labeler
    if num_drafts <= 0:
        raise ValueError("num_drafts must be positive")
    model.eval(); ref_model.eval()
    device = _model_device(model)
    input_ids = input_ids.to(device)
    vocab_size = min(int(getattr(model.config, "vocab_size", 10**9)), int(getattr(ref_model.config, "vocab_size", 10**9)))
    target_processor, draft_processor = _make_processors(process_logits_kwargs, num_drafts, device)

    # Common randomness for the whole generation, as in the uploaded invariant/strong generator.
    randomness = torch.empty((max_length + int(lookahead) + 1, int(num_drafts), vocab_size), device=device)
    randomness.uniform_()

    target_past = None
    draft_past = None
    generated = 0

    while generated < max_length:
        remaining = max_length - generated
        if remaining <= 1:
            for y in target_only_generator(model, ref_model, input_ids, 1, 1, 1, process_logits_kwargs=process_logits_kwargs, return_meta=True):
                output_ids, logprobs, meta = y
            yield _maybe_yield(output_ids, logprobs, meta, return_meta)
            input_ids = torch.cat([input_ids, output_ids], dim=1)
            generated += int(output_ids.shape[-1])
            break

        draft_len = min(int(lookahead), remaining - 1)
        output_ids, logprobs, meta, target_past, draft_past = _list_level_identical_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            target_past_key_values=target_past,
            draft_past_key_values=draft_past,
            draft_len=draft_len,
            num_drafts=int(num_drafts),
            target_processor=target_processor,
            draft_processor=draft_processor,
            randomness=randomness,
            position=generated,
            vocab_size=vocab_size,
        )
        yield _maybe_yield(output_ids, logprobs, meta, return_meta)
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        generated += int(output_ids.shape[-1])
        if getattr(model.config, "eos_token_id", None) is not None and bool((output_ids == model.config.eos_token_id).any().item()):
            break
