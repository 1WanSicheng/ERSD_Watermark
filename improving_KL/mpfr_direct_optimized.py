#!/usr/bin/env python3
"""
Optimized exact finite-support MPFR generators for multi-draft speculative decoding.

This module is intentionally self-contained.  It provides two exact MPFR variants
for finite/top-k distributions:

  proposal="direct_topk"
      Uses direct finite-support mapped-Poisson clocks for both draft and target,
      but verifies target contexts sequentially along the realized path.

  proposal="batched_target_direct_topk"
      Uses the same exact direct finite-support MPFR clocks, but computes all
      target logits needed for a speculative block with one batched target-model
      forward pass over the depth-L draft leaves.  This should reduce actual
      target forward calls per token.

The direct finite-support MPFR primitive is exact for the processed distribution
after temperature/top-k filtering.  For a context c and token u, we construct
deterministic Poisson arrival increments from

    H(private_key, c, u, arrival_index),

so the draft and target distributions at the same context share the same Poisson
source.  For distribution P(.|c), the score of token u at its j-th local arrival is

    S_{u,j} = (E_{u,1}+...+E_{u,j}) / P(u|c).

The first B MPFR samples are the labels of the B smallest scores over all active
tokens u and local indices j=1,...,B.  This is exactly the first B arrivals of
independent Poisson clocks with rates P(u|c), hence the ordered labels are i.i.d.
P(.|c), allowing repetitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import math
import struct

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor


ContextKey = Tuple[int, ...]


@dataclass(frozen=True)
class DirectMPFRResult:
    tokens: LongTensor
    scores: FloatTensor
    local_indices: LongTensor


@dataclass
class MPFRBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    attempted_draft_tokens: int
    draft_tree_size: int
    target_context_count: int
    target_forward_calls: int
    draft_forward_calls: int
    got_eos: bool


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def _model_device(model) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def _context_key(input_ids: LongTensor) -> ContextKey:
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must have shape [1, sequence_length]")
    return tuple(int(x) for x in input_ids[0].detach().cpu().tolist())


def _ids_from_context(context: ContextKey, device: torch.device) -> LongTensor:
    return torch.tensor([list(context)], device=device, dtype=torch.long)


def _batch_from_contexts(contexts: List[ContextKey], device: torch.device) -> LongTensor:
    if not contexts:
        raise ValueError("cannot batch an empty context list")
    lengths = {len(c) for c in contexts}
    if len(lengths) != 1:
        raise ValueError("all contexts in a no-padding batch must have the same length")
    return torch.tensor([list(c) for c in contexts], device=device, dtype=torch.long)


def _temperature_for_target(process_logits_kwargs: Optional[Dict[str, Any]]) -> Any:
    if not process_logits_kwargs:
        return 1.0
    temp = process_logits_kwargs.get("temperature", 1.0)
    if isinstance(temp, (list, tuple)):
        return temp[0] if len(temp) > 0 else 1.0
    return temp


def _temperature_for_draft(process_logits_kwargs: Optional[Dict[str, Any]]) -> Any:
    if not process_logits_kwargs:
        return 1.0
    temp = process_logits_kwargs.get("temperature", 1.0)
    if isinstance(temp, (list, tuple)):
        return temp[1] if len(temp) > 1 else temp[0]
    return temp


def _top_k(process_logits_kwargs: Optional[Dict[str, Any]]) -> int:
    if not process_logits_kwargs:
        return 50
    return int(process_logits_kwargs.get("top_k", 50))


def _top_p(process_logits_kwargs: Optional[Dict[str, Any]]) -> float:
    # Kept for API compatibility.  The uploaded baseline code ignores top_p, so
    # this module also ignores it by default to keep the comparison aligned.
    if not process_logits_kwargs:
        return 1.0
    return float(process_logits_kwargs.get("top_p", 1.0))


def process_logits_exact(logits: Tensor, *, temperature: Any = 1.0, top_k: int = 50, top_p: float = 1.0) -> Tensor:
    """
    Apply the same minimal top-k + temperature processing as the uploaded code.

    The uploaded baseline keeps top_p for API compatibility but does not use it,
    so this function intentionally does not nucleus-filter either.
    """
    if top_k is not None and int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        vals, idx = torch.topk(logits, int(top_k), dim=-1, sorted=False)
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, idx, vals)
    else:
        out = logits

    temp = temperature
    if not torch.is_tensor(temp):
        temp = torch.tensor(float(temp), device=out.device, dtype=out.dtype)
    else:
        temp = temp.to(device=out.device, dtype=out.dtype)
    return out / temp


def _normalize_logprobs_from_processed_logits(processed_logits: Tensor) -> Tensor:
    return F.log_softmax(processed_logits.float(), dim=-1)


# -----------------------------------------------------------------------------
# Deterministic keyed Poisson source
# -----------------------------------------------------------------------------


class KeyedPoissonSource:
    def __init__(self, private_key: bytes | str):
        if isinstance(private_key, str):
            private_key = private_key.encode("utf-8")
        self.private_key = bytes(private_key)

    def for_context(self, context: ContextKey) -> "ContextPoissonSource":
        return ContextPoissonSource(self.private_key, context)


class ContextPoissonSource:
    def __init__(self, private_key: bytes, context: ContextKey):
        self.private_key = private_key
        self.context = tuple(int(x) for x in context)


def _hash_uniform_01(private_key: bytes, context: ContextKey, token: int, local_arrival_index: int) -> float:
    """
    Deterministically generate a float in (0,1) from a cryptographic hash.

    local_arrival_index is 1-based.
    """
    h = hashlib.blake2b(digest_size=8, key=private_key)
    # Domain separator.
    h.update(b"MPFR_DIRECT_CLOCK_V1")
    h.update(struct.pack("<I", len(context)))
    for x in context:
        h.update(struct.pack("<q", int(x)))
    h.update(struct.pack("<q", int(token)))
    h.update(struct.pack("<q", int(local_arrival_index)))
    value = int.from_bytes(h.digest(), "little", signed=False)
    # Midpoint mapping avoids exactly 0 and exactly 1.
    return (value + 0.5) / float(1 << 64)


def _exp_clock(private_key: bytes, context: ContextKey, token: int, local_arrival_index: int) -> float:
    u = _hash_uniform_01(private_key, context, token, local_arrival_index)
    return -math.log(u)


@torch.no_grad()
def direct_finite_mpfr_tokens_from_logprobs(
    logprobs: FloatTensor,
    source: ContextPoissonSource,
    num_samples: int,
    *,
    return_result: bool = False,
) -> LongTensor | DirectMPFRResult:
    """
    Exact B-sample finite-support MPFR for a processed log-probability vector.

    If logprobs has finite support S, this function returns the labels of the
    first `num_samples` arrivals of independent Poisson clocks with rates
    P(u), u in S.  It is exact for the distribution represented by logprobs.
    """
    if logprobs.dim() == 2:
        if logprobs.shape[0] != 1:
            raise ValueError("logprobs must be 1D or batch size 1")
        logprobs = logprobs[0]
    if logprobs.dim() != 1:
        raise ValueError("logprobs must be a 1D tensor")
    if num_samples <= 0:
        empty = torch.empty((0,), device=logprobs.device, dtype=torch.long)
        if return_result:
            return DirectMPFRResult(
                tokens=empty,
                scores=torch.empty((0,), device=logprobs.device, dtype=torch.float32),
                local_indices=torch.empty((0,), device=logprobs.device, dtype=torch.long),
            )
        return empty

    finite = torch.isfinite(logprobs)
    if not bool(finite.any().item()):
        raise ValueError("empty finite support in direct MPFR")

    active_tokens = torch.nonzero(finite, as_tuple=False).flatten()
    active_logp = logprobs[active_tokens].float()
    # Renormalize on finite support in case logits were not perfectly normalized.
    active_p = torch.softmax(active_logp, dim=0).detach().cpu().tolist()
    active_token_list = [int(t) for t in active_tokens.detach().cpu().tolist()]

    candidates: List[Tuple[float, int, int]] = []
    key = source.private_key
    context = source.context

    # B local arrivals per token are sufficient for the first B global arrivals:
    # no token can contribute more than B labels among the first B global labels.
    for token, p in zip(active_token_list, active_p):
        if p <= 0.0:
            continue
        arrival_time = 0.0
        for j in range(1, num_samples + 1):
            arrival_time += _exp_clock(key, context, token, j)
            score = arrival_time / p
            candidates.append((score, token, j))

    if len(candidates) < num_samples:
        raise RuntimeError("not enough MPFR candidates; this should be impossible for nonempty support")

    candidates.sort(key=lambda x: x[0])
    selected = candidates[:num_samples]
    tokens = torch.tensor([t for _, t, _ in selected], device=logprobs.device, dtype=torch.long)

    if not return_result:
        return tokens

    scores = torch.tensor([s for s, _, _ in selected], device=logprobs.device, dtype=torch.float32)
    local_indices = torch.tensor([j for _, _, j in selected], device=logprobs.device, dtype=torch.long)
    return DirectMPFRResult(tokens=tokens, scores=scores, local_indices=local_indices)


# -----------------------------------------------------------------------------
# Draft tree construction using exact direct finite MPFR
# -----------------------------------------------------------------------------


@dataclass
class DraftTree:
    levels: List[List[ContextKey]]
    multiplicities: Dict[ContextKey, int]
    draft_sets: Dict[ContextKey, set[int]]
    draft_tree_size: int
    draft_forward_calls: int


@torch.no_grad()
def build_draft_tree_direct(
    *,
    ref_model,
    root: ContextKey,
    lookahead: int,
    num_drafts: int,
    source_factory: KeyedPoissonSource,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> DraftTree:
    if lookahead <= 0:
        raise ValueError("lookahead must be positive")
    if num_drafts <= 0:
        raise ValueError("num_drafts must be positive")

    draft_device = _model_device(ref_model)
    draft_temp = _temperature_for_draft(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)

    levels: List[List[ContextKey]] = [[root]]
    multiplicities: Dict[ContextKey, int] = {root: int(num_drafts)}
    draft_sets: Dict[ContextKey, set[int]] = {}
    draft_tree_size = 0
    draft_forward_calls = 0

    for depth in range(1, lookahead + 1):
        prev_level = levels[depth - 1]
        if not prev_level:
            levels.append([])
            break

        batch_ids = _batch_from_contexts(prev_level, draft_device)
        out = ref_model(
            input_ids=batch_ids,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        draft_forward_calls += 1
        draft_tree_size += len(prev_level)

        logits = out.logits[:, -1, :]
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        processed = process_logits_exact(logits, temperature=draft_temp, top_k=top_k, top_p=top_p)
        logprobs = _normalize_logprobs_from_processed_logits(processed)

        next_level: List[ContextKey] = []
        seen_next: set[ContextKey] = set()

        for row, context in enumerate(prev_level):
            source = source_factory.for_context(context)
            draft_tokens = direct_finite_mpfr_tokens_from_logprobs(
                logprobs[row],
                source,
                multiplicities[context],
            )
            token_counts: Dict[int, int] = {}
            for t in draft_tokens.detach().cpu().tolist():
                token_counts[int(t)] = token_counts.get(int(t), 0) + 1

            draft_sets[context] = set(token_counts.keys())
            for token, count in token_counts.items():
                child = context + (int(token),)
                multiplicities[child] = count
                if child not in seen_next:
                    next_level.append(child)
                    seen_next.add(child)

        levels.append(next_level)

    return DraftTree(
        levels=levels,
        multiplicities=multiplicities,
        draft_sets=draft_sets,
        draft_tree_size=draft_tree_size,
        draft_forward_calls=draft_forward_calls,
    )


# -----------------------------------------------------------------------------
# Target logprob evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def _target_logprobs_one_context(
    *,
    model,
    context: ContextKey,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> FloatTensor:
    target_device = _model_device(model)
    ids = _ids_from_context(context, target_device)
    out = model(
        input_ids=ids,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits = out.logits[:, -1, :]
    if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
        logits = logits[..., :max_vocab_size]
    processed = process_logits_exact(
        logits,
        temperature=_temperature_for_target(process_logits_kwargs),
        top_k=_top_k(process_logits_kwargs),
        top_p=_top_p(process_logits_kwargs),
    )
    return _normalize_logprobs_from_processed_logits(processed)[0]


@torch.no_grad()
def _target_logprobs_from_leaves_one_forward(
    *,
    model,
    leaves: List[ContextKey],
    root_len: int,
    lookahead: int,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> Dict[ContextKey, FloatTensor]:
    """
    Run the target model once on all depth-L draft leaves and extract logits for
    every prefix context on those leaves.

    Because the model is causal, the logit at position |c|-1 in a full leaf
    sequence is exactly the next-token logit for prefix context c.
    """
    if not leaves:
        raise ValueError("cannot batch target over an empty leaf set")
    target_device = _model_device(model)
    batch_ids = _batch_from_contexts(leaves, target_device)
    out = model(
        input_ids=batch_ids,
        use_cache=False,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits_all = out.logits
    if max_vocab_size is not None and max_vocab_size < logits_all.shape[-1]:
        logits_all = logits_all[..., :max_vocab_size]

    target_temp = _temperature_for_target(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)

    logprobs_by_context: Dict[ContextKey, FloatTensor] = {}
    for row, leaf in enumerate(leaves):
        for d in range(0, lookahead + 1):
            context = leaf[: root_len + d]
            if context in logprobs_by_context:
                continue
            pos = root_len + d - 1
            raw_logits = logits_all[row, pos, :]
            processed = process_logits_exact(raw_logits, temperature=target_temp, top_k=top_k, top_p=top_p)
            logprobs_by_context[context] = _normalize_logprobs_from_processed_logits(processed)

    return logprobs_by_context


# -----------------------------------------------------------------------------
# Block variants
# -----------------------------------------------------------------------------


@torch.no_grad()
def mpfr_direct_topk_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    source_factory: KeyedPoissonSource,
    max_new_tokens: int,
    process_logits_kwargs: Optional[Dict[str, Any]],
) -> MPFRBlock:
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = _context_key(input_ids)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    draft_tree = build_draft_tree_direct(
        ref_model=ref_model,
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        source_factory=source_factory,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
    )

    current = root
    output_tokens: List[int] = []
    output_logprobs: List[FloatTensor] = []
    accepted_count = 0
    got_eos = False
    accepted_all = True
    target_context_count = 0
    target_forward_calls = 0

    for _depth in range(block_len):
        logprobs = _target_logprobs_one_context(
            model=model,
            context=current,
            process_logits_kwargs=process_logits_kwargs,
            max_vocab_size=max_vocab_size,
        )
        target_context_count += 1
        target_forward_calls += 1

        source = source_factory.for_context(current)
        token = int(direct_finite_mpfr_tokens_from_logprobs(logprobs, source, 1)[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)

        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True
            accepted_all = False
            break

        if token not in draft_tree.draft_sets.get(current, set()):
            accepted_all = False
            break

        accepted_count += 1
        current = current + (token,)
        if len(output_tokens) >= max_new_tokens:
            accepted_all = False
            break

    if accepted_all and not got_eos and len(output_tokens) < max_new_tokens:
        logprobs = _target_logprobs_one_context(
            model=model,
            context=current,
            process_logits_kwargs=process_logits_kwargs,
            max_vocab_size=max_vocab_size,
        )
        target_context_count += 1
        target_forward_calls += 1

        source = source_factory.for_context(current)
        token = int(direct_finite_mpfr_tokens_from_logprobs(logprobs, source, 1)[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)
    return MPFRBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        attempted_draft_tokens=block_len,
        draft_tree_size=draft_tree.draft_tree_size,
        target_context_count=target_context_count,
        target_forward_calls=target_forward_calls,
        draft_forward_calls=draft_tree.draft_forward_calls,
        got_eos=got_eos,
    )


@torch.no_grad()
def mpfr_batched_target_direct_topk_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    source_factory: KeyedPoissonSource,
    max_new_tokens: int,
    process_logits_kwargs: Optional[Dict[str, Any]],
) -> MPFRBlock:
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    draft_tree = build_draft_tree_direct(
        ref_model=ref_model,
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        source_factory=source_factory,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
    )

    leaves = draft_tree.levels[block_len]
    if not leaves:
        # This should not happen for positive num_drafts and nonempty model support, but
        # fall back to the sequential exact path for safety.
        return mpfr_direct_topk_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            lookahead=lookahead,
            num_drafts=num_drafts,
            source_factory=source_factory,
            max_new_tokens=max_new_tokens,
            process_logits_kwargs=process_logits_kwargs,
        )

    logprobs_by_context = _target_logprobs_from_leaves_one_forward(
        model=model,
        leaves=leaves,
        root_len=root_len,
        lookahead=block_len,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
    )

    current = root
    output_tokens: List[int] = []
    output_logprobs: List[FloatTensor] = []
    accepted_count = 0
    got_eos = False
    accepted_all = True

    for _depth in range(block_len):
        logprobs = logprobs_by_context[current]
        source = source_factory.for_context(current)
        token = int(direct_finite_mpfr_tokens_from_logprobs(logprobs, source, 1)[0].item())

        output_tokens.append(token)
        output_logprobs.append(logprobs)

        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True
            accepted_all = False
            break

        if token not in draft_tree.draft_sets.get(current, set()):
            accepted_all = False
            break

        accepted_count += 1
        current = current + (token,)
        if len(output_tokens) >= max_new_tokens:
            accepted_all = False
            break

    if accepted_all and not got_eos and len(output_tokens) < max_new_tokens:
        # If all L drafted positions are accepted, the bonus context is a depth-L
        # draft leaf and was included in the one target forward pass.
        logprobs = logprobs_by_context[current]
        source = source_factory.for_context(current)
        token = int(direct_finite_mpfr_tokens_from_logprobs(logprobs, source, 1)[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)

    return MPFRBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        attempted_draft_tokens=block_len,
        draft_tree_size=draft_tree.draft_tree_size,
        # Number of unique prefix contexts whose target distribution was extracted.
        target_context_count=len(logprobs_by_context),
        # Actual target model calls: one batched forward pass per speculative block.
        target_forward_calls=1,
        draft_forward_calls=draft_tree.draft_forward_calls,
        got_eos=got_eos,
    )


# -----------------------------------------------------------------------------
# Public generator API matching compare_decoding_benchmark.py
# -----------------------------------------------------------------------------


@torch.no_grad()
def finite_multi_draft_pfr_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes | str = b"1234",
    labeler: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    max_proposals: int = 100_000,       # accepted for API compatibility
    allow_incomplete: bool = False,     # accepted for API compatibility
    proposal: str = "batched_target_direct_topk",
) -> Iterable[Tuple[LongTensor, FloatTensor, Dict[str, Any]]]:
    """
    Yield speculative blocks from an exact direct finite-support MPFR sampler.

    proposal options:
        "direct_topk": sequential target verification with exact direct MPFR.
        "batched_target_direct_topk": one batched target forward per block.

    Both are exact for the processed finite/top-k model distributions.
    """
    del labeler, max_proposals, allow_incomplete  # API compatibility only.

    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    model.eval()
    ref_model.eval()

    source_factory = KeyedPoissonSource(private_key)
    device = _model_device(model)
    input_ids = input_ids.to(device)
    generated = 0

    while generated < max_length:
        remaining = int(max_length - generated)
        if remaining <= 0:
            break

        if proposal in {"direct_topk", "direct", "finite_support_direct"}:
            block = mpfr_direct_topk_block(
                model=model,
                ref_model=ref_model,
                input_ids=input_ids,
                lookahead=lookahead,
                num_drafts=num_drafts,
                source_factory=source_factory,
                max_new_tokens=remaining,
                process_logits_kwargs=process_logits_kwargs,
            )
        elif proposal in {"batched_target_direct_topk", "batched_direct_topk", "direct_topk_batched_target"}:
            block = mpfr_batched_target_direct_topk_block(
                model=model,
                ref_model=ref_model,
                input_ids=input_ids,
                lookahead=lookahead,
                num_drafts=num_drafts,
                source_factory=source_factory,
                max_new_tokens=remaining,
                process_logits_kwargs=process_logits_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown optimized MPFR proposal {proposal!r}. "
                "Use 'direct_topk' or 'batched_target_direct_topk'."
            )

        meta = {
            "accepted_count": block.accepted_count,
            "attempted_draft_tokens": block.attempted_draft_tokens,
            "draft_tree_size": block.draft_tree_size,
            "target_context_count": block.target_context_count,
            "target_forward_calls": block.target_forward_calls,
            "draft_forward_calls": block.draft_forward_calls,
            "draft_len": min(int(lookahead), remaining),
            "num_drafts": int(num_drafts),
            "proposal": proposal,
        }

        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat([input_ids, block.output_ids.to(device)], dim=1)
        generated += int(block.output_ids.shape[-1])

        if block.got_eos:
            break


def finite_multi_draft_pfr_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    private_key: bytes | str = b"1234",
    num_drafts: int = 2,
    B: Optional[int] = None,
    labeler: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "batched_target_direct_topk",
):
    if B is not None:
        num_drafts = int(B)
    yield from finite_multi_draft_pfr_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        lookahead=n,
        num_drafts=num_drafts,
        max_length=max_length,
        private_key=private_key,
        labeler=labeler,
        process_logits_kwargs=process_logits_kwargs,
        return_meta=return_meta,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
        proposal=proposal,
    )
