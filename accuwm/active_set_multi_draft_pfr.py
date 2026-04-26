"""
Active-set multi-draft speculative sampling with finite MPFR races.

This variant keeps ``B`` independent draft-indexed speculative streams instead
of merging equal draft contexts into a multiplicity tree.  At verification time
the active set contains the streams that still match the emitted target path.
For every step, the target token is selected from the active stream whose
target-side MPFR mapped time is smallest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .finite_multi_draft_pfr import (
    FiniteMPFRResult,
    finite_mpfr_tokens_from_proposal,
)
from .multi_draft_pfr import (
    ContextKey,
    _advance_cache,
    _batch_from_keys,
    _context_key,
    _new_context_cache,
    _repeat_cache,
    _select_cache_row,
)
from .pfr import AbstractLabeler, PFRSourceFactory, PrefixLabeler, SharedPFRSource
from .utils import cache_len, process_logits, truncate_cache


@dataclass
class ActiveSetMultiDraftBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    draft_context_count: int
    target_context_count: int
    active_counts: list[int]
    draft_mpfr_results: list[FiniteMPFRResult]
    target_mpfr_results: list[FiniteMPFRResult]
    target_past_key_values: any
    draft_past_key_values: any
    got_eos: bool


def _indexed_source_label(base_label: bytes, draft_index: int) -> bytes:
    return (
        b"active-set-draft-index::"
        + int(draft_index).to_bytes(8, byteorder="big", signed=False)
        + b"::"
        + base_label
    )


def _indexed_source(
    context_cache,
    source_factory: PFRSourceFactory,
    draft_index: int,
    context: ContextKey,
) -> SharedPFRSource:
    return source_factory.build(_indexed_source_label(context_cache.label(context), draft_index))


@torch.no_grad()
def build_active_set_drafts(
    ref_model,
    input_ids: LongTensor,
    root: ContextKey,
    lookahead: int,
    num_drafts: int,
    context_cache,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
) -> tuple[
    list[list[ContextKey]],
    list[list[int]],
    list[dict[int, any]],
    dict[int, tuple[any, int]],
    list[FiniteMPFRResult],
]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if lookahead <= 0:
        raise ValueError("lookahead must be positive")
    if num_drafts <= 0:
        raise ValueError("num_drafts must be positive")
    if input_ids.shape[0] != 1:
        raise AssertionError("only batch_size=1 is supported")

    device = input_ids.device
    cached_n = cache_len(past_key_values)
    contexts_by_depth: list[list[ContextKey]] = [[root for _ in range(num_drafts)]]
    draft_tokens_by_depth: list[list[int]] = []
    draft_pasts_by_depth: list[dict[int, any]] = []
    draft_mpfr_results: list[FiniteMPFRResult] = []
    current_contexts = [root for _ in range(num_drafts)]
    running_ids = _batch_from_keys(current_contexts, device)
    running_past = _repeat_cache(past_key_values, num_drafts)
    input_tokens = running_ids[:, cached_n:] if cached_n > 0 else running_ids

    for _depth in range(lookahead):
        output = ref_model(
            input_tokens,
            past_key_values=running_past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)

        step_tokens: list[int] = []
        step_pasts: dict[int, any] = {}
        next_contexts: list[ContextKey] = []
        for b, context in enumerate(current_contexts):
            step_pasts[b] = _select_cache_row(output.past_key_values, b)
            source = _indexed_source(context_cache, source_factory, b, context)
            mpfr_result = finite_mpfr_tokens_from_proposal(
                logprobs[b],
                source=source,
                num_samples=1,
                proposal=proposal,
                max_proposals=max_proposals,
                allow_incomplete=allow_incomplete,
                return_result=True,
            )
            token = int(mpfr_result.tokens[0].item())
            step_tokens.append(token)
            next_contexts.append(context + (token,))
            draft_mpfr_results.append(mpfr_result)

        draft_tokens_by_depth.append(step_tokens)
        draft_pasts_by_depth.append(step_pasts)
        current_contexts = next_contexts
        contexts_by_depth.append(current_contexts)
        new_tokens = torch.tensor(step_tokens, device=device, dtype=torch.long).unsqueeze(1)
        running_ids = torch.cat([running_ids, new_tokens], dim=1)
        running_past = output.past_key_values
        input_tokens = new_tokens

    final_past_refs = {
        b: (_select_cache_row(running_past, b), draft_tokens_by_depth[-1][b])
        for b in range(num_drafts)
    }

    return (
        contexts_by_depth,
        draft_tokens_by_depth,
        draft_pasts_by_depth,
        final_past_refs,
        draft_mpfr_results,
    )


@torch.no_grad()
def evaluate_active_set_targets(
    model,
    contexts_by_depth: list[list[ContextKey]],
    context_cache,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
) -> tuple[
    list[list[int]],
    list[list[float]],
    list[list[FloatTensor]],
    list[dict[int, any]],
    list[FiniteMPFRResult],
]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    device = model.device
    cached_n = cache_len(past_key_values)
    target_tokens_by_depth: list[list[int]] = []
    target_scores_by_depth: list[list[float]] = []
    target_logprobs_by_depth: list[list[FloatTensor]] = []
    target_pasts_by_depth: list[dict[int, any]] = []
    target_mpfr_results: list[FiniteMPFRResult] = []

    for contexts in contexts_by_depth:
        batch_ids = _batch_from_keys(contexts, device)
        batch_past = _repeat_cache(past_key_values, batch_ids.shape[0])
        input_tokens = batch_ids[:, cached_n:] if cached_n > 0 else batch_ids
        output = model(
            input_tokens,
            past_key_values=batch_past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = process_logits(batch_ids, output.logits[:, -1, :], **process_logits_kwargs)
        logprobs = F.log_softmax(logits, dim=-1)

        step_tokens: list[int] = []
        step_scores: list[float] = []
        step_logprobs: list[FloatTensor] = []
        step_pasts: dict[int, any] = {}
        for b, context in enumerate(contexts):
            step_pasts[b] = _select_cache_row(output.past_key_values, b)
            source = _indexed_source(context_cache, source_factory, b, context)
            mpfr_result = finite_mpfr_tokens_from_proposal(
                logprobs[b],
                source=source,
                num_samples=1,
                proposal=proposal,
                max_proposals=max_proposals,
                allow_incomplete=allow_incomplete,
                return_result=True,
            )
            step_tokens.append(int(mpfr_result.tokens[0].item()))
            score = float(mpfr_result.scores[0].item())
            step_scores.append(score if math.isfinite(score) else float("inf"))
            step_logprobs.append(logprobs[b].unsqueeze(0))
            target_mpfr_results.append(mpfr_result)

        target_tokens_by_depth.append(step_tokens)
        target_scores_by_depth.append(step_scores)
        target_logprobs_by_depth.append(step_logprobs)
        target_pasts_by_depth.append(step_pasts)

    return (
        target_tokens_by_depth,
        target_scores_by_depth,
        target_logprobs_by_depth,
        target_pasts_by_depth,
        target_mpfr_results,
    )


@torch.no_grad()
def evaluate_active_set_targets_onepass(
    model,
    contexts_by_depth: list[list[ContextKey]],
    context_cache,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
) -> tuple[
    list[list[int]],
    list[list[float]],
    list[list[FloatTensor]],
    list[dict[int, any]],
    list[FiniteMPFRResult],
]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    device = model.device
    block_len = len(contexts_by_depth) - 1
    if block_len < 0:
        raise ValueError("contexts_by_depth must not be empty")
    full_contexts = contexts_by_depth[-1]
    batch_ids = _batch_from_keys(full_contexts, device)
    cached_n = cache_len(past_key_values)
    input_tokens = batch_ids[:, cached_n:] if cached_n > 0 else batch_ids
    output = model(
        input_tokens,
        past_key_values=_repeat_cache(past_key_values, batch_ids.shape[0]),
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits = output.logits[:, -block_len - 1 :, :]

    target_tokens_by_depth: list[list[int]] = []
    target_scores_by_depth: list[list[float]] = []
    target_logprobs_by_depth: list[list[FloatTensor]] = []
    target_pasts_by_depth: list[dict[int, any]] = []
    target_mpfr_results: list[FiniteMPFRResult] = []

    for depth, contexts in enumerate(contexts_by_depth):
        context_ids = _batch_from_keys(contexts, device)
        step_logits = process_logits(
            context_ids,
            logits[:, depth, :],
            **process_logits_kwargs,
        )
        step_logprobs = F.log_softmax(step_logits, dim=-1)

        step_tokens: list[int] = []
        step_scores: list[float] = []
        step_logprobs_list: list[FloatTensor] = []
        for b, context in enumerate(contexts):
            source = _indexed_source(context_cache, source_factory, b, context)
            mpfr_result = finite_mpfr_tokens_from_proposal(
                step_logprobs[b],
                source=source,
                num_samples=1,
                proposal=proposal,
                max_proposals=max_proposals,
                allow_incomplete=allow_incomplete,
                return_result=True,
            )
            step_tokens.append(int(mpfr_result.tokens[0].item()))
            score = float(mpfr_result.scores[0].item())
            step_scores.append(score if math.isfinite(score) else float("inf"))
            step_logprobs_list.append(step_logprobs[b].unsqueeze(0))
            target_mpfr_results.append(mpfr_result)

        target_tokens_by_depth.append(step_tokens)
        target_scores_by_depth.append(step_scores)
        target_logprobs_by_depth.append(step_logprobs_list)
        target_pasts_by_depth.append(
            {
                b: (output.past_key_values, b, len(context))
                for b, context in enumerate(contexts)
            }
        )

    return (
        target_tokens_by_depth,
        target_scores_by_depth,
        target_logprobs_by_depth,
        target_pasts_by_depth,
        target_mpfr_results,
    )


def _argmin_active(scores: list[float], active: set[int]) -> int:
    if not active:
        raise ValueError("active set is empty")
    return min(active, key=lambda b: (scores[b], b))


def _materialize_past(past_ref):
    if isinstance(past_ref, tuple) and len(past_ref) == 3:
        full_past, row, new_len = past_ref
        return truncate_cache(_select_cache_row(full_past, row), new_len)
    return past_ref


@torch.no_grad()
def _evaluate_context_pasts(model, contexts: list[ContextKey], past_key_values=None) -> dict[int, any]:
    device = model.device
    cached_n = cache_len(past_key_values)
    batch_ids = _batch_from_keys(contexts, device)
    batch_past = _repeat_cache(past_key_values, batch_ids.shape[0])
    input_tokens = batch_ids[:, cached_n:] if cached_n > 0 else batch_ids
    output = model(
        input_tokens,
        past_key_values=batch_past,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return {
        row: _select_cache_row(output.past_key_values, row)
        for row in range(batch_ids.shape[0])
    }


@torch.no_grad()
def active_set_multi_draft_pfr_block(
    model,
    ref_model,
    input_ids: LongTensor,
    root: ContextKey | None,
    lookahead: int,
    num_drafts: int,
    labeler: AbstractLabeler,
    source_factory: PFRSourceFactory,
    max_new_tokens: int,
    past_key_values=None,
    ref_past_key_values=None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
) -> ActiveSetMultiDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if input_ids.shape[0] != 1:
        raise AssertionError("only batch_size=1 is supported")

    device = model.device
    input_ids = input_ids.to(device)
    if root is None:
        root = _context_key(input_ids)

    source_factory = source_factory
    target_context_cache = _new_context_cache(labeler, source_factory, device)
    draft_context_cache = _new_context_cache(labeler, source_factory, ref_model.device)
    block_len = min(lookahead, max_new_tokens)

    (
        contexts_by_depth,
        draft_tokens_by_depth,
        draft_pasts_by_depth,
        final_draft_past_refs,
        draft_mpfr_results,
    ) = build_active_set_drafts(
        ref_model=ref_model,
        input_ids=input_ids.to(ref_model.device),
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        context_cache=draft_context_cache,
        source_factory=source_factory,
        past_key_values=ref_past_key_values,
        max_vocab_size=model.config.vocab_size,
        process_logits_kwargs=process_logits_kwargs,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
        proposal=proposal,
    )
    (
        target_tokens_by_depth,
        target_scores_by_depth,
        target_logprobs_by_depth,
        target_pasts_by_depth,
        target_mpfr_results,
    ) = evaluate_active_set_targets_onepass(
        model=model,
        contexts_by_depth=contexts_by_depth,
        context_cache=target_context_cache,
        source_factory=source_factory,
        past_key_values=past_key_values,
        process_logits_kwargs=process_logits_kwargs,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
        proposal=proposal,
    )

    active = set(range(num_drafts))
    output_tokens: list[int] = []
    output_logprobs: list[FloatTensor] = []
    active_counts: list[int] = []
    accepted = True
    accepted_count = 0
    got_eos = False
    target_past_for_next = past_key_values
    draft_past_for_next = ref_past_key_values

    for depth in range(block_len):
        active_counts.append(len(active))
        winner_b = _argmin_active(target_scores_by_depth[depth], active)
        token = target_tokens_by_depth[depth][winner_b]
        output_tokens.append(token)
        output_logprobs.append(target_logprobs_by_depth[depth][winner_b])
        target_past_for_next = _materialize_past(target_pasts_by_depth[depth][winner_b])
        draft_past_for_next = draft_pasts_by_depth[depth].get(winner_b, draft_past_for_next)

        if token == model.config.eos_token_id:
            got_eos = True
            break

        active = {
            b
            for b in active
            if draft_tokens_by_depth[depth][b] == token
        }
        if not active:
            accepted = False
            break

        accepted_count += 1
        if len(output_tokens) >= max_new_tokens:
            break

    if accepted and not got_eos and len(output_tokens) < max_new_tokens:
        active_counts.append(len(active))
        winner_b = _argmin_active(target_scores_by_depth[block_len], active)
        token = target_tokens_by_depth[block_len][winner_b]
        output_tokens.append(token)
        output_logprobs.append(target_logprobs_by_depth[block_len][winner_b])
        target_past_for_next = _materialize_past(target_pasts_by_depth[block_len][winner_b])
        final_base_past, final_token = final_draft_past_refs[winner_b]
        draft_past_for_next = _advance_cache(ref_model, final_base_past, final_token)
        if token == model.config.eos_token_id:
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=1)

    return ActiveSetMultiDraftBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        draft_context_count=block_len * num_drafts,
        target_context_count=(block_len + 1) * num_drafts,
        active_counts=active_counts,
        draft_mpfr_results=draft_mpfr_results,
        target_mpfr_results=target_mpfr_results,
        target_past_key_values=target_past_for_next,
        draft_past_key_values=draft_past_for_next,
        got_eos=got_eos,
    )


@torch.no_grad()
def active_set_multi_draft_pfr_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes,
    labeler: AbstractLabeler | None = None,
    process_logits_kwargs: dict | None = None,
    return_meta: bool = False,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
):
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if labeler is None:
        labeler = PrefixLabeler()
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")

    model.eval()
    ref_model.eval()
    source_factory = PFRSourceFactory(private_key=private_key)
    input_ids = input_ids.to(model.device)
    current_key = _context_key(input_ids)
    past_key_values = None
    ref_past_key_values = None
    generated = 0

    while generated < max_length:
        block = active_set_multi_draft_pfr_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            root=current_key,
            lookahead=lookahead,
            num_drafts=num_drafts,
            labeler=labeler,
            source_factory=source_factory,
            max_new_tokens=max_length - generated,
            past_key_values=past_key_values,
            ref_past_key_values=ref_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            proposal=proposal,
        )
        meta = {
            "accepted_count": block.accepted_count,
            "draft_context_count": block.draft_context_count,
            "target_context_count": block.target_context_count,
            "active_counts": block.active_counts,
            "draft_len": min(lookahead, max_length - generated),
            "num_drafts": num_drafts,
            "proposal": proposal,
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat([input_ids, block.output_ids], dim=1)
        current_key = current_key + tuple(int(token) for token in block.output_ids[0].tolist())
        past_key_values = block.target_past_key_values
        ref_past_key_values = block.draft_past_key_values
        generated += block.output_ids.shape[1]
        if block.got_eos:
            break


def active_set_multi_draft_pfr_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    private_key: bytes = b"1234",
    num_drafts: int = 2,
    B: int | None = None,
    labeler: AbstractLabeler | None = None,
    process_logits_kwargs: dict | None = None,
    return_meta: bool = False,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "model",
):
    if B is not None:
        num_drafts = B
    yield from active_set_multi_draft_pfr_generator(
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
