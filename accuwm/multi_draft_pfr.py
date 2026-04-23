"""
Watermarkable multi-draft speculative sampling with shared PFR sources.

This module implements the simplified multi-draft algorithm where each context
owns a keyed Poisson race source.  Draft branches are generated from the first
``B`` arrivals of the draft-model race, then target-model winners are evaluated
on every internal context and the realized target path is followed until the
first rejection.

The implementation intentionally keeps the branching logic explicit.  It does
not try to maintain KV caches across speculative branches; contexts at the same
tree depth are evaluated as a batch instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils._pytree import tree_map
from transformers.cache_utils import DynamicCache

from .pfr import (
    AbstractLabeler,
    PFRSourceFactory,
    PrefixLabeler,
    SharedPFRSource,
    _safe_log,
)
from .utils import cache_is_dynamic, cache_is_legacy, cache_len, process_logits


ContextKey = tuple[int, ...]


@dataclass
class MultiDraftBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    draft_tree_size: int
    target_context_count: int
    target_past_key_values: any
    draft_past_key_values: any
    got_eos: bool


def _context_key(ids: LongTensor) -> ContextKey:
    return tuple(int(x) for x in ids[0].detach().cpu().tolist())


def _batch_from_keys(keys: Iterable[ContextKey], device) -> LongTensor:
    key_list = list(keys)
    if not key_list:
        raise ValueError("cannot build an empty context batch")
    return torch.tensor(key_list, device=device, dtype=torch.long)


def _prefix_label_from_key(key: ContextKey, include_length: bool = True) -> bytes:
    payload = b"".join(int(token).to_bytes(4, "little", signed=True) for token in key)
    if not include_length:
        return payload
    return len(key).to_bytes(8, byteorder="big", signed=False) + payload


@dataclass
class ContextRuntimeCache:
    labeler: AbstractLabeler
    source_factory: PFRSourceFactory
    device: torch.device
    ids_by_key: dict[ContextKey, LongTensor]
    label_by_key: dict[ContextKey, bytes]
    source_by_key: dict[ContextKey, SharedPFRSource]

    def ids(self, key: ContextKey) -> LongTensor:
        ids = self.ids_by_key.get(key)
        if ids is None:
            ids = torch.tensor([key], device=self.device, dtype=torch.long)
            self.ids_by_key[key] = ids
        return ids

    def label(self, key: ContextKey) -> bytes:
        label = self.label_by_key.get(key)
        if label is None:
            if isinstance(self.labeler, PrefixLabeler):
                label = _prefix_label_from_key(
                    key,
                    include_length=self.labeler.include_length,
                )
            else:
                label = self.labeler.label(self.ids(key))
            self.label_by_key[key] = label
        return label

    def source(self, key: ContextKey) -> SharedPFRSource:
        source = self.source_by_key.get(key)
        if source is None:
            source = self.source_factory.build(self.label(key))
            self.source_by_key[key] = source
        return source


def _repeat_cache(past_key_values, batch_size: int):
    if past_key_values is None:
        return None
    if cache_is_legacy(past_key_values):
        return tree_map(lambda x: x.repeat((batch_size, 1, 1, 1)), past_key_values)
    if cache_is_dynamic(past_key_values):
        repeated = DynamicCache()
        repeated.key_cache = [
            key.repeat((batch_size, 1, 1, 1)) for key in past_key_values.key_cache
        ]
        repeated.value_cache = [
            value.repeat((batch_size, 1, 1, 1)) for value in past_key_values.value_cache
        ]
        return repeated
    return None


def _select_cache_row(past_key_values, row: int):
    if past_key_values is None:
        return None
    if cache_is_legacy(past_key_values):
        return tree_map(lambda x: x[row : row + 1], past_key_values)
    if cache_is_dynamic(past_key_values):
        selected = DynamicCache()
        selected.key_cache = [key[row : row + 1] for key in past_key_values.key_cache]
        selected.value_cache = [value[row : row + 1] for value in past_key_values.value_cache]
        return selected
    return None


@torch.no_grad()
def _advance_cache(model, past_key_values, token_id: int):
    token = torch.tensor([[token_id]], device=model.device, dtype=torch.long)
    output = model(
        token,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    return output.past_key_values


def _new_context_cache(
    labeler: AbstractLabeler,
    source_factory: PFRSourceFactory,
    device,
    label_by_key: dict[ContextKey, bytes] | None = None,
    source_by_key: dict[ContextKey, SharedPFRSource] | None = None,
) -> ContextRuntimeCache:
    return ContextRuntimeCache(
        labeler=labeler,
        source_factory=source_factory,
        device=torch.device(device),
        ids_by_key={},
        label_by_key=label_by_key if label_by_key is not None else {},
        source_by_key=source_by_key if source_by_key is not None else {},
    )


def ms_pfr_tokens_from_logprobs(
    logprobs: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    device,
) -> LongTensor:
    """
    Return the token ids for the first ``num_samples`` arrivals in a shared
    Poisson race.

    For each vocabulary item ``v`` with probability ``p_v``, the arrival process
    has rate ``p_v``.  The kth arrival time for that item is the cumulative sum
    of kth independent ``Exp(1)`` noises divided by ``p_v``.  Taking the first
    global arrivals can therefore produce repeated token ids, which are later
    collapsed into branch multiplicities.
    """
    if num_samples <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    if logprobs.dim() == 2:
        if logprobs.shape[0] != 1:
            raise ValueError("batched MS-PFR is handled by the caller")
        logprobs = logprobs[0]
    probs = logprobs.exp()
    vocab_size = probs.shape[-1]

    uniform_noise = source.uniform_noise((num_samples, vocab_size), device=device)
    exp_interarrival = -_safe_log(uniform_noise)
    arrival_times = torch.cumsum(exp_interarrival, dim=0)
    arrival_times = torch.where(
        probs.unsqueeze(0) > 0,
        arrival_times / probs.unsqueeze(0),
        torch.full_like(arrival_times, float("inf")),
    )

    flat_winners = torch.topk(
        arrival_times.flatten(),
        k=num_samples,
        largest=False,
        sorted=True,
    ).indices
    return (flat_winners % vocab_size).to(torch.long)


def _single_ms_pfr_from_logits(
    logits: FloatTensor,
    source: SharedPFRSource,
    device,
) -> tuple[int, FloatTensor]:
    logprobs = F.log_softmax(logits, dim=-1)
    token = ms_pfr_tokens_from_logprobs(
        logprobs,
        source=source,
        num_samples=1,
        device=device,
    )[0]
    return int(token.item()), logprobs


@torch.no_grad()
def build_multi_draft_tree(
    ref_model,
    input_ids: LongTensor,
    root: ContextKey,
    lookahead: int,
    num_drafts: int,
    context_cache: ContextRuntimeCache,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
) -> tuple[
    list[list[ContextKey]],
    dict[ContextKey, int],
    dict[ContextKey, set[int]],
    dict[ContextKey, SharedPFRSource],
    dict[ContextKey, any],
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
    levels: list[list[ContextKey]] = [[root]]
    multiplicities: dict[ContextKey, int] = {root: num_drafts}
    draft_sets: dict[ContextKey, set[int]] = {}
    sources: dict[ContextKey, SharedPFRSource] = {}
    past_by_context: dict[ContextKey, any] = {}
    cached_n = cache_len(past_key_values)

    for depth in range(1, lookahead + 1):
        prev_level = levels[depth - 1]
        batch_ids = _batch_from_keys(prev_level, device)
        batch_past = _repeat_cache(past_key_values, batch_ids.shape[0])
        input_tokens = batch_ids[:, cached_n:] if cached_n > 0 else batch_ids
        output = ref_model(
            input_tokens,
            past_key_values=batch_past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(batch_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)

        next_level: list[ContextKey] = []
        seen_next: set[ContextKey] = set()
        for row, context in enumerate(prev_level):
            past_by_context[context] = _select_cache_row(output.past_key_values, row)
            source = context_cache.source(context)
            sources[context] = source
            draft_tokens = ms_pfr_tokens_from_logprobs(
                logprobs[row],
                source=source,
                num_samples=multiplicities[context],
                device=device,
            )
            token_counts: dict[int, int] = {}
            for token in draft_tokens.detach().cpu().tolist():
                token_counts[int(token)] = token_counts.get(int(token), 0) + 1
            draft_sets[context] = set(token_counts)

            for token, count in token_counts.items():
                child = context + (token,)
                multiplicities[child] = count
                if child not in seen_next:
                    next_level.append(child)
                    seen_next.add(child)
        levels.append(next_level)
        if not next_level:
            break

    return levels, multiplicities, draft_sets, sources, past_by_context


@torch.no_grad()
def evaluate_target_contexts(
    model,
    levels: list[list[ContextKey]],
    context_cache: ContextRuntimeCache,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
) -> tuple[dict[ContextKey, int], dict[ContextKey, FloatTensor], dict[ContextKey, SharedPFRSource]]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    device = model.device
    winners: dict[ContextKey, int] = {}
    logprobs_by_context: dict[ContextKey, FloatTensor] = {}
    sources: dict[ContextKey, SharedPFRSource] = {}
    cached_n = cache_len(past_key_values)

    for level in levels[:-1]:
        if not level:
            continue
        batch_ids = _batch_from_keys(level, device)
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
        logits = output.logits[:, -1, :]
        logits = process_logits(batch_ids, logits, **process_logits_kwargs)
        for row, context in enumerate(level):
            source = context_cache.source(context)
            token, logprobs = _single_ms_pfr_from_logits(
                logits[row].unsqueeze(0),
                source=source,
                device=device,
            )
            winners[context] = token
            logprobs_by_context[context] = logprobs
            sources[context] = source

    return winners, logprobs_by_context, sources


@torch.no_grad()
def evaluate_target_context(
    model,
    context: ContextKey,
    context_cache: ContextRuntimeCache,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
) -> tuple[int, FloatTensor, SharedPFRSource, any]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    device = model.device
    context_ids = context_cache.ids(context)
    cached_n = cache_len(past_key_values)
    input_tokens = context_ids[:, cached_n:] if cached_n > 0 else context_ids
    output = model(
        input_tokens,
        past_key_values=_repeat_cache(past_key_values, 1),
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits = process_logits(
        context_ids,
        output.logits[:, -1, :],
        **process_logits_kwargs,
    )
    source = context_cache.source(context)
    token, logprobs = _single_ms_pfr_from_logits(
        logits,
        source=source,
        device=device,
    )
    return token, logprobs, source, output.past_key_values


@torch.no_grad()
def multi_draft_pfr_block(
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
) -> MultiDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if input_ids.shape[0] != 1:
        raise AssertionError("only batch_size=1 is supported")

    device = model.device
    input_ids = input_ids.to(device)
    if root is None:
        root = _context_key(input_ids)
    shared_labels: dict[ContextKey, bytes] = {}
    shared_sources: dict[ContextKey, SharedPFRSource] = {}
    target_context_cache = _new_context_cache(
        labeler,
        source_factory,
        device,
        label_by_key=shared_labels,
        source_by_key=shared_sources,
    )
    draft_context_cache = _new_context_cache(
        labeler,
        source_factory,
        ref_model.device,
        label_by_key=shared_labels,
        source_by_key=shared_sources,
    )
    block_len = min(lookahead, max_new_tokens)
    levels, _, draft_sets, draft_sources, draft_past_by_context = build_multi_draft_tree(
        ref_model=ref_model,
        input_ids=input_ids.to(ref_model.device),
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        context_cache=draft_context_cache,
        past_key_values=ref_past_key_values,
        max_vocab_size=model.config.vocab_size,
        process_logits_kwargs=process_logits_kwargs,
    )
    winners: dict[ContextKey, int] = {}
    logprobs_by_context: dict[ContextKey, FloatTensor] = {}
    target_sources: dict[ContextKey, SharedPFRSource] = {}

    current = root
    output_tokens: list[int] = []
    output_logprobs: list[FloatTensor] = []
    accepted = True
    accepted_count = 0
    got_eos = False
    target_past_for_next = past_key_values
    draft_past_for_next = ref_past_key_values
    last_accepted_token: int | None = None

    for _ in range(block_len):
        if current not in winners:
            token, logprobs, source, target_context_past = evaluate_target_context(
                model=model,
                context=current,
                context_cache=target_context_cache,
                past_key_values=past_key_values,
                process_logits_kwargs=process_logits_kwargs,
            )
            winners[current] = token
            logprobs_by_context[current] = logprobs
            target_sources[current] = source
        else:
            target_context_past = target_past_for_next
        token = winners[current]
        target_past_for_next = target_context_past
        draft_past_for_next = draft_past_by_context.get(current, draft_past_for_next)
        output_tokens.append(token)
        output_logprobs.append(logprobs_by_context[current])
        if token == model.config.eos_token_id:
            got_eos = True
            break

        token_was_drafted = token in draft_sets.get(current, set())
        if not token_was_drafted:
            accepted = False
            break

        accepted_count += 1
        last_accepted_token = token
        current = current + (token,)
        if len(output_tokens) >= max_new_tokens:
            break

    if accepted and not got_eos and len(output_tokens) < max_new_tokens:
        token, logprobs, bonus_source, target_past_for_next = evaluate_target_context(
            model=model,
            context=current,
            context_cache=target_context_cache,
            past_key_values=past_key_values,
            process_logits_kwargs=process_logits_kwargs,
        )
        if current in draft_past_by_context:
            draft_past_for_next = draft_past_by_context[current]
        elif last_accepted_token is not None:
            draft_past_for_next = _advance_cache(
                ref_model,
                draft_past_for_next,
                last_accepted_token,
            )
        target_sources[current] = bonus_source
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        if token == model.config.eos_token_id:
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=1)
    target_context_count = len(target_sources)
    draft_tree_size = len(draft_sources)

    return MultiDraftBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        draft_tree_size=draft_tree_size,
        target_context_count=target_context_count,
        target_past_key_values=target_past_for_next,
        draft_past_key_values=draft_past_for_next,
        got_eos=got_eos,
    )


@torch.no_grad()
def multi_draft_pfr_generator(
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
        block = multi_draft_pfr_block(
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
        )
        meta = {
            "accepted_count": block.accepted_count,
            "draft_tree_size": block.draft_tree_size,
            "target_context_count": block.target_context_count,
            "draft_len": min(lookahead, max_length - generated),
            "num_drafts": num_drafts,
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


def multi_draft_pfr_sample_generator(
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
):
    """
    Repository-style wrapper.

    Args:
        n: lookahead length ``L``.
        num_drafts: number of drafts ``B`` launched at the root and propagated
            as branch multiplicities.  ``B`` is accepted as an alias.
    """
    if B is not None:
        num_drafts = B
    yield from multi_draft_pfr_generator(
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
    )
