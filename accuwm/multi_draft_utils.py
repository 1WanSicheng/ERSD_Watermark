"""
Multi-draft utility primitives shared across PFR codepaths.

This module hosts the small, stateless building blocks that the multi-draft
speculative decoding implementations need:

  - ContextKey / MultiDraftBlock dataclasses
  - prefix-key helpers (`_context_key`, `_batch_from_keys`, `_prefix_label_from_key`)
  - per-context source + label cache (`ContextRuntimeCache`, `_new_context_cache`)
  - cache row helpers (`_repeat_cache`, `_select_cache_row`, `_advance_cache`)
  - the GPU MPFR sampling primitive (`ms_pfr_tokens_from_logprobs`)
  - the draft-tree builder (`build_multi_draft_tree`)

The actual multi-draft generators (`multi_draft_pfr_block`,
`multi_draft_pfr_sample_generator`, etc.) used to live alongside these
primitives but have been retired in favour of the cached batched-verify
variants under `MPFR_spec/`.  Keeping the primitives here lets those
variants (and the B=1 cached pipeline in `accuwm.pfr`) share a single source
of truth without dragging the obsolete sequential block-level code along.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils._pytree import tree_map

from .pfr import (
    AbstractLabeler,
    PFRSourceFactory,
    PrefixLabeler,
    SharedPFRSource,
    _safe_log,
)
from .utils import (
    cache_is_dynamic,
    cache_is_legacy,
    cache_len,
    dynamic_cache_from_legacy,
    dynamic_cache_to_legacy,
    process_logits,
)


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
    if batch_size == 1 and cache_is_legacy(past_key_values):
        # Legacy caches are immutable tuples from the model's perspective;
        # forwarding one row needs neither traversal nor materialization.
        return past_key_values

    def expand_copy(x):
        # ``repeat`` uses a relatively expensive generic tiling kernel for
        # these one-row K/V tensors.  Broadcasting the row and materializing
        # once produces the identical contiguous cache substantially faster.
        return x.expand((batch_size,) + tuple(x.shape[1:])).contiguous()

    if cache_is_legacy(past_key_values):
        return tree_map(expand_copy, past_key_values)
    if cache_is_dynamic(past_key_values):
        legacy_cache = dynamic_cache_to_legacy(past_key_values)
        repeated = tree_map(expand_copy, legacy_cache)
        return dynamic_cache_from_legacy(repeated)
    return None


def _select_cache_row(past_key_values, row: int):
    if past_key_values is None:
        return None
    if cache_is_legacy(past_key_values):
        return tree_map(lambda x: x[row : row + 1], past_key_values)
    if cache_is_dynamic(past_key_values):
        legacy_cache = dynamic_cache_to_legacy(past_key_values)
        selected = tree_map(lambda x: x[row : row + 1], legacy_cache)
        return dynamic_cache_from_legacy(selected)
    return None


def _gather_cache_rows(past_key_values, row_indices: torch.LongTensor):
    """Gather rows of a B-batched KV cache by ``row_indices``.

    Used inside the draft-tree builder to avoid re-encoding ``[cached_n:]``
    suffixes at every depth: instead of re-feeding the cross-block cache
    plus a growing suffix, we keep ``level_cache`` aligned with
    ``prev_level`` and at each depth gather rows by their parents' indices
    in the previous level.  Each subsequent forward then sees only the
    1 new token per row, mirroring InvariantMultiDraftStrategy's pattern.

    ``row_indices`` is a 1D LongTensor (length n_new_rows) on the same
    device as the cache tensors.
    """
    if past_key_values is None:
        return None
    if cache_is_legacy(past_key_values):
        return tree_map(
            lambda x: x.index_select(0, row_indices.to(x.device)), past_key_values
        )
    if cache_is_dynamic(past_key_values):
        legacy_cache = dynamic_cache_to_legacy(past_key_values)
        gathered = tree_map(
            lambda x: x.index_select(0, row_indices.to(x.device)), legacy_cache
        )
        return dynamic_cache_from_legacy(gathered)
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
    *,
    generator=None,
    support_indices: LongTensor | None = None,
) -> LongTensor:
    """Return the token ids for the first ``num_samples`` arrivals in a shared
    Poisson race.

    For each vocabulary item ``v`` with probability ``p_v``, the arrival
    process has rate ``p_v``.  The kth arrival time for that item is the
    cumulative sum of kth independent ``Exp(1)`` noises divided by ``p_v``.
    Taking the first global arrivals can therefore produce repeated token
    ids, which are later collapsed into branch multiplicities.

    ``generator`` is an optional pre-allocated ``torch.Generator`` reused by
    the multi-draft hot path; it gets re-seeded in place via the source's
    ``seed()``.  Output bytes are identical to the no-generator path because
    ``torch.rand(shape, generator=g)`` is deterministic in ``(seed, shape)``.
    """
    if num_samples <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    if logprobs.dim() == 2:
        if logprobs.shape[0] != 1:
            raise ValueError("batched MS-PFR is handled by the caller")
        logprobs = logprobs[0]
    vocab_size = logprobs.shape[-1]

    uniform_noise = source.uniform_noise(
        (num_samples, vocab_size), device=device, generator=generator,
    )
    if num_samples == 1:
        # Comparing E / p is equivalent to comparing log(p) - log(E), while
        # avoiding a vocabulary-wide exp(logprobs) and division.  Masked
        # entries remain -inf in log space.
        exp_interarrival = -_safe_log(uniform_noise)
        scores = logprobs - torch.log(exp_interarrival[0])
        return torch.argmax(scores).reshape(1).to(torch.long)

    exp_interarrival = -_safe_log(uniform_noise)
    if support_indices is not None:
        support_indices = support_indices.reshape(-1).to(device=device)
        active_noise = exp_interarrival.index_select(1, support_indices)
        active_probs = logprobs.index_select(0, support_indices).exp()
        arrival_times = torch.cumsum(active_noise, dim=0)
        arrival_times = arrival_times / active_probs.unsqueeze(0)
        active_size = int(support_indices.numel())
    else:
        probs = logprobs.exp()
        arrival_times = torch.cumsum(exp_interarrival, dim=0)
        arrival_times = arrival_times / probs.unsqueeze(0)
        active_size = vocab_size

    flat_winners = torch.topk(
        arrival_times.flatten(),
        k=num_samples,
        largest=False,
        sorted=True,
    ).indices
    active_winners = (flat_winners % active_size).to(torch.long)
    if support_indices is not None:
        return support_indices.index_select(0, active_winners)
    return active_winners


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

    # Mirror InvariantMultiDraftStrategy's incremental decode: keep
    # ``level_cache`` aligned with ``prev_level``; depth d's forward grows
    # it by 1 token to length ``root_len + d - 1``.  Previously this loop
    # fed ``batch_ids[:, cached_n:]`` of length d at depth d, doing
    # ``L(L+1)/2`` token-positions of work per block instead of ``L``.
    level_cache: any = None

    # B1: reuse one torch.Generator across all per-row noise calls.  Each
    # call re-seeds it via ``source.seed()`` so output bytes match the old
    # per-call Generator path (and detection's `_uniform_for_token`).
    shared_gen = torch.Generator(device=device)

    for depth in range(1, lookahead + 1):
        prev_level = levels[depth - 1]
        batch_ids = _batch_from_keys(prev_level, device)
        n_prev = batch_ids.shape[0]

        if depth == 1:
            if cached_n > 0:
                batch_past = _repeat_cache(past_key_values, n_prev)
                input_tokens = batch_ids[:, cached_n:]
            else:
                batch_past = None
                input_tokens = batch_ids
        else:
            prev_prev_level = levels[depth - 2]
            prev_idx_map = {ctx: i for i, ctx in enumerate(prev_prev_level)}
            parent_idx = torch.tensor(
                [prev_idx_map[ctx[:-1]] for ctx in prev_level],
                device=device,
                dtype=torch.long,
            )
            batch_past = _gather_cache_rows(level_cache, parent_idx)
            input_tokens = batch_ids[:, -1:]

        output = ref_model(
            input_tokens,
            past_key_values=batch_past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        level_cache = output.past_key_values
        logits = output.logits[:, -1, :]
        logits = process_logits(batch_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)

        next_level: list[ContextKey] = []
        seen_next: set[ContextKey] = set()

        # B3: collect per-row token tensors on GPU first, then a SINGLE
        # ``.cpu().tolist()`` per depth -- previously each row triggered its
        # own CUDA sync, which dominated wall-time at small B because the
        # target-7B forward couldn't hide that latency.
        per_row_tokens: list[torch.Tensor] = []
        per_row_mults: list[int] = []
        for row, context in enumerate(prev_level):
            past_by_context[context] = _select_cache_row(level_cache, row)
            source = context_cache.source(context)
            sources[context] = source
            mult = multiplicities[context]
            per_row_mults.append(mult)
            per_row_tokens.append(ms_pfr_tokens_from_logprobs(
                logprobs[row],
                source=source,
                num_samples=mult,
                device=device,
                generator=shared_gen,
            ))

        flat = (
            torch.cat(per_row_tokens).cpu().tolist()
            if per_row_tokens
            else []
        )
        offset = 0
        for row, context in enumerate(prev_level):
            mult = per_row_mults[row]
            row_tokens = flat[offset:offset + mult]
            offset += mult
            token_counts: dict[int, int] = {}
            for token in row_tokens:
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
