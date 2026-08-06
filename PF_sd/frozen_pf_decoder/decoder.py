"""Tree-free list implementation of independent Latin-PF speculative decoding.

The coupling is the frozen independent Latin-PF coupling. The optimized
representation of the B draft trajectories keeps a dense
``[B, L]`` token matrix and verifies it with a surviving-row mask instead of
building and traversing a Python prefix trie.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import FloatTensor, LongTensor

from accuwm.pfr import PFRSourceFactory
from accuwm.utils import cache_len
from MPFR_spec.mpfr_batched_torchgen_cached import _select_and_truncate_cache
from MPFR_spec.mpfr_direct_optimized import ContextKey, _context_key, _model_device
from PF_sd.frozen_pf_decoder.core import max_order_pf as M


@dataclass
class TreeFreeLatinPFBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    output_labels: List[bytes]
    output_pivots: FloatTensor
    accepted_count: int
    attempted_draft_tokens: int
    draft_tree_size: int
    target_context_count: int
    target_forward_calls: int
    draft_forward_calls: int
    got_eos: bool
    target_past_key_values: Any
    draft_past_key_values: Any


def _append_context(context: ContextKey, token: int) -> ContextKey:
    return context + (int(token),)


def _first_alive(alive: List[bool]) -> int:
    for row, keep in enumerate(alive):
        if keep:
            return row
    raise RuntimeError("tree-free verifier has no surviving draft row")


def _stable_unique_context_rows(
    contexts: List[ContextKey],
) -> Tuple[List[int], List[int]]:
    """Return first-occurrence representatives and row-to-compact mapping."""
    compact_by_context: Dict[ContextKey, int] = {}
    representatives: List[int] = []
    row_to_compact: List[int] = []
    for row, context in enumerate(contexts):
        compact = compact_by_context.get(context)
        if compact is None:
            compact = len(representatives)
            compact_by_context[context] = compact
            representatives.append(row)
        row_to_compact.append(compact)
    return representatives, row_to_compact


def verify_draft_matrix(
    draft_tokens: List[List[int]], target_tokens: List[int]
) -> Tuple[int, int]:
    """Pure list-level verifier used by tests.

    Returns ``(accepted_count, representative_row_before_rejection_or_bonus)``.
    ``target_tokens`` may contain the L draft-position winners plus a bonus.
    """
    if not draft_tokens:
        raise ValueError("at least one draft row is required")
    width = len(draft_tokens)
    lookahead = len(draft_tokens[0])
    if any(len(row) != lookahead for row in draft_tokens):
        raise ValueError("draft rows must have equal length")
    alive = [True] * width
    representative = 0
    accepted = 0
    for depth in range(min(lookahead, len(target_tokens))):
        representative = _first_alive(alive)
        winner = int(target_tokens[depth])
        matched = [
            keep and int(draft_tokens[row][depth]) == winner
            for row, keep in enumerate(alive)
        ]
        if not any(matched):
            return accepted, representative
        alive = matched
        accepted += 1
    return accepted, _first_alive(alive)


@torch.no_grad()
def _generate_latin_draft_list(
    *,
    ref_model,
    input_ids: LongTensor,
    root: ContextKey,
    lookahead: int,
    width: int,
    source_factory: PFRSourceFactory,
    process_logits_kwargs: Dict[str, Any],
    max_vocab_size: int,
    past_key_values: Any,
) -> Tuple[
    LongTensor,
    List[List[ContextKey]],
    Any,
    List[int],
    int,
    int,
    Dict[ContextKey, bytes],
    Dict[ContextKey, int],
]:
    """Generate B trajectories while forwarding each unique prefix once.

    The logical state remains a flat list of B trajectories.  At each depth a
    stable, temporary compaction map avoids duplicate draft-model rows; no
    persistent prefix tree or child adjacency structure is constructed.
    """
    device = _model_device(ref_model)
    top_k = M._top_k(process_logits_kwargs)
    if not (0 < int(top_k) < int(max_vocab_size)):
        raise ValueError("tree-free counter backend currently requires 0 < top_k < vocab")
    temperature = M._temperature_for_draft(process_logits_kwargs)
    contexts: List[ContextKey] = [root for _ in range(int(width))]
    contexts_by_depth: List[List[ContextKey]] = [list(contexts)]
    level_cache = None
    previous_row_to_compact = [0 for _ in range(int(width))]
    final_row_to_cache = [0 for _ in range(int(width))]
    generated_columns: List[LongTensor] = []
    forwarded_contexts = 0
    labels_by_context: Dict[ContextKey, bytes] = {}
    seeds_by_context: Dict[ContextKey, int] = {}

    for depth in range(int(lookahead)):
        representatives, row_to_compact = _stable_unique_context_rows(contexts)
        compact_contexts = [contexts[row] for row in representatives]
        if depth == 0:
            batch_cache = past_key_values
            cached_n = cache_len(batch_cache)
            model_input = input_ids.to(device)[:, cached_n:]
        else:
            parent_rows = [previous_row_to_compact[row] for row in representatives]
            if parent_rows == list(range(len(parent_rows))):
                batch_cache = level_cache
            else:
                parent_idx = torch.tensor(
                    parent_rows, device=device, dtype=torch.long
                )
                batch_cache = M._gather_cache_rows(level_cache, parent_idx)
            model_input = torch.tensor(
                [[int(context[-1])] for context in compact_contexts],
                device=device,
                dtype=torch.long,
            )
        out = M._model_forward(
            ref_model,
            num_logits_to_keep=1,
            input_ids=model_input,
            past_key_values=batch_cache,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        level_cache = out.past_key_values
        forwarded_contexts += len(compact_contexts)
        raw_logits = out.logits[:, -1, :max_vocab_size]
        processed, supports = M._compact_topk_logits(
            raw_logits, top_k=int(top_k), temperature=temperature
        )
        processed_f = processed.float()
        processed_weights = torch.exp(
            processed_f - processed_f.amax(dim=-1, keepdim=True)
        )
        compact_seeds: List[int] = []
        for context in compact_contexts:
            label = labels_by_context.get(context)
            if label is None:
                label = M.max_order_context_label(context, int(width))
                labels_by_context[context] = label
            seed = seeds_by_context.get(context)
            if seed is None:
                seed = M._counter_context_seed(source_factory, label)
                seeds_by_context[context] = seed
            compact_seeds.append(seed)
        seeds = [compact_seeds[row] for row in row_to_compact]
        next_tokens = M.counter_latin_draft_select_batch_support(
            processed_logits=processed,
            supports=supports,
            context_rows=row_to_compact,
            fields=list(range(int(width))),
            context_seeds=seeds,
            width=int(width),
            vocab_size=int(max_vocab_size),
            processed_weights=processed_weights,
        )
        # One dependency synchronization per draft depth remains necessary for
        # the exact byte-level context labels used by the existing detector.
        token_values = [int(token) for token in next_tokens.detach().cpu().tolist()]
        contexts = [
            _append_context(context, token)
            for context, token in zip(contexts, token_values)
        ]
        contexts_by_depth.append(list(contexts))
        generated_columns.append(next_tokens)
        previous_row_to_compact = row_to_compact
        final_row_to_cache = row_to_compact

    draft_tokens = torch.stack(generated_columns, dim=1)
    return (
        draft_tokens,
        contexts_by_depth,
        level_cache,
        final_row_to_cache,
        int(lookahead),
        forwarded_contexts,
        labels_by_context,
        seeds_by_context,
    )


@torch.no_grad()
def speculative_tree_free_latin_pf_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    width: int,
    source_factory: PFRSourceFactory,
    max_new_tokens: int,
    target_past_key_values: Any = None,
    ref_past_key_values: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_logprobs: bool = False,
    record_pivots: bool = False,
    root_context: Optional[ContextKey] = None,
) -> TreeFreeLatinPFBlock:
    """Generate and verify one tree-free independent Latin-PF block."""
    if lookahead <= 0 or width <= 0:
        raise ValueError("lookahead and width must be positive")
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = root_context if root_context is not None else _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    kwargs = process_logits_kwargs or {}
    max_vocab_size = min(
        int(model.config.vocab_size), int(ref_model.config.vocab_size)
    )
    top_k = M._top_k(kwargs)
    if not (0 < int(top_k) < int(max_vocab_size)):
        raise ValueError("tree-free counter backend currently requires 0 < top_k < vocab")

    (
        draft_tokens,
        contexts_by_depth,
        draft_batch_cache,
        draft_row_to_cache,
        draft_forward_calls,
        draft_context_count,
        labels_by_context,
        seeds_by_context,
    ) = (
        _generate_latin_draft_list(
            ref_model=ref_model,
            input_ids=input_ids,
            root=root,
            lookahead=block_len,
            width=int(width),
            source_factory=source_factory,
            process_logits_kwargs=kwargs,
            max_vocab_size=max_vocab_size,
            past_key_values=ref_past_key_values,
        )
    )
    # Stable leaf compaction: duplicate complete trajectories do not need a
    # second target-model row.  This is a flat list operation, not a prefix
    # tree, and preserves the first-occurrence order used by the reference
    # backend.
    draft_cpu = [
        [int(token) for token in row]
        for row in draft_tokens.detach().cpu().tolist()
    ]
    leaf_row_by_tokens: Dict[Tuple[int, ...], int] = {}
    unique_leaf_rows: List[int] = []
    full_to_compact_leaf: List[int] = []
    for row, tokens in enumerate(draft_cpu):
        key = tuple(tokens)
        compact_row = leaf_row_by_tokens.get(key)
        if compact_row is None:
            compact_row = len(unique_leaf_rows)
            leaf_row_by_tokens[key] = compact_row
            unique_leaf_rows.append(row)
        full_to_compact_leaf.append(compact_row)
    leaf_idx = torch.tensor(unique_leaf_rows, device=device, dtype=torch.long)
    unique_draft_tokens = draft_tokens.index_select(0, leaf_idx)
    draft_sequences = torch.cat(
        (input_ids.repeat(len(unique_leaf_rows), 1), unique_draft_tokens), dim=1
    )

    cached_n = cache_len(target_past_key_values)
    target_input = draft_sequences[:, cached_n:] if cached_n > 0 else draft_sequences
    repeated_target_cache = (
        M._repeat_cache(target_past_key_values, len(unique_leaf_rows))
        if target_past_key_values is not None
        else None
    )
    target_out = M._model_forward(
        model,
        num_logits_to_keep=block_len + 1,
        input_ids=target_input,
        past_key_values=repeated_target_cache,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    target_cache_all = target_out.past_key_values
    logits = target_out.logits[:, -(block_len + 1):, :max_vocab_size]

    # Compact repeated prefixes after the single target forward.  Each unique
    # context is processed by top-k/PF exactly once and looked up by its token
    # tuple during alive-mask verification.
    context_positions: List[Tuple[ContextKey, int, int]] = []
    context_to_row: Dict[ContextKey, int] = {}
    for compact_leaf_row, full_row in enumerate(unique_leaf_rows):
        for depth in range(block_len + 1):
            context = contexts_by_depth[depth][full_row]
            if context in context_to_row:
                continue
            context_to_row[context] = len(context_positions)
            context_positions.append((context, compact_leaf_row, depth))
    context_leaf_rows = torch.tensor(
        [item[1] for item in context_positions], device=device, dtype=torch.long
    )
    context_depths = torch.tensor(
        [item[2] for item in context_positions], device=device, dtype=torch.long
    )
    unique_logits = logits[context_leaf_rows, context_depths, :]
    processed, supports = M._compact_topk_logits(
        unique_logits,
        top_k=int(top_k),
        temperature=M._temperature_for_target(kwargs),
    )

    eos_tokens = M._as_eos_set(getattr(model.config, "eos_token_id", None))
    alive = [True] * int(width)
    current = root
    cache_row = 0
    accepted = 0
    output_tokens: List[int] = []
    output_labels: List[bytes] = []
    output_pivots: List[FloatTensor] = []
    output_processed: List[FloatTensor] = []
    got_eos = False

    for depth in range(block_len + 1):
        cache_row = _first_alive(alive)
        context = contexts_by_depth[depth][cache_row]
        if context != current:
            raise RuntimeError("surviving list row is not aligned with target context")
        target_row = context_to_row[context]
        context_label = labels_by_context.get(context)
        if context_label is None:
            context_label = M.max_order_context_label(context, int(width))
            labels_by_context[context] = context_label
        context_seed = seeds_by_context.get(context)
        if context_seed is None:
            context_seed = M._counter_context_seed(source_factory, context_label)
            seeds_by_context[context] = context_seed
        token, pivot = M.counter_latin_target_select_support(
            processed_logits=processed[target_row],
            support=supports[target_row],
            width=int(width),
            source_factory=source_factory,
            context_label=context_label,
            exact_pivot=record_pivots,
            compact_logits=True,
            vocab_size=max_vocab_size,
            context_seed=context_seed,
        )
        output_tokens.append(int(token))
        output_labels.append(context_label)
        if record_pivots:
            output_pivots.append(pivot)
        if return_logprobs:
            output_processed.append(processed[target_row])
        if int(token) in eos_tokens:
            got_eos = True
            break
        if depth == block_len:
            break
        matched = [
            keep and draft_cpu[row][depth] == int(token)
            for row, keep in enumerate(alive)
        ]
        if not any(matched):
            break
        alive = matched
        # Follow a surviving trajectory before the block can terminate.  The
        # selected KV row must correspond to the accepted prefix even when the
        # token budget is exhausted immediately after this acceptance.
        cache_row = _first_alive(alive)
        accepted += 1
        current = _append_context(current, int(token))
        if len(output_tokens) >= int(max_new_tokens):
            break

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    if return_logprobs:
        output_logprobs = torch.nn.functional.log_softmax(
            torch.stack(output_processed).float(), dim=-1
        ).unsqueeze(0)
    else:
        output_logprobs = torch.empty(
            (1, len(output_tokens), 0), device=device, dtype=torch.float32
        )
    output_pivots_tensor = (
        torch.stack(output_pivots).float()
        if output_pivots
        else torch.empty(0, device=device, dtype=torch.float32)
    )
    new_cache_len = root_len + accepted
    target_cache = _select_and_truncate_cache(
        target_cache_all,
        int(full_to_compact_leaf[cache_row]),
        int(new_cache_len),
        make_contiguous=False,
    )
    draft_cache = _select_and_truncate_cache(
        draft_batch_cache,
        int(draft_row_to_cache[cache_row]),
        int(new_cache_len),
        make_contiguous=False,
    )
    return TreeFreeLatinPFBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        output_labels=output_labels,
        output_pivots=output_pivots_tensor,
        accepted_count=accepted,
        attempted_draft_tokens=int(width) * block_len,
        draft_tree_size=draft_context_count,
        target_context_count=len(context_positions),
        target_forward_calls=1,
        draft_forward_calls=draft_forward_calls,
        got_eos=got_eos,
        target_past_key_values=target_cache,
        draft_past_key_values=draft_cache,
    )


@torch.no_grad()
def speculative_tree_free_latin_pf_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    width: int,
    max_length: int,
    private_key: bytes | str = b"1234",
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    return_logprobs: bool = False,
    record_pivots: bool = False,
) -> Iterable:
    """Yield tree-free Latin-PF speculative blocks."""
    model.eval()
    ref_model.eval()
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")
    source_factory = PFRSourceFactory(private_key=bytes(private_key))
    device = _model_device(model)
    input_ids = input_ids.to(device)
    target_cache = None
    draft_cache = None
    generated = 0
    root_context = _context_key(input_ids)

    while generated < int(max_length):
        block = speculative_tree_free_latin_pf_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            lookahead=lookahead,
            width=width,
            source_factory=source_factory,
            max_new_tokens=int(max_length) - generated,
            target_past_key_values=target_cache,
            ref_past_key_values=draft_cache,
            process_logits_kwargs=process_logits_kwargs,
            return_logprobs=return_logprobs,
            record_pivots=record_pivots,
            root_context=root_context,
        )
        meta = {
            "accepted_count": block.accepted_count,
            "attempted_draft_tokens": block.attempted_draft_tokens,
            "draft_tree_size": block.draft_tree_size,
            "target_context_count": block.target_context_count,
            "target_forward_calls": block.target_forward_calls,
            "draft_forward_calls": block.draft_forward_calls,
            "width": int(width),
            "proposal": "latin_hypercube_pf_tree_free",
            "source_labels": block.output_labels,
            "aggregate_pivots": (
                block.output_pivots if block.output_pivots.numel() else None
            ),
            "rng_backend": "counter_philox",
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat((input_ids, block.output_ids.to(device)), dim=1)
        emitted = [int(token) for token in block.output_ids[0].detach().cpu().tolist()]
        root_context = root_context + tuple(emitted)
        target_cache = block.target_past_key_values
        draft_cache = block.draft_past_key_values
        generated += len(emitted)
        if block.got_eos:
            break
