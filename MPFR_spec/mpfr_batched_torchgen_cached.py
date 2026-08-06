"""
Cache-aware variant of mpfr_batched_torchgen.

Same algorithm as mpfr_batched_torchgen.py (MPFR_spec direct-finite MPFR with
GPU torch.Generator sampling, draft tree + ONE batched target forward over
depth-L leaves) but threads the TARGET KV cache across blocks AND within the
batched verify forward.  The draft KV cache is also threaded across depths and
blocks.

Cache convention:
- At the start of block, `target_past_key_values` covers [0, len(input_ids)-1).
- We construct and feed only each leaf suffix after ``cached_n`` to the target.
- After forward, cache covers [0, root_len + L) per batch row.
- We select the alive row (the leaf row containing the realized path) and
  truncate to length (root_len + accepted_count), exposing the next block's
  initial cache covering [0, new_input_len - 1).
"""
from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MPFR_spec"))

from mpfr_direct_optimized import (
    ContextKey,
    DraftTree,
    MPFRBlock,
    _batch_from_contexts,
    _context_key,
    _model_device,
    _normalize_logprobs_from_processed_logits,
    _temperature_for_draft,
    _temperature_for_target,
    _top_k,
    _top_p,
    process_logits_exact,
)

from accuwm.multi_draft_utils import (
    _gather_cache_rows,
    _repeat_cache,
    _select_cache_row,
    ms_pfr_tokens_from_logprobs,
)
from accuwm.pfr import PFRSourceFactory, SharedPFRSource
from accuwm.utils import cache_len


@lru_cache(maxsize=None)
def _supports_num_logits_to_keep(model_type: type) -> bool:
    return "num_logits_to_keep" in inspect.signature(model_type.forward).parameters


def _model_forward(model, *, num_logits_to_keep: int, **kwargs):
    """Avoid materializing unused vocabulary logits when the model supports it."""
    if _supports_num_logits_to_keep(type(model)):
        kwargs["num_logits_to_keep"] = int(num_logits_to_keep)
    return model(**kwargs)


def _context_label(context: ContextKey) -> bytes:
    """Per-context bytes label fed to SharedPFRSource.  Mirrors the
    per-context blake2b seed in the original mpfr_direct_optimized."""
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in context
    )


def _make_source_cache(factory: PFRSourceFactory):
    """Memoise (label bytes, SharedPFRSource) per generation context.
    Same context can appear at multiple depths within a tree; the cache lets us
    pay the prefix byte-serialisation + sha256 cost once per unique context.
    Output bytes downstream are unchanged because ``factory.build(label).seed()``
    is a deterministic function of (label, private_key)."""
    cache: Dict[ContextKey, SharedPFRSource] = {}

    def get(context: ContextKey) -> SharedPFRSource:
        src = cache.get(context)
        if src is None:
            parent = cache.get(context[:-1]) if context else None
            if parent is not None:
                # Context labels are prefix bytes followed by one int64 token,
                # so children can reuse the already serialized parent label.
                label = parent.label + int(context[-1]).to_bytes(
                    8, "big", signed=True
                )
            else:
                label = _context_label(context)
            src = factory.build(label)
            cache[context] = src
        return src

    return get


@torch.no_grad()
def build_draft_tree_torchgen(
    *,
    ref_model,
    root: ContextKey,
    lookahead: int,
    num_drafts: int,
    source_factory: PFRSourceFactory,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
    past_key_values: Any = None,
    source_for_context: Any = None,
) -> Tuple[DraftTree, List[Any]]:
    """Build the multi-draft tree using the GPU torch.Generator MPFR primitive.
    Identical to mpfr_direct_optimized.build_draft_tree_direct except the
    sampler is ms_pfr_tokens_from_logprobs (GPU) instead of the CPU/blake2b
    direct_finite primitive.

    If ``past_key_values`` is provided (covering [0, cached_n)), only the
    suffix ``batch_ids[:, cached_n:]`` is fed to the draft model at each depth
    and the draft KV cache is threaded across depths.  The returned cache list
    is aligned with tree levels 0..lookahead-1; the caller selects the single
    realized row for cross-block reuse after target verification.
    """
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
    # One batched cache per expanded level.  The old path wrapped every row in
    # a separate DynamicCache even though only the realized path survives the
    # block.  Retaining level-aligned caches lets the caller select exactly one
    # row after verification.
    level_caches: List[Any] = []
    draft_tree_size = 0
    draft_forward_calls = 0
    cached_n = cache_len(past_key_values)

    # B1+B2: reuse one torch.Generator across all per-row noise calls (saves
    # the per-call Generator alloc) and memoise SharedPFRSource per context
    # (saves redundant prefix byte-serialisation + sha256 hashing when the
    # same context appears at multiple depths).
    source_for = source_for_context or _make_source_cache(source_factory)
    shared_gen = torch.Generator(device=draft_device)

    # ``level_cache`` is the B-batched draft KV cache aligned with the current
    # ``prev_level``; depth d's forward grows it by 1 position to length
    # ``root_len + d - 1``.  This avoids re-encoding the ``[cached_n:]``
    # suffix at every depth (the previous implementation fed
    # ``batch_ids[:, cached_n:]`` of length d at depth d, doing
    # ``L(L+1)/2`` token-positions of work per block instead of ``L``).
    # Mirrors InvariantMultiDraftStrategy's incremental decode pattern.
    level_cache: Any = None  # set after depth-1 forward, reused at depth>=2

    for depth in range(1, lookahead + 1):
        prev_level = levels[depth - 1]
        if not prev_level:
            levels.append([])
            break

        n_prev = len(prev_level)

        if depth == 1:
            # First depth: prev_level == [root]; need to encode the part of
            # the prompt not yet in the cross-block cache.
            if cached_n > 0:
                batch_past = _repeat_cache(past_key_values, n_prev)
                input_tokens = _batch_from_contexts(
                    [context[cached_n:] for context in prev_level],
                    draft_device,
                )
            else:
                batch_past = None
                input_tokens = _batch_from_contexts(prev_level, draft_device)
        else:
            # Depth >= 2: gather parent rows from the previous depth's
            # batched cache (length root_len + depth - 2) and feed only the
            # 1 new token per row.
            prev_prev_level = levels[depth - 2]
            prev_idx_map = {ctx: i for i, ctx in enumerate(prev_prev_level)}
            parent_rows = [prev_idx_map[ctx[:-1]] for ctx in prev_level]
            if (
                len(parent_rows) == len(prev_prev_level)
                and all(row == index for index, row in enumerate(parent_rows))
            ):
                # One child per parent in unchanged order: the cache is
                # already aligned, so index_select would only copy it.
                batch_past = level_cache
            else:
                parent_idx = torch.tensor(
                    parent_rows, device=draft_device, dtype=torch.long
                )
                batch_past = _gather_cache_rows(level_cache, parent_idx)
            input_tokens = _batch_from_contexts(
                [context[-1:] for context in prev_level], draft_device
            )

        out = _model_forward(
            ref_model,
            num_logits_to_keep=1,
            input_ids=input_tokens,
            past_key_values=batch_past,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        draft_forward_calls += 1
        draft_tree_size += len(prev_level)
        level_cache = out.past_key_values
        level_caches.append(level_cache)

        logits = out.logits[:, -1, :]
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        processed, support_indices = process_logits_exact(
            logits, temperature=draft_temp, top_k=top_k, top_p=top_p,
            return_support=True,
        )
        logprobs = _normalize_logprobs_from_processed_logits(processed)

        next_level: List[ContextKey] = []
        seen_next: set[ContextKey] = set()

        # B1+B3: collect per-row token tensors on GPU, then sync ONCE per
        # depth.  Previous code did `.cpu().tolist()` per parent context,
        # which is a CUDA sync per row; on Qwen B=2/L=4 that's 30 syncs per
        # block.  We also reuse ``shared_gen`` so each ``ms_pfr_tokens_from_
        # logprobs`` call skips the per-call Generator allocation.
        per_row_tokens: List[torch.Tensor] = []
        per_row_mults: List[int] = []
        for row, context in enumerate(prev_level):
            mult = multiplicities[context]
            per_row_mults.append(mult)
            source = source_for(context)
            per_row_tokens.append(ms_pfr_tokens_from_logprobs(
                logprobs[row],
                source=source,
                num_samples=mult,
                device=draft_device,
                generator=shared_gen,
                support_indices=(support_indices[row] if mult > 1 else None),
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
            token_counts: Dict[int, int] = {}
            for t in row_tokens:
                token_counts[int(t)] = token_counts.get(int(t), 0) + 1

            draft_sets[context] = set(token_counts.keys())
            for token, count in token_counts.items():
                child = context + (int(token),)
                multiplicities[child] = count
                if child not in seen_next:
                    next_level.append(child)
                    seen_next.add(child)

        levels.append(next_level)

    tree = DraftTree(
        levels=levels,
        multiplicities=multiplicities,
        draft_sets=draft_sets,
        draft_tree_size=draft_tree_size,
        draft_forward_calls=draft_forward_calls,
    )
    return tree, level_caches


@dataclass
class MPFRCachedBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    output_tokens: Tuple[int, ...]
    accepted_count: int
    attempted_draft_tokens: int
    draft_tree_size: int
    target_context_count: int
    target_forward_calls: int
    draft_forward_calls: int
    got_eos: bool
    target_past_key_values: Any
    draft_past_key_values: Any = None


@dataclass
class TargetLogitStore:
    """Raw target logits plus the row/position for each verified context.

    Target verification materializes the tree logits in one model forward,
    but generation follows only one realized path.  Keeping raw logits here
    lets us defer top-k and log-softmax until a context is actually visited,
    instead of processing every context in the proposal tree.
    """

    logits: FloatTensor
    position_by_context: Dict[ContextKey, Tuple[int, int]]
    temperature: Any
    top_k: int
    top_p: float

    def logprobs(self, context: ContextKey) -> FloatTensor:
        row, pos = self.position_by_context[context]
        processed = process_logits_exact(
            self.logits[row, pos, :],
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return _normalize_logprobs_from_processed_logits(processed)


def _select_and_truncate_cache(
    cache: Any,
    batch_index: int,
    seq_len: int,
    *,
    make_contiguous: bool = True,
) -> Any:
    if cache is None:
        return None
    seq_len = max(int(seq_len), 0)

    def finish(tensor):
        view = tensor[batch_index, :, :seq_len, :].unsqueeze(0)
        return view.contiguous() if make_contiguous else view

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = finish(cache.key_cache[i])
            cache.value_cache[i] = finish(cache.value_cache[i])
        return cache
    selected = []
    for layer in cache:
        k, v = layer[:2]
        selected.append((
            finish(k),
            finish(v),
        ) + tuple(layer[2:]))
    return tuple(selected)


@torch.no_grad()
def _target_logprobs_from_leaves_one_forward_cached(
    *,
    model,
    leaves: List[ContextKey],
    root_len: int,
    lookahead: int,
    target_past_key_values: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> Tuple[TargetLogitStore, Any]:
    if not leaves:
        raise ValueError("cannot batch target over an empty leaf set")
    device = _model_device(model)
    n_leaves = len(leaves)

    if target_past_key_values is None:
        cached_n = 0
        target_past_repeated = None
    else:
        cached_n = cache_len(target_past_key_values)
        if cached_n > root_len - 1:
            raise RuntimeError(
                f"target cache too long (cached_n={cached_n}, root_len={root_len}); "
                "cache should cover at most [0, root_len - 1)"
            )
        target_past_repeated = _repeat_cache(
            target_past_key_values, n_leaves
        )

    new_input_ids = _batch_from_contexts(
        [leaf[cached_n:] for leaf in leaves], device
    )

    full_input_length = int(new_input_ids.shape[1])
    out = _model_forward(
        model,
        num_logits_to_keep=lookahead + 1,
        input_ids=new_input_ids,
        past_key_values=target_past_repeated,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    new_target_cache = out.past_key_values
    logits_all = out.logits  # (B, full_seq_len - cached_n, V)
    kept_start = full_input_length - int(logits_all.shape[1])
    if max_vocab_size is not None and max_vocab_size < logits_all.shape[-1]:
        logits_all = logits_all[..., :max_vocab_size]

    target_temp = _temperature_for_target(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)

    position_by_context: Dict[ContextKey, Tuple[int, int]] = {}
    for row, leaf in enumerate(leaves):
        for d in range(0, lookahead + 1):
            context = leaf[: root_len + d]
            if context in position_by_context:
                continue
            pos = root_len + d - 1 - cached_n - kept_start
            if pos < 0:
                raise RuntimeError(
                    f"output position {pos} is negative; cached_n={cached_n}, "
                    f"root_len={root_len}, d={d}"
                )
            position_by_context[context] = (row, pos)

    return TargetLogitStore(
        logits=logits_all,
        position_by_context=position_by_context,
        temperature=target_temp,
        top_k=top_k,
        top_p=top_p,
    ), new_target_cache


@torch.no_grad()
def mpfr_batched_torchgen_cached_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    source_factory: PFRSourceFactory,
    max_new_tokens: int,
    target_past_key_values: Any = None,
    ref_past_key_values: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    root_context: Optional[ContextKey] = None,
    return_logprobs: bool = True,
    source_for_context: Any = None,
) -> MPFRCachedBlock:
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = root_context if root_context is not None else _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    source_for = source_for_context or _make_source_cache(source_factory)
    draft_tree, draft_level_caches = build_draft_tree_torchgen(
        ref_model=ref_model,
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        source_factory=source_factory,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
        past_key_values=ref_past_key_values,
        source_for_context=source_for,
    )

    leaves = draft_tree.levels[block_len]
    if not leaves:
        raise RuntimeError("draft tree did not reach lookahead depth")

    target_logits, new_target_cache_repeated = (
        _target_logprobs_from_leaves_one_forward_cached(
            model=model,
            leaves=leaves,
            root_len=root_len,
            lookahead=block_len,
            target_past_key_values=target_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            max_vocab_size=max_vocab_size,
        )
    )

    current = root
    output_tokens: List[int] = []
    output_logprobs: List[FloatTensor] = []
    accepted_count = 0
    got_eos = False
    accepted_all = True
    target_generator = torch.Generator(device=device)

    for _ in range(block_len):
        logprobs = target_logits.logprobs(current)
        source = source_for(current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
            generator=target_generator,
        )[0].item())

        output_tokens.append(token)
        if return_logprobs:
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
        logprobs = target_logits.logprobs(current)
        source = source_for(current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
            generator=target_generator,
        )[0].item())
        output_tokens.append(token)
        if return_logprobs:
            output_logprobs.append(logprobs)
        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    if return_logprobs:
        output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)
    else:
        # Generation and watermark detection use token ids plus the keyed
        # context labels; retaining full-vocabulary log-probability rows is
        # optional.  An explicit fast path avoids a multi-MiB stack/copy per
        # block while preserving the default API for callers that need them.
        output_logprobs_tensor = torch.empty(
            (1, 0, int(max_vocab_size or 0)), device=device
        )

    alive_row_idx = 0
    cur_len = len(current)
    for row_idx, leaf in enumerate(leaves):
        if leaf[:cur_len] == current:
            alive_row_idx = row_idx
            break

    new_cache_len = root_len + accepted_count
    truncated_target_cache = _select_and_truncate_cache(
        new_target_cache_repeated, alive_row_idx, new_cache_len
    )

    # Pick the realized row only once.  Caches exist for contexts at depths
    # 0..block_len-1; a fully accepted depth-block_len leaf therefore reuses
    # its parent cache and feeds the final accepted token in the next block.
    cache_depth = min(accepted_count, block_len - 1)
    cache_context = current[: root_len + cache_depth]
    cache_row = draft_tree.levels[cache_depth].index(cache_context)
    draft_past_for_next = _select_cache_row(
        draft_level_caches[cache_depth], cache_row
    )

    return MPFRCachedBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        output_tokens=tuple(output_tokens),
        accepted_count=accepted_count,
        attempted_draft_tokens=block_len,
        draft_tree_size=draft_tree.draft_tree_size,
        target_context_count=len(target_logits.position_by_context),
        target_forward_calls=1,
        draft_forward_calls=draft_tree.draft_forward_calls,
        got_eos=got_eos,
        target_past_key_values=truncated_target_cache,
        draft_past_key_values=draft_past_for_next,
    )


@torch.no_grad()
def finite_multi_draft_pfr_cached_generator(
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
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "batched_target_torchgen_cached",
    return_logprobs: bool = True,
) -> Iterable[Tuple[LongTensor, FloatTensor, Dict[str, Any]]]:
    del labeler, max_proposals, allow_incomplete, proposal

    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    model.eval()
    ref_model.eval()
    source_factory = PFRSourceFactory(
        private_key=(private_key.encode("utf-8") if isinstance(private_key, str) else bytes(private_key))
    )
    device = _model_device(model)
    input_ids = input_ids.to(device)
    target_past_key_values = None
    ref_past_key_values = None
    current_context = _context_key(input_ids)
    # Keep the immutable context -> keyed-source mapping across blocks.  The
    # realized next root was already visited in the preceding block, and its
    # descendants can extend the cached label bytes incrementally.
    source_for_context = _make_source_cache(source_factory)
    generated = 0

    while generated < max_length:
        remaining = int(max_length - generated)
        if remaining <= 0:
            break
        block = mpfr_batched_torchgen_cached_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            lookahead=lookahead,
            num_drafts=num_drafts,
            source_factory=source_factory,
            max_new_tokens=remaining,
            target_past_key_values=target_past_key_values,
            ref_past_key_values=ref_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            root_context=current_context,
            return_logprobs=return_logprobs,
            source_for_context=source_for_context,
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
            "proposal": "batched_target_torchgen_cached",
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        # These Python token ids already exist because verification branches
        # on each sampled id.  Reuse them instead of copying the newly-created
        # output tensor GPU -> CPU once more on every block.
        new_tokens = block.output_tokens
        current_context = current_context + new_tokens
        # ``current_context`` and the threaded caches fully specify subsequent
        # blocks, so rebuilding a growing dense input tensor is unnecessary.
        target_past_key_values = block.target_past_key_values
        ref_past_key_values = block.draft_past_key_values
        generated += int(block.output_ids.shape[-1])
        if block.got_eos:
            break


def finite_multi_draft_pfr_cached_sample_generator(
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
    proposal: str = "batched_target_torchgen_cached",
    return_logprobs: bool = True,
):
    if B is not None:
        num_drafts = int(B)
    yield from finite_multi_draft_pfr_cached_generator(
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
        return_logprobs=return_logprobs,
    )
