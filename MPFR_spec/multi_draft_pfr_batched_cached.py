"""
Cache-aware batched-verify variant of accuwm.multi_draft_pfr.

Same algorithm as multi_draft_pfr_batched.py (shared-Poisson MPFR via
torch.Generator + ONE batched target forward over depth-L draft leaves) but
threads the TARGET KV cache across blocks AND within the batched verify
forward.  Draft side remains cache-free for simplicity (draft model is small
0.5B; the dominant cost is the 7B target).

Cache convention (matches SpeculativeDecoding.StrongMultiDraftStrategy):
- At the start of block, `target_past_key_values` covers [0, len(input_ids)-1).
- We feed batch_ids[:, cached_n:] of shape (B, root_len + L - cached_n) to
  the target -- only the new tokens since the cache was last filled.
- After forward, cache covers [0, root_len + L) per batch row.
- We select the alive row (the leaf row containing the realized path) and
  truncate to length (root_len + accepted_count), exposing the next block's
  initial cache covering [0, new_input_len - 1).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accuwm.multi_draft_utils import (
    ContextKey,
    MultiDraftBlock,
    _context_key,
    _new_context_cache,
    _repeat_cache,
    build_multi_draft_tree,
    ms_pfr_tokens_from_logprobs,
)
from accuwm.pfr import (
    AbstractLabeler,
    PFRSourceFactory,
    PrefixLabeler,
    SharedPFRSource,
)
from accuwm.utils import cache_len, process_logits


def _batch_from_contexts(contexts: List[ContextKey], device: torch.device) -> LongTensor:
    if not contexts:
        raise ValueError("empty context batch")
    lengths = {len(c) for c in contexts}
    if len(lengths) != 1:
        raise ValueError("contexts must have same length for no-padding batch")
    return torch.tensor([list(c) for c in contexts], device=device, dtype=torch.long)


def _select_and_truncate_cache(cache: Any, batch_index: int, seq_len: int) -> Any:
    """Select one row from a B-repeated cache and truncate to seq_len, in
    place where possible.  Returns the same cache object (or a new tuple for
    legacy caches)."""
    if cache is None:
        return None
    seq_len = max(int(seq_len), 0)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = (
                cache.key_cache[i][batch_index, :, :seq_len, :].unsqueeze(0).contiguous()
            )
            cache.value_cache[i] = (
                cache.value_cache[i][batch_index, :, :seq_len, :].unsqueeze(0).contiguous()
            )
        return cache
    selected = []
    for layer in cache:
        k, v = layer[:2]
        selected.append((
            k[batch_index, :, :seq_len, :].unsqueeze(0).contiguous(),
            v[batch_index, :, :seq_len, :].unsqueeze(0).contiguous(),
        ) + tuple(layer[2:]))
    return tuple(selected)


@torch.no_grad()
def _target_batched_forward_with_cache(
    *,
    model,
    leaves: List[ContextKey],
    root_len: int,
    lookahead: int,
    target_past_key_values: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> Tuple[Dict[ContextKey, FloatTensor], Any]:
    """ONE batched target forward over all depth-L leaves.  Handles cache
    reuse: feeds only the suffix [cached_n:] per leaf.  Returns the
    per-prefix logprobs dict and the new B-repeated cache (caller must
    select+truncate to keep the alive row)."""
    if not leaves:
        raise ValueError("cannot batch target over an empty leaf set")
    device = model.device
    batch_ids = _batch_from_contexts(leaves, device)  # (B, root_len + L)
    n_leaves = batch_ids.shape[0]
    full_seq_len = batch_ids.shape[1]

    if target_past_key_values is None:
        cached_n = 0
        target_past_repeated = None
    else:
        cached_n = cache_len(target_past_key_values)
        if cached_n > root_len - 1:
            # Cache is "ahead" of where we want to predict from; this would lose
            # the d=0 (next-token from root) logits.  In a sane generator flow,
            # cached_n should always be exactly root_len - 1 at this point.
            raise RuntimeError(
                f"target cache too long (cached_n={cached_n}, root_len={root_len}); "
                "cache should cover at most [0, root_len - 1)"
            )
        target_past_repeated = _repeat_cache(target_past_key_values, n_leaves)

    new_input_ids = batch_ids[:, cached_n:]  # (B, full_seq_len - cached_n)

    out = model(
        input_ids=new_input_ids,
        past_key_values=target_past_repeated,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    new_target_cache = out.past_key_values
    logits_all = out.logits  # (B, full_seq_len - cached_n, V)
    if max_vocab_size is not None and max_vocab_size < logits_all.shape[-1]:
        logits_all = logits_all[..., :max_vocab_size]

    logprobs_by_context: Dict[ContextKey, FloatTensor] = {}
    for row, leaf in enumerate(leaves):
        for d in range(0, lookahead + 1):
            context = leaf[: root_len + d]
            if context in logprobs_by_context:
                continue
            pos = root_len + d - 1 - cached_n  # position in OUTPUT (not full seq)
            if pos < 0:
                raise RuntimeError(
                    f"output position {pos} is negative; cached_n={cached_n}, "
                    f"root_len={root_len}, d={d}"
                )
            raw_logits = logits_all[row, pos, :]
            ids_for_processor = torch.tensor(
                [list(context)], device=device, dtype=torch.long
            )
            processed = process_logits(
                ids_for_processor,
                raw_logits.unsqueeze(0),
                **(process_logits_kwargs or {}),
            )
            logprobs_by_context[context] = F.log_softmax(processed.float(), dim=-1)[0]

    return logprobs_by_context, new_target_cache


@torch.no_grad()
def multi_draft_pfr_batched_cached_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    root: Optional[ContextKey],
    lookahead: int,
    num_drafts: int,
    labeler: AbstractLabeler,
    source_factory: PFRSourceFactory,
    max_new_tokens: int,
    target_past_key_values: Any = None,
    ref_past_key_values: Any = None,    # currently passed through to draft tree
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
) -> MultiDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if input_ids.shape[0] != 1:
        raise AssertionError("only batch_size=1 is supported")

    device = model.device
    input_ids = input_ids.to(device)
    if root is None:
        root = _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    shared_labels: Dict[ContextKey, bytes] = {}
    shared_sources: Dict[ContextKey, SharedPFRSource] = {}
    target_context_cache = _new_context_cache(
        labeler, source_factory, device,
        label_by_key=shared_labels, source_by_key=shared_sources,
    )
    draft_context_cache = _new_context_cache(
        labeler, source_factory, ref_model.device,
        label_by_key=shared_labels, source_by_key=shared_sources,
    )

    levels, _, draft_sets, _draft_sources, draft_past_by_context = build_multi_draft_tree(
        ref_model=ref_model,
        input_ids=input_ids.to(ref_model.device),
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        context_cache=draft_context_cache,
        past_key_values=ref_past_key_values,
        max_vocab_size=max_vocab_size,
        process_logits_kwargs=process_logits_kwargs,
    )

    leaves = levels[block_len] if block_len < len(levels) else []
    if not leaves:
        raise RuntimeError("draft tree did not reach lookahead depth")

    logprobs_by_context, new_target_cache_repeated = _target_batched_forward_with_cache(
        model=model,
        leaves=leaves,
        root_len=root_len,
        lookahead=block_len,
        target_past_key_values=target_past_key_values,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
    )

    current = root
    output_tokens: List[int] = []
    output_logprobs: List[FloatTensor] = []
    accepted = True
    accepted_count = 0
    got_eos = False
    target_context_count = 0

    for _ in range(block_len):
        logprobs = logprobs_by_context[current]
        source = target_context_cache.source(current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
        )[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        target_context_count += 1

        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True
            accepted = False
            break
        if token not in draft_sets.get(current, set()):
            accepted = False
            break

        accepted_count += 1
        current = current + (token,)
        if len(output_tokens) >= max_new_tokens:
            break

    if accepted and not got_eos and len(output_tokens) < max_new_tokens:
        logprobs = logprobs_by_context[current]
        source = target_context_cache.source(current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
        )[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        target_context_count += 1
        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)

    # Find a leaf row whose prefix matches `current` (the realized path up to
    # the accepted depth).  Any such row will do: we only use cache up to
    # length root_len + accepted_count, where all leaves with `current` as
    # prefix have identical cache contents.
    alive_row_idx = 0
    cur_len = len(current)
    for row_idx, leaf in enumerate(leaves):
        if leaf[:cur_len] == current:
            alive_row_idx = row_idx
            break

    # Truncate B-repeated cache to alive row, keep [0, root_len + accepted_count).
    new_cache_len = root_len + accepted_count
    truncated_target_cache = _select_and_truncate_cache(
        new_target_cache_repeated, alive_row_idx, new_cache_len
    )

    # Pick the deepest cached draft ancestor of `current`. build_multi_draft_tree
    # populates past_by_context for depths 0..block_len-1; if accepted_count
    # reached block_len the realized prefix sits at the deepest level which has
    # no entry, so fall back to its parent (next block re-encodes 1-2 tokens
    # instead of the full prompt). Cache covers [0, len(ctx)) which is at most
    # root_len + accepted_count, matching the next block's expected
    # cache_len <= new_root_len - 1 = root_len + accepted_count.
    draft_past_for_next = draft_past_by_context.get(current)
    if draft_past_for_next is None and len(current) > root_len:
        draft_past_for_next = draft_past_by_context.get(current[:-1])

    return MultiDraftBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        draft_tree_size=sum(len(level) for level in levels),
        target_context_count=target_context_count,
        target_past_key_values=truncated_target_cache,
        draft_past_key_values=draft_past_for_next,
        got_eos=got_eos,
    )


@torch.no_grad()
def multi_draft_pfr_batched_cached_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    num_drafts: int,
    max_length: int,
    private_key: bytes,
    labeler: Optional[AbstractLabeler] = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
) -> Iterable[Tuple[LongTensor, FloatTensor, Dict[str, Any]]]:
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
    target_past_key_values = None
    ref_past_key_values = None
    generated = 0

    while generated < max_length:
        block = multi_draft_pfr_batched_cached_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            root=current_key,
            lookahead=lookahead,
            num_drafts=num_drafts,
            labeler=labeler,
            source_factory=source_factory,
            max_new_tokens=max_length - generated,
            target_past_key_values=target_past_key_values,
            ref_past_key_values=ref_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
        )
        meta = {
            "accepted_count": block.accepted_count,
            "draft_tree_size": block.draft_tree_size,
            "target_context_count": block.target_context_count,
            "draft_len": min(int(lookahead), int(max_length - generated)),
            "num_drafts": int(num_drafts),
            "target_forward_calls": 1,
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat([input_ids, block.output_ids], dim=1)
        current_key = current_key + tuple(int(t) for t in block.output_ids[0].tolist())
        target_past_key_values = block.target_past_key_values
        ref_past_key_values = block.draft_past_key_values
        generated += int(block.output_ids.shape[1])
        if block.got_eos:
            break


def multi_draft_pfr_batched_cached_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    private_key: bytes = b"1234",
    num_drafts: int = 2,
    B: Optional[int] = None,
    labeler: Optional[AbstractLabeler] = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
):
    if B is not None:
        num_drafts = int(B)
    yield from multi_draft_pfr_batched_cached_generator(
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
