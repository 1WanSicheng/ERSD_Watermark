"""
Cache-aware variant of mpfr_batched_torchgen.

Same algorithm as mpfr_batched_torchgen.py (MPFR_spec direct-finite MPFR with
GPU torch.Generator sampling, draft tree + ONE batched target forward over
depth-L leaves) but threads the TARGET KV cache across blocks AND within the
batched verify forward.  Draft side remains cache-free for simplicity.

Cache convention (matches multi_draft_pfr_batched_cached / StrongMultiDraft):
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
from dataclasses import dataclass
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

from accuwm.multi_draft_utils import _repeat_cache, ms_pfr_tokens_from_logprobs
from accuwm.pfr import PFRSourceFactory, SharedPFRSource
from accuwm.utils import cache_len


def _context_label(context: ContextKey) -> bytes:
    """Per-context bytes label fed to SharedPFRSource.  Mirrors the
    per-context blake2b seed in the original mpfr_direct_optimized."""
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in context
    )


def _gpu_source_for_context(factory: PFRSourceFactory, context: ContextKey) -> SharedPFRSource:
    return factory.build(_context_label(context))


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
) -> DraftTree:
    """Build the multi-draft tree using the GPU torch.Generator MPFR primitive.
    Identical to mpfr_direct_optimized.build_draft_tree_direct except the
    sampler is ms_pfr_tokens_from_logprobs (GPU) instead of the CPU/blake2b
    direct_finite primitive."""
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
        processed = process_logits_exact(
            logits, temperature=draft_temp, top_k=top_k, top_p=top_p
        )
        logprobs = _normalize_logprobs_from_processed_logits(processed)

        next_level: List[ContextKey] = []
        seen_next: set[ContextKey] = set()

        for row, context in enumerate(prev_level):
            source = _gpu_source_for_context(source_factory, context)
            draft_tokens = ms_pfr_tokens_from_logprobs(
                logprobs[row],
                source=source,
                num_samples=multiplicities[context],
                device=draft_device,
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


@dataclass
class MPFRCachedBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    attempted_draft_tokens: int
    draft_tree_size: int
    target_context_count: int
    target_forward_calls: int
    draft_forward_calls: int
    got_eos: bool
    target_past_key_values: Any


def _select_and_truncate_cache(cache: Any, batch_index: int, seq_len: int) -> Any:
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
def _target_logprobs_from_leaves_one_forward_cached(
    *,
    model,
    leaves: List[ContextKey],
    root_len: int,
    lookahead: int,
    target_past_key_values: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> Tuple[Dict[ContextKey, FloatTensor], Any]:
    if not leaves:
        raise ValueError("cannot batch target over an empty leaf set")
    device = _model_device(model)
    batch_ids = _batch_from_contexts(leaves, device)  # (B, root_len + L)
    n_leaves = batch_ids.shape[0]

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
        target_past_repeated = _repeat_cache(target_past_key_values, n_leaves)

    new_input_ids = batch_ids[:, cached_n:]

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

    target_temp = _temperature_for_target(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)

    logprobs_by_context: Dict[ContextKey, FloatTensor] = {}
    for row, leaf in enumerate(leaves):
        for d in range(0, lookahead + 1):
            context = leaf[: root_len + d]
            if context in logprobs_by_context:
                continue
            pos = root_len + d - 1 - cached_n
            if pos < 0:
                raise RuntimeError(
                    f"output position {pos} is negative; cached_n={cached_n}, "
                    f"root_len={root_len}, d={d}"
                )
            raw_logits = logits_all[row, pos, :]
            processed = process_logits_exact(
                raw_logits, temperature=target_temp, top_k=top_k, top_p=top_p
            )
            logprobs_by_context[context] = _normalize_logprobs_from_processed_logits(processed)

    return logprobs_by_context, new_target_cache


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
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
) -> MPFRCachedBlock:
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    draft_tree = build_draft_tree_torchgen(
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
        raise RuntimeError("draft tree did not reach lookahead depth")

    logprobs_by_context, new_target_cache_repeated = (
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

    for _ in range(block_len):
        logprobs = logprobs_by_context[current]
        source = _gpu_source_for_context(source_factory, current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
        )[0].item())

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
        logprobs = logprobs_by_context[current]
        source = _gpu_source_for_context(source_factory, current)
        token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=source, num_samples=1, device=device,
        )[0].item())
        output_tokens.append(token)
        output_logprobs.append(logprobs)
        if token == getattr(model.config, "eos_token_id", None):
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)

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

    return MPFRCachedBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        attempted_draft_tokens=block_len,
        draft_tree_size=draft_tree.draft_tree_size,
        target_context_count=len(logprobs_by_context),
        target_forward_calls=1,
        draft_forward_calls=draft_tree.draft_forward_calls,
        got_eos=got_eos,
        target_past_key_values=truncated_target_cache,
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
            process_logits_kwargs=process_logits_kwargs,
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

        input_ids = torch.cat([input_ids, block.output_ids.to(device)], dim=1)
        target_past_key_values = block.target_past_key_values
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
    )
