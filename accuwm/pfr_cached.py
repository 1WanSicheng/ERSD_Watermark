"""
Single-draft PFR speculative decoding with batched verify and KV-cache reuse.

This is the B=1 specialization of MPFR_spec.multi_draft_pfr_batched_cached:
no draft tree, just one draft chain of length K.  Both target and draft KV
caches are threaded across blocks; the verify step is a single batched
target forward over the (1, L_in + K) sequence.

Both pfr (watermarked) and pfr_no_watermark (H0 control) wrap this module
with different `source_factory` arguments — every other code path is shared,
so AATPS is structurally identical and only the noise source differs.

Source factories:
  - PFRSourceFactory(private_key) -> seeded torch.Generator(cuda) noise
                                     (watermarked).
  - FreshNoiseSourceFactory()     -> fresh torch.rand noise, cached per
                                     (shape, device) within a source so
                                     draft and target see the SAME noise
                                     for a given context (no watermark).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .multi_draft_pfr import (
    ContextKey,
    _context_key,
    _repeat_cache,
    ms_pfr_tokens_from_logprobs,
)
from .pfr import AbstractLabeler, PFRSourceFactory, PrefixLabeler, SharedPFRSource
from .utils import cache_len, process_logits, truncate_cache


# ---------------------------------------------------------------------------
# Source factories
# ---------------------------------------------------------------------------


@dataclass
class FreshNoiseSource:
    """Drop-in for SharedPFRSource that uses fresh torch.rand on first call,
    caching the noise so draft and verify see the SAME values for a given
    context (the property that lets PFR speculative decoding deterministically
    accept tokens the target would also have chosen).

    Shape invariance: the cache GROWS along each axis as larger requests come
    in (never regenerated).  Smaller requests return a slice of the cached
    tensor so prefix-overlapping shapes (e.g. draft (1, 151936) and verify
    (1, 152064)) agree bit-for-bit on the overlapping range — exactly what
    torch.Generator(cuda) gives us in the seeded SharedPFRSource path.

    Different source instances produce independent noise (mirroring how
    different (label, key) pairs give independent noise in SharedPFRSource),
    but the noise itself is unseeded, so detection has no way to recover
    the per-token uniforms (no watermark).
    """
    _cache: Dict[str, torch.Tensor] = field(default_factory=dict)

    def uniform_noise(self, shape, device):
        device_key = str(device)
        target_n, target_v = int(shape[0]), int(shape[1])
        existing = self._cache.get(device_key)
        if existing is None:
            self._cache[device_key] = torch.rand(
                (target_n, target_v), device=device, dtype=torch.float32,
            )
            return self._cache[device_key][:target_n, :target_v]

        cur_n, cur_v = existing.shape
        # Grow column dimension first (vocab axis), then row (samples).
        if target_v > cur_v:
            extra = torch.rand(
                (cur_n, target_v - cur_v), device=device, dtype=torch.float32,
            )
            existing = torch.cat([existing, extra], dim=1)
            cur_v = target_v
        if target_n > cur_n:
            extra = torch.rand(
                (target_n - cur_n, cur_v), device=device, dtype=torch.float32,
            )
            existing = torch.cat([existing, extra], dim=0)
            cur_n = target_n
        self._cache[device_key] = existing
        return existing[:target_n, :target_v]


@dataclass(frozen=True)
class FreshNoiseSourceFactory:
    """Source factory that ignores label/key and produces fresh-noise sources
    per call.  Each `build(label)` returns a new FreshNoiseSource."""
    def build(self, label: bytes) -> FreshNoiseSource:
        del label  # unused
        return FreshNoiseSource()


# ---------------------------------------------------------------------------
# Cache helpers (mirror MPFR_spec.multi_draft_pfr_batched_cached for B=1)
# ---------------------------------------------------------------------------


def _select_first_row_and_truncate_cache(cache: Any, seq_len: int) -> Any:
    """For B=1 there is only one row; this is the cheap version of the
    multi_draft variant's _select_and_truncate_cache."""
    if cache is None:
        return None
    seq_len = max(int(seq_len), 0)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:1, :, :seq_len, :].contiguous()
            cache.value_cache[i] = cache.value_cache[i][:1, :, :seq_len, :].contiguous()
        return cache
    selected = []
    for layer in cache:
        k, v = layer[:2]
        selected.append((
            k[:1, :, :seq_len, :].contiguous(),
            v[:1, :, :seq_len, :].contiguous(),
        ) + tuple(layer[2:]))
    return tuple(selected)


# ---------------------------------------------------------------------------
# Block: build draft chain (B=1), one batched target forward, walk realized
# path with shared per-context source.  Both target and draft caches are
# rolled forward to [0, L_in + acc) at end of block.
# ---------------------------------------------------------------------------


@torch.no_grad()
def _build_draft_chain(
    *,
    ref_model,
    input_ids: LongTensor,
    root: ContextKey,
    lookahead: int,
    labeler: AbstractLabeler,
    source_factory: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    ref_past_key_values: Any,
    max_vocab_size: Optional[int],
) -> Tuple[List[ContextKey], Dict[ContextKey, Any], Dict[ContextKey, Any], Any]:
    """Sample a single draft chain of length `lookahead` via Poisson
    first-arrival under the draft (ref) model.  Reuses ref_past_key_values
    across iterations.

    Caller MUST pass an already-forked labeler (we mutate it freely);
    ``labeler._seen`` will be polluted with labels for speculative tokens
    that may end up rejected, which is fine because the caller commits
    only the realized-path labels to its own labeler.
    """
    device = ref_model.device
    cached_n = cache_len(ref_past_key_values)
    if cached_n > 0:
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids

    chain: List[ContextKey] = [root]
    sources: Dict[ContextKey, Any] = {}
    label_infos: Dict[ContextKey, Any] = {}

    current = root
    running_ids = input_ids
    past_key_values = ref_past_key_values

    for _ in range(lookahead):
        # Resolve source for current context (deterministic per label).
        if current not in sources:
            label_info = labeler.label_info(running_ids)
            sources[current] = source_factory.build(label_info.source_label)
            label_infos[current] = label_info

        out = ref_model(
            input_tokens,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **(process_logits_kwargs or {}))
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)

        token = int(ms_pfr_tokens_from_logprobs(
            logprobs[0], source=sources[current], num_samples=1, device=device,
        )[0].item())

        new_token = torch.tensor([[token]], device=device, dtype=torch.long)
        current = current + (token,)
        chain.append(current)
        running_ids = torch.cat([running_ids, new_token], dim=1)
        input_tokens = new_token
        past_key_values = out.past_key_values

    return chain, sources, label_infos, past_key_values


@torch.no_grad()
def _target_batched_forward_with_cache(
    *,
    model,
    leaf_ids: LongTensor,         # (1, L_in + K)
    root_len: int,
    lookahead: int,
    target_past_key_values: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
) -> Tuple[List[FloatTensor], Any]:
    """ONE target forward over the depth-K chain with cache reuse.  Returns
    per-position log-probs at depths 0..K (each shape (V,)) and the new
    cache (covering [0, root_len + K))."""
    device = model.device
    full_seq_len = leaf_ids.shape[1]
    if target_past_key_values is None:
        cached_n = 0
    else:
        cached_n = cache_len(target_past_key_values)
        if cached_n > root_len - 1:
            raise RuntimeError(
                f"target cache too long (cached_n={cached_n}, root_len={root_len}); "
                "cache should cover at most [0, root_len - 1)"
            )

    new_input_ids = leaf_ids[:, cached_n:]
    out = model(
        input_ids=new_input_ids,
        past_key_values=target_past_key_values,
        use_cache=True,
        return_dict=True,
    )
    new_target_cache = out.past_key_values
    logits_all = out.logits  # (1, full_seq_len - cached_n, V)
    if max_vocab_size is not None and max_vocab_size < logits_all.shape[-1]:
        logits_all = logits_all[..., :max_vocab_size]

    logprobs_per_depth: List[FloatTensor] = []
    for d in range(lookahead + 1):
        pos = root_len + d - 1 - cached_n
        if pos < 0:
            raise RuntimeError(
                f"output position {pos} negative; cached_n={cached_n}, root_len={root_len}, d={d}"
            )
        raw_logits = logits_all[0, pos, :]
        prefix_ids = leaf_ids[:, : root_len + d]
        processed = process_logits(
            prefix_ids, raw_logits.unsqueeze(0), **(process_logits_kwargs or {}),
        )
        logprobs_per_depth.append(F.log_softmax(processed[0].float(), dim=-1))

    return logprobs_per_depth, new_target_cache


@dataclass
class PFRCachedBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    draft_len: int
    target_past_key_values: Any
    ref_past_key_values: Any
    labels: List[Any]
    got_eos: bool


@torch.no_grad()
def pfr_cached_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    root: Optional[ContextKey],
    lookahead: int,
    labeler: AbstractLabeler,
    source_factory: Any,
    target_past_key_values: Any = None,
    ref_past_key_values: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
) -> PFRCachedBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if input_ids.shape[0] != 1:
        raise AssertionError("only batch_size=1 is supported")

    device = model.device
    input_ids = input_ids.to(device)
    if root is None:
        root = _context_key(input_ids)
    root_len = len(root)
    if max_new_tokens is None:
        max_new_tokens = lookahead
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = getattr(model.config, "vocab_size", None)

    # Fork the labeler for this block: draft-side label_info() calls mutate
    # _seen with K speculative tokens, most of which may end up rejected.
    # We commit ONLY the realized path's labels back to `labeler` at the
    # end, mirroring accuwm.pfr.pfr_speculative_generator's fork+commit
    # pattern.  This is essential: without it, parent labeler accumulates
    # rejected speculative cc's and later real tokens with those cc's get
    # mis-flagged as repeated, which corrupts both watermark labels and
    # decoded output.
    if hasattr(labeler, "fork"):
        block_labeler = labeler.fork()
    else:
        block_labeler = labeler

    # 1. Build draft chain with cache reuse on ref_model.
    chain, sources, label_infos, new_ref_cache = _build_draft_chain(
        ref_model=ref_model,
        input_ids=input_ids.to(ref_model.device),
        root=root,
        lookahead=block_len,
        labeler=block_labeler,
        source_factory=source_factory,
        process_logits_kwargs=process_logits_kwargs,
        ref_past_key_values=ref_past_key_values,
        max_vocab_size=max_vocab_size,
    )

    # 2. ONE batched target forward over the chain.
    leaf = chain[block_len]
    leaf_ids = torch.tensor([list(leaf)], device=device, dtype=torch.long)
    logprobs_per_depth, new_target_cache = _target_batched_forward_with_cache(
        model=model,
        leaf_ids=leaf_ids,
        root_len=root_len,
        lookahead=block_len,
        target_past_key_values=target_past_key_values,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
    )

    # 3. Walk realized path: at each depth, sample target's winner under the
    #    SAME source.uniform_noise as the draft used.  Accept iff draft's
    #    chosen token at this depth equals target's winner.
    current = root
    output_tokens: List[int] = []
    output_logprobs: List[FloatTensor] = []
    accepted = True
    accepted_count = 0
    got_eos = False
    labels: List[Any] = []

    eos_token_id = getattr(model.config, "eos_token_id", None)

    for d in range(block_len):
        logprobs = logprobs_per_depth[d]
        # Source for `current` was set during draft; lookup is sufficient.
        target_token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=sources[current], num_samples=1, device=device,
        )[0].item())

        draft_token = int(chain[d + 1][-1])

        output_tokens.append(target_token)
        output_logprobs.append(logprobs)

        if eos_token_id is not None and target_token == eos_token_id:
            got_eos = True
            accepted = False
            break

        if target_token != draft_token:
            accepted = False
            break

        accepted_count += 1
        current = current + (target_token,)
        if len(output_tokens) >= max_new_tokens:
            break

    if accepted and not got_eos and len(output_tokens) < max_new_tokens:
        # Bonus token: target distribution at depth = block_len, sampled
        # under the source for `current` = chain[block_len] which was NOT
        # visited during draft (draft only creates sources for chain[0..K-1]).
        logprobs = logprobs_per_depth[block_len]
        if current not in sources:
            if len(current) > root_len:
                tail = list(current[root_len:])
                ctx_ids = torch.cat([
                    input_ids,
                    torch.tensor([tail], device=device, dtype=torch.long),
                ], dim=1)
            else:
                ctx_ids = input_ids
            label_info = block_labeler.label_info(ctx_ids)
            sources[current] = source_factory.build(label_info.source_label)
            label_infos[current] = label_info
        bonus_token = int(ms_pfr_tokens_from_logprobs(
            logprobs, source=sources[current], num_samples=1, device=device,
        )[0].item())
        output_tokens.append(bonus_token)
        output_logprobs.append(logprobs)
        if eos_token_id is not None and bonus_token == eos_token_id:
            got_eos = True

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    output_logprobs_tensor = torch.stack(output_logprobs, dim=0).unsqueeze(0)

    # 4. Roll caches forward to [0, root_len + accepted_count).
    new_cache_len = root_len + accepted_count
    truncated_target_cache = _select_first_row_and_truncate_cache(
        new_target_cache, new_cache_len
    )
    truncated_ref_cache = truncate_cache(
        new_ref_cache, min(new_cache_len, cache_len(new_ref_cache)),
    )

    # 5. Realized-path labels: for each emitted token at index i, its label
    # is the one used to source the noise that picked it -- chain[i] is the
    # prefix at sampling time.  Commit these (and only these) to the parent
    # labeler so it tracks accepted-history correctly.
    realized_labels: List[Any] = []
    for i in range(len(output_tokens)):
        ctx = chain[i] if i < len(chain) else chain[-1]
        if ctx in label_infos:
            realized_labels.append(label_infos[ctx])
    if hasattr(labeler, "commit"):
        labeler.commit(realized_labels)

    return PFRCachedBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        accepted_count=accepted_count,
        draft_len=block_len,
        target_past_key_values=truncated_target_cache,
        ref_past_key_values=truncated_ref_cache,
        labels=realized_labels,
        got_eos=got_eos,
    )


def _labels_for_realized_path(
    *,
    chain: List[ContextKey],
    sources: Dict[ContextKey, Any],
    accepted_count: int,
    emitted: int,
    labeler: AbstractLabeler,
    source_factory: Any,
    input_ids: LongTensor,
    root_len: int,
    device,
) -> List[Any]:
    """Collect the source labels that drove each emitted token's noise.
    Follows the same convention as accuwm.pfr.PFRVerifyBlock.labels: one
    LabelInfo per emitted token, using the prefix at that step."""
    labels: List[Any] = []
    for i in range(emitted):
        ctx = chain[i] if i < len(chain) else chain[-1]
        # Reconstruct the context as a token tensor for labeler.label_info.
        if len(ctx) > root_len:
            tail = list(ctx[root_len:])
            ctx_ids = torch.cat([
                input_ids,
                torch.tensor([tail], device=device, dtype=torch.long),
            ], dim=1)
        else:
            ctx_ids = input_ids
        labels.append(labeler.label_info(ctx_ids))
    return labels


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@torch.no_grad()
def pfr_cached_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    max_length: int,
    source_factory: Any,
    labeler: Optional[AbstractLabeler] = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
) -> Iterable[Tuple[LongTensor, FloatTensor, Dict[str, Any]]]:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    if labeler is None:
        labeler = PrefixLabeler()

    model.eval()
    ref_model.eval()
    input_ids = input_ids.to(model.device)
    current_key = _context_key(input_ids)
    target_past_key_values = None
    ref_past_key_values = None
    generated = 0

    while generated < max_length:
        block = pfr_cached_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            root=current_key,
            lookahead=lookahead,
            labeler=labeler,
            source_factory=source_factory,
            target_past_key_values=target_past_key_values,
            ref_past_key_values=ref_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            max_new_tokens=max_length - generated,
        )
        meta = {
            "accepted_count": block.accepted_count,
            "draft_len": block.draft_len,
            "labels": block.labels,
            "num_drafts": 1,
            "target_forward_calls": 1,
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat([input_ids, block.output_ids], dim=1)
        current_key = current_key + tuple(int(t) for t in block.output_ids[0].tolist())
        target_past_key_values = block.target_past_key_values
        ref_past_key_values = block.ref_past_key_values
        generated += int(block.output_ids.shape[1])
        if block.got_eos:
            break


def pfr_cached_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    *,
    source_factory: Any = None,
    private_key: bytes = b"1234",
    watermark: bool = True,
    labeler: Optional[AbstractLabeler] = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
):
    """Single-draft PFR speculative generator with batched verify and KV
    cache reuse.  Pass `watermark=False` to swap the keyed source for a
    fresh-noise source (drops the watermark, keeps the algorithm).

    `source_factory` overrides `watermark`/`private_key` if provided.
    """
    if source_factory is None:
        if watermark:
            source_factory = PFRSourceFactory(private_key=private_key)
        else:
            source_factory = FreshNoiseSourceFactory()
    yield from pfr_cached_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        lookahead=n,
        max_length=max_length,
        source_factory=source_factory,
        labeler=labeler,
        process_logits_kwargs=process_logits_kwargs,
        return_meta=return_meta,
    )
