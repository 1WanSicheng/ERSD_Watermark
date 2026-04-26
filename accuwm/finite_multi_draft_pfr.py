"""
Watermarkable multi-draft speculative sampling with finite MPFR races.

This module follows the same speculative structure as ``multi_draft_pfr.py``:
draft branches come from the draft model ``Q`` and target-model winners verify
the realized path.  The difference is only the implementation of the abstract
``MPFR(P, source, B)`` primitive.  Here MPFR is implemented by the finite
proposal algorithm with a fixed public uniform proposal distribution; the model
distribution is the target distribution inside that primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

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
from .utils import cache_len, process_logits


@dataclass(frozen=True)
class FiniteMPFRResult:
    tokens: LongTensor
    scores: FloatTensor
    proposal_indices: LongTensor
    num_proposals: int
    w_lower: float
    terminated: bool


@dataclass
class FiniteMultiDraftBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    draft_tree_size: int
    target_context_count: int
    target_past_key_values: any
    draft_past_key_values: any
    got_eos: bool


def _safe_log(x: FloatTensor, eps: float = 1e-20) -> FloatTensor:
    return torch.log(x.clamp(min=eps))


def _as_1d_logprobs(logprobs: FloatTensor, name: str) -> FloatTensor:
    if logprobs.dim() == 2:
        if logprobs.shape[0] != 1:
            raise ValueError(f"{name} must be unbatched or have batch size 1")
        logprobs = logprobs[0]
    if logprobs.dim() != 1:
        raise ValueError(f"{name} must be a 1D log-probability vector")
    return logprobs


def _support_checked_ratio_and_bound(
    target_logprobs: FloatTensor,
    proposal_logprobs: FloatTensor,
    w_lower: float | None,
) -> tuple[FloatTensor, float]:
    target_logprobs = target_logprobs.float()
    proposal_logprobs = proposal_logprobs.float()
    target_positive = torch.isfinite(target_logprobs)
    proposal_positive = torch.isfinite(proposal_logprobs)
    if bool((target_positive & ~proposal_positive).any().item()):
        raise ValueError("finite MPFR requires P to be absolutely continuous with respect to Q")

    ratio = torch.zeros_like(target_logprobs)
    ratio[proposal_positive] = torch.exp(
        target_logprobs[proposal_positive] - proposal_logprobs[proposal_positive]
    )

    if w_lower is None:
        active = target_positive & proposal_positive
        if not bool(active.any().item()):
            raise ValueError("target distribution has empty support")
        log_inv_ratio = proposal_logprobs[active] - target_logprobs[active]
        w_lower_value = math.exp(float(log_inv_ratio.min().item()))
    else:
        w_lower_value = float(w_lower)

    if not math.isfinite(w_lower_value) or w_lower_value <= 0:
        raise ValueError("w_lower must be finite and positive")
    return ratio, w_lower_value


def finite_mpfr_tokens_from_logprobs(
    target_logprobs: FloatTensor,
    proposal_logprobs: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    *,
    w_lower: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    return_result: bool = False,
    chunk_size: int = 2048,
) -> LongTensor | FiniteMPFRResult:
    if num_samples <= 0:
        empty_tokens = torch.empty((0,), device=target_logprobs.device, dtype=torch.long)
        if return_result:
            return FiniteMPFRResult(
                tokens=empty_tokens,
                scores=torch.empty((0,), device=target_logprobs.device),
                proposal_indices=torch.empty((0,), device=target_logprobs.device, dtype=torch.long),
                num_proposals=0,
                w_lower=float(w_lower) if w_lower is not None else float("nan"),
                terminated=True,
            )
        return empty_tokens
    if max_proposals < num_samples:
        raise ValueError("max_proposals must be at least num_samples")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = target_logprobs.device
    target_logprobs = _as_1d_logprobs(target_logprobs, "target_logprobs")
    proposal_logprobs = _as_1d_logprobs(proposal_logprobs, "proposal_logprobs")
    if target_logprobs.shape != proposal_logprobs.shape:
        raise ValueError("target_logprobs and proposal_logprobs must have the same shape")

    ratio, w_lower_value = _support_checked_ratio_and_bound(
        target_logprobs,
        proposal_logprobs,
        w_lower,
    )
    proposal_probs = proposal_logprobs.exp()
    proposal_probs = proposal_probs / proposal_probs.sum()
    proposal_cdf = torch.cumsum(proposal_probs, dim=0)
    proposal_cdf[-1] = 1.0

    same_distribution = bool(torch.equal(target_logprobs, proposal_logprobs))
    numpy_rng = source.numpy_rng()

    def draw_uniforms(count: int) -> FloatTensor:
        uniforms_np = numpy_rng.random((count, 2)).astype("float32")
        return torch.from_numpy(uniforms_np).to(device=device)

    if num_samples == 1 and same_distribution:
        uniforms = draw_uniforms(1)
        exp_interarrival = -_safe_log(uniforms[:, 0])
        token = torch.searchsorted(proposal_cdf, uniforms[:, 1].contiguous(), right=False).long()
        if not return_result:
            return token
        return FiniteMPFRResult(
            tokens=token,
            scores=exp_interarrival.to(dtype=target_logprobs.dtype),
            proposal_indices=torch.ones((1,), device=device, dtype=torch.long),
            num_proposals=1,
            w_lower=w_lower_value,
            terminated=True,
        )

    best_scores = torch.empty((0,), device=device, dtype=torch.float32)
    best_indices = torch.empty((0,), device=device, dtype=torch.long)
    best_tokens = torch.empty((0,), device=device, dtype=torch.long)
    t = 0.0
    used = 0
    terminated = False

    while used < max_proposals:
        take = min(chunk_size, max_proposals - used)
        uniforms = draw_uniforms(take)
        exp_interarrivals = -_safe_log(uniforms[:, 0])
        times = float(t) + torch.cumsum(exp_interarrivals, dim=0)
        tokens = torch.searchsorted(proposal_cdf, uniforms[:, 1].contiguous(), right=False).long()
        token_ratio = ratio[tokens]
        scores = torch.where(
            token_ratio > 0,
            times / token_ratio,
            torch.full_like(times, float("inf")),
        )
        indices = torch.arange(
            used + 1,
            used + take + 1,
            device=device,
            dtype=torch.long,
        )

        if best_scores.numel() > 0:
            combined_scores = torch.cat([best_scores, scores])
            combined_indices = torch.cat([best_indices, indices])
            combined_tokens = torch.cat([best_tokens, tokens])
        else:
            combined_scores = scores
            combined_indices = indices
            combined_tokens = tokens

        keep = min(num_samples, combined_scores.numel())
        best_scores, best_pos = torch.topk(
            combined_scores,
            k=keep,
            largest=False,
            sorted=True,
        )
        best_indices = combined_indices[best_pos]
        best_tokens = combined_tokens[best_pos]

        used += take
        t = float(times[-1].item())
        if best_scores.numel() >= num_samples and float(best_scores[-1].item()) <= t * w_lower_value:
            terminated = True
            break

    if not terminated and not allow_incomplete:
        raise RuntimeError(
            "finite MPFR did not terminate within max_proposals; "
            "increase max_proposals or check w_lower"
        )

    tokens = best_tokens
    scores_out = best_scores.to(dtype=target_logprobs.dtype)
    indices_out = best_indices
    if not return_result:
        return tokens

    return FiniteMPFRResult(
        tokens=tokens,
        scores=scores_out,
        proposal_indices=indices_out,
        num_proposals=used,
        w_lower=w_lower_value,
        terminated=terminated,
    )


def finite_mpfr_tokens_from_logits(
    target_logits: FloatTensor,
    proposal_logits: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    **kwargs,
) -> LongTensor | FiniteMPFRResult:
    return finite_mpfr_tokens_from_logprobs(
        F.log_softmax(target_logits, dim=-1),
        F.log_softmax(proposal_logits, dim=-1),
        source,
        num_samples,
        **kwargs,
    )


def finite_mpfr_tokens_from_uniform_proposal(
    target_logprobs: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    **kwargs,
) -> LongTensor | FiniteMPFRResult:
    target_logprobs = _as_1d_logprobs(target_logprobs, "target_logprobs")
    target_probs = target_logprobs.float().exp()
    target_mass = target_probs.sum()
    if float(target_mass.item()) <= 0:
        raise ValueError("target distribution has empty support")
    target_probs = target_probs / target_mass
    proposal_logprobs = torch.full(
        target_probs.shape,
        -math.log(target_probs.shape[-1]),
        device=target_logprobs.device,
        dtype=torch.float32,
    )
    kwargs.setdefault(
        "w_lower",
        float((1.0 / (target_probs.shape[-1] * target_probs.max())).item()),
    )
    return finite_mpfr_tokens_from_logprobs(
        _safe_log(target_probs),
        proposal_logprobs,
        source,
        num_samples,
        **kwargs,
    )


def finite_mpfr_tokens_from_model_proposal(
    target_logprobs: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    **kwargs,
) -> LongTensor | FiniteMPFRResult:
    target_logprobs = _as_1d_logprobs(target_logprobs, "target_logprobs")
    return finite_mpfr_tokens_from_logprobs(
        target_logprobs,
        target_logprobs,
        source,
        num_samples,
        **kwargs,
    )


def finite_mpfr_tokens_from_proposal(
    target_logprobs: FloatTensor,
    source: SharedPFRSource,
    num_samples: int,
    *,
    proposal: str = "uniform",
    **kwargs,
) -> LongTensor | FiniteMPFRResult:
    if proposal == "uniform":
        return finite_mpfr_tokens_from_uniform_proposal(
            target_logprobs,
            source=source,
            num_samples=num_samples,
            **kwargs,
        )
    if proposal == "model":
        return finite_mpfr_tokens_from_model_proposal(
            target_logprobs,
            source=source,
            num_samples=num_samples,
            **kwargs,
        )
    raise ValueError(f"unknown finite MPFR proposal: {proposal}")


@torch.no_grad()
def build_finite_multi_draft_tree(
    ref_model,
    input_ids: LongTensor,
    root: ContextKey,
    lookahead: int,
    num_drafts: int,
    context_cache,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "uniform",
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
            draft_tokens = finite_mpfr_tokens_from_proposal(
                logprobs[row],
                source=source,
                num_samples=multiplicities[context],
                proposal=proposal,
                max_proposals=max_proposals,
                allow_incomplete=allow_incomplete,
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
def evaluate_finite_target_context(
    model,
    context: ContextKey,
    context_cache,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    proposal: str = "uniform",
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
    logprobs = F.log_softmax(logits, dim=-1)
    source = context_cache.source(context)
    tokens = finite_mpfr_tokens_from_proposal(
        logprobs[0],
        source=source,
        num_samples=1,
        proposal=proposal,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
    )
    return int(tokens[0].item()), logprobs, source, output.past_key_values


@torch.no_grad()
def finite_multi_draft_pfr_block(
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
    proposal: str = "uniform",
) -> FiniteMultiDraftBlock:
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
    _levels, _, draft_sets, draft_sources, draft_past_by_context = build_finite_multi_draft_tree(
        ref_model=ref_model,
        input_ids=input_ids.to(ref_model.device),
        root=root,
        lookahead=block_len,
        num_drafts=num_drafts,
        context_cache=draft_context_cache,
        past_key_values=ref_past_key_values,
        max_vocab_size=model.config.vocab_size,
        process_logits_kwargs=process_logits_kwargs,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
        proposal=proposal,
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
            token, logprobs, source, target_context_past = evaluate_finite_target_context(
                model=model,
                context=current,
                context_cache=target_context_cache,
                past_key_values=past_key_values,
                process_logits_kwargs=process_logits_kwargs,
                max_proposals=max_proposals,
                allow_incomplete=allow_incomplete,
                proposal=proposal,
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
        token, logprobs, bonus_source, target_past_for_next = evaluate_finite_target_context(
            model=model,
            context=current,
            context_cache=target_context_cache,
            past_key_values=past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            proposal=proposal,
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
    return FiniteMultiDraftBlock(
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
def finite_multi_draft_pfr_generator(
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
    proposal: str = "uniform",
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
        block = finite_multi_draft_pfr_block(
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


def finite_multi_draft_pfr_sample_generator(
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
    proposal: str = "uniform",
):
    if B is not None:
        num_drafts = B
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
