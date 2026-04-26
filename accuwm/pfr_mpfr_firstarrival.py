"""
Single-draft speculative PFR with finite-MPFR first arrivals.

This is the single-draft algorithm from ``pfr.py`` with the winner primitive
replaced by ``MPFR(dist, proposal, source, 1)``.  The target and draft sides use
the same keyed source for the same context.  The finite MPFR proposal is a
fixed public discrete Gaussian over vocabulary indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .finite_multi_draft_pfr import FiniteMPFRResult, finite_mpfr_tokens_from_logprobs
from .multi_draft_pfr import _repeat_cache
from .pfr import (
    AbstractContextCodeExtractor,
    LabelInfo,
    PFRSourceFactory,
    RepeatedContextMaskingLabeler,
    SharedPFRSource,
    _clone_cache,
    build_default_labeler,
)
from .utils import cache_len, process_logits, truncate_cache


@dataclass
class MPFRFirstArrivalDraftBlock:
    draft_tokens: LongTensor
    draft_logprobs: FloatTensor
    bonus_logprobs: FloatTensor | None
    draft_mpfr_results: list[FiniteMPFRResult]
    labels: list[LabelInfo]
    sources: list[SharedPFRSource]
    draft_past_key_values: any
    got_eos: bool


@dataclass
class MPFRFirstArrivalVerifyBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    labels: list[LabelInfo]
    accepted_count: int
    verify_mpfr_results: list[FiniteMPFRResult]
    bonus_mpfr_result: FiniteMPFRResult | None
    target_past_key_values: any
    got_eos: bool


def gaussian_proposal_logprobs(
    vocab_size: int,
    device,
    *,
    mean: float | None = None,
    sigma: float | None = None,
) -> FloatTensor:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if mean is None:
        mean = (vocab_size - 1) / 2.0
    if sigma is None:
        sigma = max(vocab_size / 6.0, 1.0)
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("gaussian proposal sigma must be finite and positive")

    ids = torch.arange(vocab_size, device=device, dtype=torch.float32)
    logits = -0.5 * ((ids - float(mean)) / float(sigma)) ** 2
    return F.log_softmax(logits, dim=-1)


def uniform_proposal_logprobs(vocab_size: int, device) -> FloatTensor:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return torch.full(
        (vocab_size,),
        -math.log(vocab_size),
        device=device,
        dtype=torch.float32,
    )


def mpfr_first_arrival_from_logits(
    logits: FloatTensor,
    source: SharedPFRSource,
    device,
    *,
    proposal_logprobs: FloatTensor | None = None,
    proposal: str = "uniform",
    gaussian_mean: float | None = None,
    gaussian_sigma: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
    return_result: bool = False,
) -> tuple[LongTensor, FloatTensor] | tuple[LongTensor, FloatTensor, FiniteMPFRResult]:
    logprobs = F.log_softmax(logits, dim=-1)
    if logprobs.dim() == 2:
        if logprobs.shape[0] != 1:
            raise ValueError("mpfr_first_arrival_from_logits only supports batch size 1")
        flat_logprobs = logprobs[0]
    else:
        flat_logprobs = logprobs

    if proposal_logprobs is None:
        if proposal == "uniform":
            proposal_logprobs = uniform_proposal_logprobs(flat_logprobs.shape[-1], device=device)
        elif proposal == "draft_model":
            proposal_logprobs = flat_logprobs
        elif proposal == "gaussian":
            proposal_logprobs = gaussian_proposal_logprobs(
                flat_logprobs.shape[-1],
                device=device,
                mean=gaussian_mean,
                sigma=gaussian_sigma,
            )
        else:
            raise ValueError(f"unknown MPFR proposal: {proposal}")
    else:
        proposal_logprobs = proposal_logprobs.to(device=device)
        if proposal_logprobs.shape[-1] != flat_logprobs.shape[-1]:
            raise ValueError("proposal_logprobs and logits must have the same vocabulary size")

    mpfr_result = finite_mpfr_tokens_from_logprobs(
        flat_logprobs,
        proposal_logprobs,
        source=source,
        num_samples=1,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
        return_result=True,
    )
    token = mpfr_result.tokens[0].view(1)
    if return_result:
        return token, logprobs, mpfr_result
    return token, logprobs


def _mpfr_result_meta(results: list[FiniteMPFRResult]) -> list[dict]:
    return [
        {
            "num_proposals": result.num_proposals,
            "w_lower": result.w_lower,
            "terminated": result.terminated,
        }
        for result in results
    ]


@torch.no_grad()
def draft_mpfr_firstarrival_block(
    model,
    input_ids: LongTensor,
    lookahead: int,
    labeler: RepeatedContextMaskingLabeler,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
    proposal: str = "uniform",
    gaussian_mean: float | None = None,
    gaussian_sigma: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
) -> MPFRFirstArrivalDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    cached_n = cache_len(past_key_values)
    input_tokens = input_ids[:, cached_n:] if cached_n > 0 else input_ids
    running_ids = input_ids
    draft_tokens = []
    draft_logprobs = []
    draft_mpfr_results: list[FiniteMPFRResult] = []
    labels: list[LabelInfo] = []
    sources: list[SharedPFRSource] = []
    got_eos = False
    bonus_logprobs = None

    for _ in range(lookahead):
        output = model(input_tokens, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]

        label_info = labeler.label_info(running_ids)
        source = source_factory.build(label_info.source_label)
        new_token, logprobs, mpfr_result = mpfr_first_arrival_from_logits(
            logits,
            source,
            model.device,
            proposal=proposal,
            gaussian_mean=gaussian_mean,
            gaussian_sigma=gaussian_sigma,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            return_result=True,
        )

        new_token = new_token.unsqueeze(1)
        draft_tokens.append(new_token)
        draft_logprobs.append(logprobs)
        draft_mpfr_results.append(mpfr_result)
        labels.append(label_info)
        sources.append(source)

        running_ids = torch.cat([running_ids, new_token], dim=1)
        input_tokens = new_token
        past_key_values = output.past_key_values

        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break

    if draft_tokens:
        draft_tokens_tensor = torch.cat(draft_tokens, dim=1)
        draft_logprobs_tensor = torch.stack(draft_logprobs, dim=1)
        if not got_eos:
            bonus_past_key_values = _repeat_cache(past_key_values, 1)
            bonus_output = model(input_tokens, past_key_values=bonus_past_key_values)
            bonus_logits = bonus_output.logits[:, -1, :]
            bonus_logits = process_logits(running_ids, bonus_logits, **process_logits_kwargs)
            if max_vocab_size is not None and max_vocab_size < bonus_logits.shape[-1]:
                bonus_logits = bonus_logits[..., :max_vocab_size]
            bonus_logprobs = F.log_softmax(bonus_logits, dim=-1)
    else:
        vocab_size = min(model.config.vocab_size, max_vocab_size or model.config.vocab_size)
        draft_tokens_tensor = torch.empty((1, 0), device=input_ids.device, dtype=torch.long)
        draft_logprobs_tensor = torch.empty(
            (1, 0, vocab_size), device=input_ids.device, dtype=torch.float32
        )

    return MPFRFirstArrivalDraftBlock(
        draft_tokens=draft_tokens_tensor,
        draft_logprobs=draft_logprobs_tensor,
        bonus_logprobs=bonus_logprobs,
        draft_mpfr_results=draft_mpfr_results,
        labels=labels,
        sources=sources,
        draft_past_key_values=past_key_values,
        got_eos=got_eos,
    )


@torch.no_grad()
def verify_mpfr_firstarrival_block(
    model,
    input_ids: LongTensor,
    draft_block: MPFRFirstArrivalDraftBlock,
    labeler: RepeatedContextMaskingLabeler,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
    proposal: str = "uniform",
    gaussian_mean: float | None = None,
    gaussian_sigma: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
) -> MPFRFirstArrivalVerifyBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    draft_len = draft_block.draft_tokens.shape[1]
    if draft_len == 0:
        raise ValueError("verify_mpfr_firstarrival_block requires at least one draft token")
    draft_tokens = draft_block.draft_tokens.to(model.device)

    cached_n = cache_len(past_key_values)
    if cached_n > 0:
        input_tokens = torch.cat([input_ids[:, cached_n:], draft_tokens], dim=1)
    else:
        input_tokens = torch.cat([input_ids, draft_tokens], dim=1)

    full_ids = torch.cat([input_ids, draft_tokens], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    logits = output.logits[:, -draft_len - 1 :, :]
    if proposal == "draft_model" and draft_block.draft_logprobs.shape[-1] < logits.shape[-1]:
        logits = logits[..., : draft_block.draft_logprobs.shape[-1]]

    for i in range(draft_len + 1):
        logits[:, i, :] = process_logits(
            full_ids[:, : input_ids.shape[1] + i],
            logits[:, i, :],
            **process_logits_kwargs,
        )

    target_logprobs = F.log_softmax(logits, dim=-1)
    proposal_logprobs = (
        draft_block.draft_logprobs.to(model.device) if proposal == "draft_model" else None
    )
    accepted_tokens: list[int] = []
    labels: list[LabelInfo] = []
    verify_mpfr_results: list[FiniteMPFRResult] = []
    bonus_mpfr_result = None
    accepted_count = 0

    for s in range(draft_len):
        source = draft_block.sources[s]
        winner, _, mpfr_result = mpfr_first_arrival_from_logits(
            logits[:, s, :],
            source,
            model.device,
            proposal_logprobs=(
                proposal_logprobs[0, s, :] if proposal_logprobs is not None else None
            ),
            proposal=proposal,
            gaussian_mean=gaussian_mean,
            gaussian_sigma=gaussian_sigma,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            return_result=True,
        )
        verify_mpfr_results.append(mpfr_result)
        winner_id = int(winner.item())
        draft_token = int(draft_tokens[0, s].item())
        if winner_id != draft_token:
            accepted_prefix = (
                torch.cat([input_ids, draft_tokens[:, :s]], dim=1)
                if s > 0
                else input_ids
            )
            label_info = labeler.label_info(accepted_prefix)
            labels.append(label_info)
            accepted_tokens.append(winner_id)
            accepted_count = s
            break

        accepted_tokens.append(draft_token)
        labels.append(draft_block.labels[s])
        labeler.commit([draft_block.labels[s]])
        accepted_count += 1
    else:
        bonus_context = full_ids
        bonus_label = labeler.label_info(bonus_context)
        bonus_source = source_factory.build(bonus_label.source_label)
        bonus_proposal_logprobs = None
        if proposal == "draft_model":
            if draft_block.bonus_logprobs is None:
                raise ValueError("draft_model proposal requires bonus draft logprobs")
            bonus_proposal_logprobs = draft_block.bonus_logprobs.to(model.device)
        bonus_winner, _, bonus_mpfr_result = mpfr_first_arrival_from_logits(
            logits[:, draft_len, :],
            bonus_source,
            model.device,
            proposal_logprobs=bonus_proposal_logprobs,
            proposal=proposal,
            gaussian_mean=gaussian_mean,
            gaussian_sigma=gaussian_sigma,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
            return_result=True,
        )
        accepted_tokens.append(int(bonus_winner.item()))
        labels.append(bonus_label)

    output_ids = torch.tensor([accepted_tokens], device=input_ids.device, dtype=torch.long)
    output_logprobs = target_logprobs[:, : len(accepted_tokens), :]
    got_eos = bool((output_ids == model.config.eos_token_id).any())
    new_cache_len = input_ids.shape[1] + output_ids.shape[1] - 1
    new_past = truncate_cache(output.past_key_values, new_cache_len)

    return MPFRFirstArrivalVerifyBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        labels=labels,
        accepted_count=accepted_count,
        verify_mpfr_results=verify_mpfr_results,
        bonus_mpfr_result=bonus_mpfr_result,
        target_past_key_values=new_past,
        got_eos=got_eos,
    )


@torch.no_grad()
def pfr_mpfr_firstarrival_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    max_length: int,
    labeler: RepeatedContextMaskingLabeler,
    private_key: bytes,
    past_key_values=None,
    ref_past_key_values=None,
    process_logits_kwargs: dict | None = None,
    proposal: str = "uniform",
    gaussian_mean: float | None = None,
    gaussian_sigma: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
):
    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    source_factory = PFRSourceFactory(private_key=private_key)
    initial_len = input_ids.shape[1]

    while input_ids.shape[1] < initial_len + max_length:
        remaining = initial_len + max_length - input_ids.shape[1]
        block_len = min(lookahead, remaining)
        draft_labeler = labeler.fork()
        verify_labeler = labeler.fork()
        draft_block = draft_mpfr_firstarrival_block(
            model=ref_model,
            input_ids=input_ids.to(ref_model.device),
            lookahead=block_len,
            labeler=draft_labeler,
            source_factory=source_factory,
            past_key_values=_clone_cache(ref_past_key_values),
            process_logits_kwargs=process_logits_kwargs,
            max_vocab_size=model.config.vocab_size,
            proposal=proposal,
            gaussian_mean=gaussian_mean,
            gaussian_sigma=gaussian_sigma,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
        )
        verify_block = verify_mpfr_firstarrival_block(
            model=model,
            input_ids=input_ids.to(model.device),
            draft_block=draft_block,
            labeler=verify_labeler,
            source_factory=source_factory,
            past_key_values=past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            proposal=proposal,
            gaussian_mean=gaussian_mean,
            gaussian_sigma=gaussian_sigma,
            max_proposals=max_proposals,
            allow_incomplete=allow_incomplete,
        )
        labeler.commit(verify_block.labels)

        yield (
            verify_block.output_ids,
            verify_block.output_logprobs,
            {
                "accepted_count": verify_block.accepted_count,
                "labels": verify_block.labels,
                "draft_len": draft_block.draft_tokens.shape[1],
                "draft_mpfr": _mpfr_result_meta(draft_block.draft_mpfr_results),
                "verify_mpfr": _mpfr_result_meta(verify_block.verify_mpfr_results),
                "bonus_mpfr": (
                    _mpfr_result_meta([verify_block.bonus_mpfr_result])[0]
                    if verify_block.bonus_mpfr_result is not None
                    else None
                ),
            },
        )

        input_ids = torch.cat([input_ids.to(model.device), verify_block.output_ids], dim=1)
        past_key_values = verify_block.target_past_key_values

        ref_cached_n = cache_len(ref_past_key_values)
        ref_update_ids = input_ids[:, ref_cached_n:].to(ref_model.device)
        ref_update_out = ref_model(ref_update_ids, past_key_values=ref_past_key_values)
        ref_past_key_values = truncate_cache(
            ref_update_out.past_key_values,
            input_ids.shape[1] - 1,
        )

        if verify_block.got_eos:
            break


def pfr_mpfr_firstarrival_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    private_key: bytes = b"1234",
    labeler: RepeatedContextMaskingLabeler | None = None,
    labeler_mode: str = "context_code",
    cc_extractor: AbstractContextCodeExtractor | None = None,
    process_logits_kwargs: dict | None = None,
    proposal: str = "uniform",
    gaussian_mean: float | None = None,
    gaussian_sigma: float | None = None,
    max_proposals: int = 100_000,
    allow_incomplete: bool = False,
):
    if labeler is None:
        labeler = build_default_labeler(
            mode=labeler_mode,
            cc_extractor=cc_extractor,
        )
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")
    yield from pfr_mpfr_firstarrival_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        lookahead=n,
        max_length=max_length,
        labeler=labeler,
        private_key=private_key,
        process_logits_kwargs=process_logits_kwargs,
        proposal=proposal,
        gaussian_mean=gaussian_mean,
        gaussian_sigma=gaussian_sigma,
        max_proposals=max_proposals,
        allow_incomplete=allow_incomplete,
    )
