"""
Pure, non-watermarked PFR speculative decoding.

This implements the PFR/ERSD-style acceleration mechanism without a keyed
watermark source:

1. Draft samples tokens by exponential race under Q.
2. Target verifies with the same sampled exponential noises under P.
3. On full acceptance, target emits one extra token using a fresh unkeyed race.

No context labels, no key, no repeated-context masking, and no watermark score
metadata are used here. This file is intended as an acceleration-only baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .utils import cache_len, process_logits, truncate_cache


def _safe_log(x: FloatTensor, eps: float = 1e-20) -> FloatTensor:
    return torch.log(x.clamp(min=eps))


def _sample_exp_noise(shape, device) -> FloatTensor:
    return -_safe_log(torch.rand(shape, device=device, dtype=torch.float32))


def _noise_to_winner(probs: FloatTensor, exp_noise: FloatTensor) -> LongTensor:
    ratios = torch.where(
        probs > 0,
        exp_noise / probs,
        torch.full_like(probs, float("inf")),
    )
    return ratios.argmin(dim=-1)


def _win_from_logits(
    logits: FloatTensor,
    exp_noise: FloatTensor | None = None,
) -> tuple[LongTensor, FloatTensor, FloatTensor]:
    logprobs = F.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    if exp_noise is None:
        exp_noise = _sample_exp_noise(probs.shape, logits.device)
    winner = _noise_to_winner(probs, exp_noise)
    return winner, logprobs, exp_noise


@dataclass
class PurePFRDraftBlock:
    draft_tokens: LongTensor
    draft_logprobs: FloatTensor
    exp_noises: list[FloatTensor]
    draft_past_key_values: any
    got_eos: bool


@dataclass
class PurePFRVerifyBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    accepted_count: int
    target_past_key_values: any
    got_eos: bool


@torch.no_grad()
def draft_block(
    model,
    input_ids: LongTensor,
    lookahead: int,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
) -> PurePFRDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    cached_n = cache_len(past_key_values)
    input_tokens = input_ids[:, cached_n:] if cached_n > 0 else input_ids
    running_ids = input_ids
    draft_tokens = []
    draft_logprobs = []
    exp_noises = []
    got_eos = False

    for _ in range(lookahead):
        output = model(input_tokens, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]

        new_token, logprobs, exp_noise = _win_from_logits(logits)
        new_token = new_token.unsqueeze(1)
        draft_tokens.append(new_token)
        draft_logprobs.append(logprobs)
        exp_noises.append(exp_noise[0])

        running_ids = torch.cat([running_ids, new_token], dim=1)
        input_tokens = new_token
        past_key_values = output.past_key_values

        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break

    if draft_tokens:
        draft_tokens_tensor = torch.cat(draft_tokens, dim=1)
        draft_logprobs_tensor = torch.stack(draft_logprobs, dim=1)
    else:
        vocab_size = min(model.config.vocab_size, max_vocab_size or model.config.vocab_size)
        draft_tokens_tensor = torch.empty((1, 0), device=input_ids.device, dtype=torch.long)
        draft_logprobs_tensor = torch.empty(
            (1, 0, vocab_size), device=input_ids.device, dtype=torch.float32
        )

    return PurePFRDraftBlock(
        draft_tokens=draft_tokens_tensor,
        draft_logprobs=draft_logprobs_tensor,
        exp_noises=exp_noises,
        draft_past_key_values=past_key_values,
        got_eos=got_eos,
    )


@torch.no_grad()
def verify_block(
    model,
    input_ids: LongTensor,
    draft: PurePFRDraftBlock,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
) -> PurePFRVerifyBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    draft_len = draft.draft_tokens.shape[1]
    if draft_len == 0:
        raise ValueError("verify_block requires at least one draft token")

    cached_n = cache_len(past_key_values)
    if cached_n > 0:
        input_tokens = torch.cat([input_ids[:, cached_n:], draft.draft_tokens], dim=1)
    else:
        input_tokens = torch.cat([input_ids, draft.draft_tokens], dim=1)

    full_ids = torch.cat([input_ids, draft.draft_tokens], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    logits = output.logits[:, -draft_len - 1 :, :]
    if draft.exp_noises:
        min_vocab = min(logits.shape[-1], draft.exp_noises[0].shape[-1])
        logits = logits[..., :min_vocab]

    for i in range(draft_len + 1):
        logits[:, i, :] = process_logits(
            full_ids[:, : input_ids.shape[1] + i],
            logits[:, i, :],
            **process_logits_kwargs,
        )

    target_logprobs = F.log_softmax(logits, dim=-1)
    target_probs = target_logprobs.exp()
    accepted_tokens = []
    accepted_count = 0

    for s in range(draft_len):
        winner = _noise_to_winner(
            target_probs[0, s, :].unsqueeze(0),
            draft.exp_noises[s].unsqueeze(0).to(model.device),
        ).item()
        draft_token = draft.draft_tokens[0, s].item()
        if winner != draft_token:
            accepted_tokens.append(winner)
            accepted_count = s
            break
        accepted_tokens.append(draft_token)
        accepted_count += 1
    else:
        bonus_winner, _, _ = _win_from_logits(logits[:, draft_len, :])
        accepted_tokens.append(int(bonus_winner.item()))

    output_ids = torch.tensor([accepted_tokens], device=input_ids.device, dtype=torch.long)
    output_logprobs = target_logprobs[:, : len(accepted_tokens), :]
    got_eos = bool((output_ids == model.config.eos_token_id).any())
    new_cache_len = input_ids.shape[1] + output_ids.shape[1] - 1
    new_past = truncate_cache(output.past_key_values, new_cache_len)
    return PurePFRVerifyBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        accepted_count=accepted_count,
        target_past_key_values=new_past,
        got_eos=got_eos,
    )


@torch.no_grad()
def pfr_no_watermark_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    past_key_values=None,
    ref_past_key_values=None,
    process_logits_kwargs: dict | None = None,
):
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    initial_len = input_ids.shape[1]
    while input_ids.shape[1] < initial_len + max_length:
        remaining = initial_len + max_length - input_ids.shape[1]
        block_len = min(n, remaining)
        draft = draft_block(
            model=ref_model,
            input_ids=input_ids,
            lookahead=block_len,
            past_key_values=ref_past_key_values,
            max_vocab_size=model.config.vocab_size,
            process_logits_kwargs=process_logits_kwargs,
        )
        verified = verify_block(
            model=model,
            input_ids=input_ids,
            draft=draft,
            past_key_values=past_key_values,
            process_logits_kwargs=process_logits_kwargs,
        )
        yield (
            verified.output_ids,
            verified.output_logprobs,
            {
                "accepted_count": verified.accepted_count,
                "draft_len": draft.draft_tokens.shape[1],
            },
        )

        input_ids = torch.cat([input_ids, verified.output_ids], dim=1)
        past_key_values = verified.target_past_key_values

        # Keep this baseline correctness-oriented: rebuild draft cache on demand
        # rather than reusing speculative-path cache.
        ref_past_key_values = None

        if verified.got_eos:
            break
