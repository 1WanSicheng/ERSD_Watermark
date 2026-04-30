"""
Definition 3.1-style empirical watermark-strength estimator for keyed PFR.

This module measures, for a realized sequence w_{1:N}, the average target-side
surprisal of the keyed PFR winner together with the average target entropy:

    WS_hat = (1/M) sum_t -log P(y_t^* | c_t)
    H_hat  = (1/M) sum_t H(P(. | c_t))
    ratio  = WS_hat / H_hat

where y_t^* is the recovered target-side PFR winner under the same
context-keyed source Pi_t = G(k, Lambda(c_t)).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import LongTensor

import accuwm.pfr as pfr


@torch.no_grad()
def compute_pfr_watermark_strength_from_sequence(
    full_ids: LongTensor,
    prompt_length: int,
    target_model,
    labeler,
    private_key: bytes,
    skip_repeats: bool = False,
    process_logits_kwargs: dict | None = None,
) -> dict:
    """
    Empirical estimation of keyed-PFR watermark strength from a realized
    prompt+output sequence.

    Args:
        full_ids:
            Tensor of shape (1, N) containing prompt and generated output.
        prompt_length:
            Prompt length n0. Positions prompt_length..N-1 are candidates for
            scoring.
        target_model:
            Target model P used to evaluate P(.|c_t).
        labeler:
            Stateful Lambda implementation. It is called on each prefix c_t.
        private_key:
            Secret key k for the keyed PFR source.
        skip_repeats:
            If true, repeated labels / context codes are excluded from the
            scored set T.
        process_logits_kwargs:
            Optional logits post-processing arguments passed through
            accuwm.utils.process_logits. Defaults to none.

    Returns:
        Dict with:
            - WS_PFR_hat
            - H_P_hat
            - ratio
            - num_scored
            - masked_ratio
            - ws_sum
            - entropy_sum
    """
    assert full_ids.shape[0] == 1, "only batch_size=1 is supported"
    assert 0 <= prompt_length <= full_ids.shape[1]

    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    if prompt_length == full_ids.shape[1]:
        return {
            "WS_PFR_hat": 0.0,
            "H_P_hat": 0.0,
            "ratio": 0.0,
            "num_scored": 0,
            "masked_ratio": 0.0,
            "ws_sum": 0.0,
            "entropy_sum": 0.0,
        }

    outputs = target_model(full_ids)
    next_logits = outputs.logits[:, :-1, :]

    ws_terms = []
    entropy_terms = []
    masked = []
    source_factory = pfr.PFRSourceFactory(private_key=private_key)

    for pos in range(prompt_length, full_ids.shape[1]):
        context = full_ids[:, :pos]
        label_info = labeler.label_info(context)
        masked.append(label_info.masked)
        if skip_repeats and label_info.masked:
            continue

        raw_logits = next_logits[:, pos - 1, :]

        # Both H(p) and -log p(winner) must be evaluated on the SAME
        # distribution. Earlier this used un-truncated logits for entropy but
        # process_logits-truncated logits for ws, so on diffuse base-model
        # logits (e.g. llama-7b base, where top_k=50 strips ~18% of entropy)
        # the ratio looked like 0.80 even though the PFR algorithm itself is
        # unbiased on the truncated distribution actually being sampled from.
        # Evaluate both on the post-process_logits distribution.
        logits = pfr.process_logits(context, raw_logits, **process_logits_kwargs)
        logprobs = F.log_softmax(logits, dim=-1)
        # When top_k/top_p mask positions to -inf, the corresponding probs are
        # 0 and the 0*-inf entropy term is undefined; replace -inf with 0 in
        # the multiplicand (still gives 0 * 0 = 0, which is the correct limit).
        finite_logprobs = torch.where(
            torch.isfinite(logprobs), logprobs, torch.zeros_like(logprobs)
        )
        probs = logprobs.exp()
        entropy = -(probs * finite_logprobs).sum(dim=-1)
        entropy_terms.append(float(entropy.item()))

        source = source_factory.build(label_info.source_label)
        winner, _, _ = pfr.pfr_win_from_logits(logits, source, target_model.device)

        winner_logprob = logprobs.gather(-1, winner.unsqueeze(-1)).squeeze(-1)
        ws_terms.append(float((-winner_logprob).item()))

    m = len(ws_terms)
    ws_sum = float(np.sum(ws_terms)) if ws_terms else 0.0
    entropy_sum = float(np.sum(entropy_terms)) if entropy_terms else 0.0
    ws_hat = float(ws_sum / m) if m else 0.0
    h_hat = float(entropy_sum / m) if m else 0.0
    return {
        "WS_PFR_hat": ws_hat,
        "H_P_hat": h_hat,
        "ratio": float(ws_hat / h_hat) if h_hat > 0.0 else 0.0,
        "num_scored": m,
        "masked_ratio": float(np.mean(masked)) if masked else 0.0,
        "ws_sum": ws_sum,
        "entropy_sum": entropy_sum,
    }
