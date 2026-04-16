"""
PFR Aaronson detector with UWM-style skip-repeats semantics.

This variant keeps PFR generation unchanged, but detection skips repeated
labels/context-codes in the same spirit as `ContextCodeHistory.step()` used by
the UWM family. It is intended for apples-to-apples detector comparisons.
"""

import numpy as np
from torch import LongTensor

from .ersd_aaronson import _log_gamma_tail_p_value
from .pfr_aaronson import _uniform_for_token


def compute_pfr_aaronson_skip_from_sequence(
    full_ids: LongTensor,
    prompt_length: int,
    labeler,
    private_key: bytes,
    vocab_size: int,
) -> dict:
    assert full_ids.shape[0] == 1, "only batch_size=1 is supported"
    assert 0 <= prompt_length <= full_ids.shape[1]

    scores = []
    skipped = []
    for pos in range(prompt_length, full_ids.shape[1]):
        context = full_ids[:, :pos]
        token_id = int(full_ids[0, pos].item())
        label_info = labeler.label_info(context)
        if label_info.masked:
            skipped.append(True)
            continue
        skipped.append(False)
        r_t = _uniform_for_token(
            label_info.source_label,
            private_key,
            token_id,
            vocab_size,
        )
        scores.append(-np.log(1.0 - r_t))

    score_sum = float(np.sum(scores)) if scores else 0.0
    m = len(scores)
    log_p_value = _log_gamma_tail_p_value(m, score_sum)
    anlppt = float(-log_p_value / m) if m else 0.0
    return {
        "score_sum": score_sum,
        "num_scored": m,
        "num_skipped": int(sum(skipped)),
        "log_p_value": float(log_p_value),
        "ANLPPT_Aaronson": anlppt,
        "score_mean": float(score_sum / m) if m else 0.0,
        "masked_ratio": float(np.mean(skipped)) if skipped else 0.0,
    }
