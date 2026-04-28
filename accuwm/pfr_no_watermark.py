"""
No-watermark PFR speculative decoding baseline.

This is a thin wrapper around accuwm.pfr_cached that runs the SAME
single-draft + batched-verify + KV-cache reuse pipeline as accuwm.pfr but
with FreshNoiseSourceFactory in place of the keyed PFRSourceFactory.  Only
the noise source differs, so AATPS is structurally identical to pfr.py and
only the watermark detector strength changes (no detectable signal here).
"""
from __future__ import annotations

from torch import LongTensor

from .pfr import build_default_labeler
from .pfr_cached import FreshNoiseSourceFactory, pfr_cached_sample_generator


def pfr_no_watermark_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    process_logits_kwargs: dict | None = None,
    labeler=None,
    return_meta: bool = True,
):
    """Drop-in for the previous standalone implementation: yields
    (output_ids, output_logprobs, meta) tuples per block.  Uses fresh
    torch.rand noise per per-context source instead of the keyed PFR
    source, so the decoded sequence has no recoverable watermark."""
    if labeler is None:
        # Use a context_code labeler so the per-token source labels match
        # the convention used by pfr.py.  For the no-watermark variant the
        # FreshNoiseSourceFactory ignores the label, so the labeler choice
        # only affects bookkeeping (label_info objects in meta).
        labeler = build_default_labeler(mode="context_code")
    yield from pfr_cached_sample_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        n=n,
        max_length=max_length,
        watermark=False,
        source_factory=FreshNoiseSourceFactory(),
        labeler=labeler,
        process_logits_kwargs=process_logits_kwargs,
        return_meta=return_meta,
    )
