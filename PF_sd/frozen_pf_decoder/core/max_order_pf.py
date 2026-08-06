"""Algorithm 2 from ``ICLR_27_Speculative_PF_water.pdf``.

This is a correctness-first PyTorch implementation of watermarkable max-order
Permute-and-Flip (PF) speculative decoding with two defining properties:

* ``B`` independent keyed PF fields produce ``B`` draft trajectories.  We do
  not take the top-B order statistics of one field.
* equal draft prefixes are merged, so each depth contains at most ``B``
  contexts and the complete tree contains at most ``1 + B * L`` contexts.

PF noisy-max is evaluated through its uniform-race identity

    argmax_y {log(alpha_y) - log(U_y)} = argmin_y U_y / alpha_y,

which avoids a vocabulary-wide ``-log(U)``.  The functions are deliberately
factored so this reference primitive can later be replaced by a fused
Triton/CUDA reduction without changing the decoder.
"""
from __future__ import annotations

import inspect
import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from accuwm.multi_draft_utils import (
    _gather_cache_rows,
    _repeat_cache,
    _select_cache_row,
)
from accuwm.pfr import PFRSourceFactory, SharedPFRSource
from accuwm.utils import cache_len
from MPFR_spec.mpfr_batched_torchgen_cached import _select_and_truncate_cache
from MPFR_spec.mpfr_direct_optimized import (
    ContextKey,
    _batch_from_contexts,
    _context_key,
    _model_device,
    _temperature_for_draft,
    _temperature_for_target,
    _top_k,
    _top_p,
    process_logits_exact,
)

try:  # Optional: dense reference remains usable without Triton.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on lightweight CPU installs
    triton = None
    tl = None


MAX_ORDER_CONTEXT_DOMAIN = b"MAX_ORDER_PF_CONTEXT_V1"
MAX_ORDER_FIELD_DOMAIN = b"::FIELD::"
RANDOM_ANCHOR_DOMAIN = b"MAX_ORDER_PF_RANDOM_ANCHOR_V1::"
COUNTER_PRF_DOMAIN = b"MAX_ORDER_PF_COUNTER_PHILOX_V1::"


if triton is not None:
    @triton.jit
    def _counter_uniform_kernel(
        token_ids_ptr,
        field_ids_ptr,
        output_ptr,
        seed,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        n_fields: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        total = n_tokens * n_fields
        mask = offsets < total
        field_row = offsets // n_tokens
        token_col = offsets - field_row * n_tokens
        token = tl.load(token_ids_ptr + token_col, mask=mask, other=0)
        field = tl.load(field_ids_ptr + field_row, mask=mask, other=0)
        counter = (field * vocab_size + token).to(tl.uint32)
        values = tl.rand(seed, counter)
        tl.store(output_ptr + offsets, values, mask=mask)

    @triton.jit
    def _counter_draft_select_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        field_ids_ptr,
        output_tokens_ptr,
        seed,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        compact_logits: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        field_row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        token = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        logit_offset = offsets if compact_logits else token
        logits = tl.load(
            processed_logits_ptr + logit_offset, mask=mask, other=-float("inf")
        ).to(tl.float32)
        max_logit = tl.max(logits, axis=0)
        weight = tl.exp(logits - max_logit)
        field = tl.load(field_ids_ptr + field_row)
        counter = (field * vocab_size + token).to(tl.uint32)
        uniform = tl.rand(seed, counter)
        ratio = tl.where(mask, uniform / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner_token = tl.load(token_ids_ptr + winner_col)
        tl.store(output_tokens_ptr + field_row, winner_token)

    @triton.jit
    def _counter_latin_draft_select_kernel(
        processed_weights_ptr,
        token_ids_ptr,
        field_ids_ptr,
        output_tokens_ptr,
        seed,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        width: tl.constexpr,
        compact_logits: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fuse counter RNG, Latin stratification, and one draft PF race."""
        field_row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        token = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        logit_offset = offsets if compact_logits else token
        weight = tl.load(
            processed_weights_ptr + logit_offset, mask=mask, other=0.0
        ).to(tl.float32)
        field = tl.load(field_ids_ptr + field_row)
        shift_counter = (width * vocab_size + token).to(tl.uint32)
        shift = tl.rand(seed, shift_counter)
        shift_stratum = tl.floor(shift * width).to(tl.int32)
        stratum = (field + shift_stratum) % width
        raw_counter = (field * vocab_size + token).to(tl.uint32)
        raw = tl.rand(seed, raw_counter)
        uniform = (stratum.to(tl.float32) + raw) / width
        ratio = tl.where(mask, uniform / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner_token = tl.load(token_ids_ptr + winner_col)
        tl.store(output_tokens_ptr + field_row, winner_token)

    @triton.jit
    def _counter_latin_draft_select_batch_kernel(
        processed_weights_ptr,
        token_ids_ptr,
        context_rows_ptr,
        field_ids_ptr,
        seeds_ptr,
        output_tokens_ptr,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Select all active Latin draft trajectories at one tree depth."""
        trajectory = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        row = tl.load(context_rows_ptr + trajectory)
        field = tl.load(field_ids_ptr + trajectory)
        seed = tl.load(seeds_ptr + trajectory)
        token = tl.load(
            token_ids_ptr + row * n_tokens + offsets, mask=mask, other=0
        )
        weight = tl.load(
            processed_weights_ptr + row * n_tokens + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shift_counter = (width * vocab_size + token).to(tl.uint32)
        shift = tl.rand(seed, shift_counter)
        shift_stratum = tl.floor(shift * width).to(tl.int32)
        stratum = (field + shift_stratum) % width
        raw_counter = (field * vocab_size + token).to(tl.uint32)
        raw = tl.rand(seed, raw_counter)
        uniform = (stratum.to(tl.float32) + raw) / width
        ratio = tl.where(mask, uniform / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner = tl.load(token_ids_ptr + row * n_tokens + winner_col)
        tl.store(output_tokens_ptr + trajectory, winner)

    @triton.jit
    def _counter_latin_target_select_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        output_token_ptr,
        output_pivot_ptr,
        seed,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        width: tl.constexpr,
        compact_logits: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fuse all Latin fields, exact aggregate, and the target PF race."""
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        token = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        logit_offset = offsets if compact_logits else token
        logits = tl.load(
            processed_logits_ptr + logit_offset, mask=mask, other=-float("inf")
        ).to(tl.float32)
        max_logit = tl.max(logits, axis=0)
        weight = tl.exp(logits - max_logit)
        shift_counter = (width * vocab_size + token).to(tl.uint32)
        shift = tl.rand(seed, shift_counter)
        shift_stratum = tl.floor(shift * width).to(tl.int32)
        minimum = tl.full((BLOCK,), float("inf"), tl.float32)
        for field in range(width):
            stratum = (field + shift_stratum) % width
            raw_counter = (field * vocab_size + token).to(tl.uint32)
            raw = tl.rand(seed, raw_counter)
            uniform = (stratum.to(tl.float32) + raw) / width
            minimum = tl.minimum(minimum, uniform)
        aggregate = width * minimum
        ratio = tl.where(mask, aggregate / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner_token = tl.load(token_ids_ptr + winner_col)
        winner_pivot = tl.sum(
            tl.where(offsets == winner_col, aggregate, 0.0), axis=0
        )
        tl.store(output_token_ptr, winner_token)
        tl.store(output_pivot_ptr, winner_pivot)

    @triton.jit
    def _counter_target_select_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        output_token_ptr,
        output_pivot_ptr,
        seed,
        field_offset: tl.constexpr,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        n_fields: tl.constexpr,
        compact_logits: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        token = tl.load(token_ids_ptr + offsets, mask=mask, other=0)
        logit_offset = offsets if compact_logits else token
        logits = tl.load(
            processed_logits_ptr + logit_offset, mask=mask, other=-float("inf")
        ).to(tl.float32)
        max_logit = tl.max(logits, axis=0)
        weight = tl.exp(logits - max_logit)
        minimum = tl.full((BLOCK,), float("inf"), tl.float32)
        for field_row in range(n_fields):
            field = field_offset + field_row
            counter = (field * vocab_size + token).to(tl.uint32)
            uniform = tl.rand(seed, counter)
            minimum = tl.minimum(minimum, uniform)
        survival = tl.full((BLOCK,), 1.0, tl.float32)
        for _ in range(n_fields):
            survival *= 1.0 - minimum
        aggregate = 1.0 - survival
        ratio = tl.where(mask, aggregate / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner_token = tl.load(token_ids_ptr + winner_col)
        winner_pivot = tl.sum(
            tl.where(offsets == winner_col, aggregate, 0.0), axis=0
        )
        tl.store(output_token_ptr, winner_token)
        tl.store(output_pivot_ptr, winner_pivot)

    @triton.jit
    def _counter_target_select_batch_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        seeds_ptr,
        output_tokens_ptr,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        n_fields: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Select one max-order PF winner for every context row.

        This is the same reduction as ``_counter_target_select_kernel``.  The
        only difference is that context seeds and top-k supports are batched,
        so all already-computed target contexts require one kernel launch and
        one host synchronization instead of one of each per emitted token.
        """
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        token = tl.load(
            token_ids_ptr + row * n_tokens + offsets, mask=mask, other=0
        )
        logits = tl.load(
            processed_logits_ptr + row * vocab_size + token,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        max_logit = tl.max(logits, axis=0)
        weight = tl.exp(logits - max_logit)
        seed = tl.load(seeds_ptr + row)
        minimum = tl.full((BLOCK,), float("inf"), tl.float32)
        for field in range(n_fields):
            counter = (field * vocab_size + token).to(tl.uint32)
            uniform = tl.rand(seed, counter)
            minimum = tl.minimum(minimum, uniform)
        survival = tl.full((BLOCK,), 1.0, tl.float32)
        for _ in range(n_fields):
            survival *= 1.0 - minimum
        aggregate = 1.0 - survival
        ratio = tl.where(mask, aggregate / weight, float("inf"))
        winner_col = tl.argmin(ratio, axis=0)
        winner_token = tl.load(token_ids_ptr + row * n_tokens + winner_col)
        tl.store(output_tokens_ptr + row, winner_token)

    @triton.jit
    def _counter_target_traverse_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        seeds_ptr,
        child_tokens_ptr,
        child_rows_ptr,
        output_ptr,
        eos_token,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        n_fields: tl.constexpr,
        draft_steps: tl.constexpr,
        max_outputs: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Run the already-verified target path inside one GPU program."""
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        current_row = tl.full((), 0, tl.int64)
        active = tl.full((), True, tl.int1)
        generated = tl.full((), 0, tl.int64)
        accepted = tl.full((), 0, tl.int64)

        for step in range(max_outputs):
            token = tl.load(
                token_ids_ptr + current_row * n_tokens + offsets,
                mask=mask,
                other=0,
            )
            logits = tl.load(
                processed_logits_ptr + current_row * vocab_size + token,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            max_logit = tl.max(logits, axis=0)
            weight = tl.exp(logits - max_logit)
            seed = tl.load(seeds_ptr + current_row)
            minimum = tl.full((BLOCK,), float("inf"), tl.float32)
            for field in range(n_fields):
                counter = (field * vocab_size + token).to(tl.uint32)
                uniform = tl.rand(seed, counter)
                minimum = tl.minimum(minimum, uniform)
            survival = tl.full((BLOCK,), 1.0, tl.float32)
            for _ in range(n_fields):
                survival *= 1.0 - minimum
            aggregate = 1.0 - survival
            ratio = tl.where(mask, aggregate / weight, float("inf"))
            winner_col = tl.argmin(ratio, axis=0)
            winner = tl.load(token_ids_ptr + current_row * n_tokens + winner_col)

            tl.store(output_ptr + step, winner, mask=active != 0)
            tl.store(output_ptr + max_outputs + step, current_row, mask=active != 0)
            generated += active.to(tl.int64)

            next_row = tl.full((), -1, tl.int64)
            for child_slot in range(n_fields):
                candidate_token = tl.load(
                    child_tokens_ptr + current_row * n_fields + child_slot
                )
                candidate_row = tl.load(
                    child_rows_ptr + current_row * n_fields + child_slot
                )
                matches = (candidate_row >= 0) & (candidate_token == winner)
                next_row = tl.where(matches, candidate_row, next_row)

            can_continue = (
                (active != 0)
                & (step < draft_steps)
                & (winner != eos_token)
                & (next_row >= 0)
            )
            accepted += can_continue.to(tl.int64)
            current_row = tl.where(can_continue, next_row, current_row)
            active = can_continue

        tl.store(output_ptr + 2 * max_outputs, generated)
        tl.store(output_ptr + 2 * max_outputs + 1, accepted)
        tl.store(output_ptr + 2 * max_outputs + 2, current_row)

    @triton.jit
    def _counter_latin_target_traverse_kernel(
        processed_logits_ptr,
        token_ids_ptr,
        seeds_ptr,
        child_tokens_ptr,
        child_rows_ptr,
        output_ptr,
        eos_token,
        vocab_size: tl.constexpr,
        n_tokens: tl.constexpr,
        width: tl.constexpr,
        draft_steps: tl.constexpr,
        max_outputs: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Run compact Latin target selection and tree traversal in one kernel."""
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_tokens
        current_row = tl.full((), 0, tl.int64)
        active = tl.full((), True, tl.int1)
        generated = tl.full((), 0, tl.int64)
        accepted = tl.full((), 0, tl.int64)

        for step in range(max_outputs):
            token = tl.load(
                token_ids_ptr + current_row * n_tokens + offsets,
                mask=mask,
                other=0,
            )
            logits = tl.load(
                processed_logits_ptr + current_row * n_tokens + offsets,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            max_logit = tl.max(logits, axis=0)
            weight = tl.exp(logits - max_logit)
            seed = tl.load(seeds_ptr + current_row)
            shift_counter = (width * vocab_size + token).to(tl.uint32)
            shift = tl.rand(seed, shift_counter)
            shift_stratum = tl.floor(shift * width).to(tl.int32)
            minimum = tl.full((BLOCK,), float("inf"), tl.float32)
            for field in range(width):
                stratum = (field + shift_stratum) % width
                raw_counter = (field * vocab_size + token).to(tl.uint32)
                raw = tl.rand(seed, raw_counter)
                uniform = (stratum.to(tl.float32) + raw) / width
                minimum = tl.minimum(minimum, uniform)
            aggregate = width * minimum
            ratio = tl.where(mask, aggregate / weight, float("inf"))
            winner_col = tl.argmin(ratio, axis=0)
            winner = tl.load(token_ids_ptr + current_row * n_tokens + winner_col)

            tl.store(output_ptr + step, winner, mask=active != 0)
            tl.store(output_ptr + max_outputs + step, current_row, mask=active != 0)
            generated += active.to(tl.int64)
            next_row = tl.full((), -1, tl.int64)
            for child_slot in range(width):
                candidate_token = tl.load(
                    child_tokens_ptr + current_row * width + child_slot
                )
                candidate_row = tl.load(
                    child_rows_ptr + current_row * width + child_slot
                )
                matches = (candidate_row >= 0) & (candidate_token == winner)
                next_row = tl.where(matches, candidate_row, next_row)
            can_continue = (
                (active != 0)
                & (step < draft_steps)
                & (winner != eos_token)
                & (next_row >= 0)
            )
            accepted += can_continue.to(tl.int64)
            current_row = tl.where(can_continue, next_row, current_row)
            active = can_continue

        tl.store(output_ptr + 2 * max_outputs, generated)
        tl.store(output_ptr + 2 * max_outputs + 1, accepted)
        tl.store(output_ptr + 2 * max_outputs + 2, current_row)


@lru_cache(maxsize=None)
def _supports_num_logits_to_keep(model_type: type) -> bool:
    return "num_logits_to_keep" in inspect.signature(model_type.forward).parameters


def _model_forward(model, *, num_logits_to_keep: int, **kwargs):
    """Request only the logits needed by draft/target verification."""
    if _supports_num_logits_to_keep(type(model)):
        kwargs["num_logits_to_keep"] = int(num_logits_to_keep)
    return model(**kwargs)


def _compact_topk_logits(
    logits: FloatTensor, *, top_k: int, temperature: Any
) -> Tuple[FloatTensor, LongTensor]:
    """Return exactly the finite entries of top-k processing.

    ``process_logits_exact`` materializes a vocabulary-sized ``-inf`` tensor
    after ``torch.topk``.  Counter PF only reads the retained entries, so this
    helper preserves the same values and token IDs while avoiding that dense
    allocation/scatter.
    """
    values, support = torch.topk(logits, int(top_k), dim=-1, sorted=False)
    if not torch.is_tensor(temperature):
        temperature_value = float(temperature)
        if temperature_value == 1.0:
            return values, support
        temperature = torch.tensor(
            temperature_value, device=values.device, dtype=values.dtype
        )
    else:
        temperature = temperature.to(device=values.device, dtype=values.dtype)
    return values / temperature, support


def max_order_context_label(context: ContextKey, width: int) -> bytes:
    """Stable detector label containing the fixed aggregation width."""
    if width <= 0:
        raise ValueError("width must be positive")
    return (
        MAX_ORDER_CONTEXT_DOMAIN
        + int(width).to_bytes(4, "big", signed=False)
        + len(context).to_bytes(8, "big", signed=False)
        + b"".join(int(token).to_bytes(8, "big", signed=True) for token in context)
    )


def max_order_field_label(context_label: bytes, field: int) -> bytes:
    """Domain-separated label for one of the B independent PF fields."""
    if field < 0:
        raise ValueError("field must be non-negative")
    return (
        bytes(context_label)
        + MAX_ORDER_FIELD_DOMAIN
        + int(field).to_bytes(4, "big", signed=False)
    )


def random_anchor_field(
    source_factory: PFRSourceFactory, context_label: bytes, width: int
) -> int:
    """Choose a context-specific field symmetrically and independently of Q."""
    if width <= 0:
        raise ValueError("width must be positive")
    digest = hashlib.sha256(
        RANDOM_ANCHOR_DOMAIN
        + bytes(source_factory.private_key)
        + bytes(context_label)
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % int(width)


def _field_source(
    source_factory: PFRSourceFactory, context_label: bytes, field: int
) -> SharedPFRSource:
    return source_factory.build(max_order_field_label(context_label, field))


@torch.no_grad()
def keyed_uniform_fields(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    vocab_size: int,
    device,
    generator: Optional[torch.Generator] = None,
) -> FloatTensor:
    """Regenerate token-addressed keyed uniforms for the requested fields."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    field_ids = [int(field) for field in fields]
    if not field_ids:
        return torch.empty((0, vocab_size), device=device, dtype=torch.float32)
    if generator is None:
        generator = torch.Generator(device=device)
    rows = [
        _field_source(source_factory, context_label, field).uniform_noise(
            (1, vocab_size), device=device, generator=generator
        )[0]
        for field in field_ids
    ]
    return torch.stack(rows, dim=0)


def stratify_uniform_fields(
    raw_uniforms: FloatTensor,
    shift_uniforms: FloatTensor,
    *,
    fields: Sequence[int],
    width: int,
) -> FloatTensor:
    """Map iid jitters to cyclic Latin-hypercube uniform fields.

    At every token, ``floor(width * shift_uniform)`` assigns a random cyclic
    shift of the ``width`` strata.  Each requested field is marginally
    Uniform(0,1), while the complete field set contains exactly one value in
    each stratum.  Consequently ``width * min_b U_b`` is exactly Uniform.
    """
    field_values = [int(field) for field in fields]
    if width <= 0 or not field_values:
        raise ValueError("width and fields must be non-empty")
    if any(field < 0 or field >= int(width) for field in field_values):
        raise ValueError("Latin field IDs must lie in [0, width)")
    if raw_uniforms.dim() != 2 or raw_uniforms.shape[0] != len(field_values):
        raise ValueError("raw_uniforms must have shape [len(fields), tokens]")
    shift = shift_uniforms.reshape(-1).to(
        device=raw_uniforms.device, dtype=torch.float32
    )
    if shift.numel() != raw_uniforms.shape[1]:
        raise ValueError("one shift uniform is required per token")
    shift_stratum = torch.floor(shift * int(width)).to(torch.long).clamp_max(
        int(width) - 1
    )
    field_ids = torch.tensor(
        field_values, device=raw_uniforms.device, dtype=torch.long
    ).unsqueeze(1)
    strata = torch.remainder(field_ids + shift_stratum.unsqueeze(0), int(width))
    return (strata.float() + raw_uniforms.float()) / float(width)


def reverse_stratify_uniform_fields(
    base_uniforms: FloatTensor,
    shift_uniforms: FloatTensor,
    *,
    fields: Sequence[int],
    width: int,
) -> FloatTensor:
    """Latin fields with reverse matching to the minimum stratum.

    At B=2, before the random field-label swap, this constructs
    ``(V/2, 1-V/2)``.  Each labeled field is marginally Uniform and
    ``B*min_b U_b=V`` is exactly Uniform.  The definition also supports partial
    active field sets, which is required after speculative trajectories merge.
    """
    field_values = [int(field) for field in fields]
    if width <= 0 or not field_values:
        raise ValueError("width and fields must be non-empty")
    if any(field < 0 or field >= int(width) for field in field_values):
        raise ValueError("Latin field IDs must lie in [0, width)")
    base = base_uniforms.reshape(-1).float()
    shift = shift_uniforms.reshape(-1).to(device=base.device, dtype=torch.float32)
    if base.numel() != shift.numel():
        raise ValueError("base and shift require one value per token")
    shift_stratum = torch.floor(shift * int(width)).to(torch.long).clamp_max(
        int(width) - 1
    )
    field_ids = torch.tensor(
        field_values, device=base.device, dtype=torch.long
    ).unsqueeze(1)
    strata = torch.remainder(field_ids + shift_stratum.unsqueeze(0), int(width))
    jitter = torch.where(strata == 0, base.unsqueeze(0), 1.0 - base.unsqueeze(0))
    return (strata.float() + jitter) / float(width)


@torch.no_grad()
def keyed_latin_uniform_fields(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    width: int,
    vocab_size: int,
    device,
    generator: Optional[torch.Generator] = None,
) -> FloatTensor:
    """Dense keyed Latin fields; virtual field ``width`` stores the shift."""
    field_values = [int(field) for field in fields]
    raw = keyed_uniform_fields(
        source_factory=source_factory,
        context_label=context_label,
        fields=field_values + [int(width)],
        vocab_size=vocab_size,
        device=device,
        generator=generator,
    )
    return stratify_uniform_fields(
        raw[:-1], raw[-1], fields=field_values, width=int(width)
    )


@torch.no_grad()
def keyed_reverse_latin_uniform_fields(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    width: int,
    vocab_size: int,
    device,
    generator: Optional[torch.Generator] = None,
) -> FloatTensor:
    """Dense keyed reverse-matched Latin fields."""
    field_values = [int(field) for field in fields]
    # Virtual fields width and width+1 provide the label shift and shared base.
    raw = keyed_uniform_fields(
        source_factory=source_factory,
        context_label=context_label,
        fields=[int(width), int(width) + 1],
        vocab_size=vocab_size,
        device=device,
        generator=generator,
    )
    return reverse_stratify_uniform_fields(
        raw[1], raw[0], fields=field_values, width=int(width)
    )


@torch.no_grad()
def _counter_context_seed(
    source_factory: PFRSourceFactory, context_label: bytes
) -> int:
    return _field_source(
        source_factory, COUNTER_PRF_DOMAIN + bytes(context_label), 0
    ).seed()


@torch.no_grad()
def keyed_counter_uniforms_on_support(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    support: LongTensor,
    vocab_size: int,
    device,
) -> FloatTensor:
    """Token-addressable Philox PRF evaluated only on decoding support.

    The context/key determine the Philox seed and ``(field, token)`` determines
    its counter.  This directly implements the paper's PRF(k,c,b,y) interface
    and reduces top-k RNG work from ``B*|V|`` to ``B*k``.  It is a separate
    bit-level backend from PyTorch's sequential dense RNG stream.
    """
    if triton is None:
        raise RuntimeError("counter_philox backend requires Triton")
    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("counter_philox backend currently requires CUDA")
    token_ids = support.reshape(-1).to(device=device, dtype=torch.int64).contiguous()
    field_ids = torch.tensor(
        [int(field) for field in fields], device=device, dtype=torch.int64
    )
    if token_ids.numel() == 0 or field_ids.numel() == 0:
        raise ValueError("counter PRF requires non-empty fields and support")
    seed = _counter_context_seed(source_factory, context_label)
    output = torch.empty(
        (int(field_ids.numel()), int(token_ids.numel())),
        device=device,
        dtype=torch.float32,
    )
    total = int(output.numel())
    block = triton.next_power_of_2(total)
    _counter_uniform_kernel[(1,)](
        token_ids,
        field_ids,
        output,
        seed=int(seed),
        vocab_size=int(vocab_size),
        n_tokens=int(token_ids.numel()),
        n_fields=int(field_ids.numel()),
        BLOCK=block,
    )
    return output


@torch.no_grad()
def keyed_counter_latin_uniforms_on_support(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    width: int,
    support: LongTensor,
    vocab_size: int,
    device,
) -> FloatTensor:
    """Counter-PRF Latin fields evaluated only on the decoding support."""
    field_values = [int(field) for field in fields]
    raw = keyed_counter_uniforms_on_support(
        source_factory=source_factory,
        context_label=context_label,
        fields=field_values + [int(width)],
        support=support,
        vocab_size=vocab_size,
        device=device,
    )
    return stratify_uniform_fields(
        raw[:-1], raw[-1], fields=field_values, width=int(width)
    )


@torch.no_grad()
def keyed_counter_reverse_latin_uniforms_on_support(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    width: int,
    support: LongTensor,
    vocab_size: int,
    device,
) -> FloatTensor:
    """Counter-PRF reverse-matched Latin fields on decoding support."""
    raw = keyed_counter_uniforms_on_support(
        source_factory=source_factory,
        context_label=context_label,
        fields=[int(width), int(width) + 1],
        support=support,
        vocab_size=vocab_size,
        device=device,
    )
    return reverse_stratify_uniform_fields(
        raw[1], raw[0], fields=fields, width=int(width)
    )


@torch.no_grad()
def counter_draft_select_support(
    *,
    processed_logits: FloatTensor,
    support: LongTensor,
    fields: Sequence[int],
    source_factory: PFRSourceFactory,
    context_label: bytes,
    compact_logits: bool = False,
    vocab_size: Optional[int] = None,
) -> LongTensor:
    """Fused token-addressable PRF and one PF race per draft field."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter PF requires Triton/CUDA")
    token_ids = support.reshape(-1).to(
        device=processed_logits.device, dtype=torch.int64
    ).contiguous()
    field_ids = torch.tensor(
        [int(field) for field in fields],
        device=processed_logits.device,
        dtype=torch.int64,
    )
    output = torch.empty_like(field_ids)
    block = triton.next_power_of_2(int(token_ids.numel()))
    effective_vocab_size = (
        int(vocab_size) if vocab_size is not None else int(processed_logits.shape[-1])
    )
    _counter_draft_select_kernel[(int(field_ids.numel()),)](
        processed_logits,
        token_ids,
        field_ids,
        output,
        seed=_counter_context_seed(source_factory, context_label),
        vocab_size=effective_vocab_size,
        n_tokens=int(token_ids.numel()),
        compact_logits=bool(compact_logits),
        BLOCK=block,
    )
    return output


@torch.no_grad()
def counter_latin_draft_select_support(
    *,
    processed_logits: FloatTensor,
    support: LongTensor,
    fields: Sequence[int],
    width: int,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    compact_logits: bool = False,
    vocab_size: Optional[int] = None,
    processed_weights: Optional[FloatTensor] = None,
) -> LongTensor:
    """Fused counter RNG, independent Latin stratification, and draft races."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter Latin PF requires Triton/CUDA")
    field_values = [int(field) for field in fields]
    if not field_values or any(field < 0 or field >= int(width) for field in field_values):
        raise ValueError("Latin draft fields must lie in [0,width)")
    token_ids = support.reshape(-1).to(
        device=processed_logits.device, dtype=torch.int64
    ).contiguous()
    field_ids = torch.tensor(
        field_values, device=processed_logits.device, dtype=torch.int64
    )
    output = torch.empty_like(field_ids)
    effective_vocab_size = (
        int(vocab_size) if vocab_size is not None else int(processed_logits.shape[-1])
    )
    if processed_weights is None:
        logits_f = processed_logits.float()
        processed_weights = torch.exp(logits_f - logits_f.max())
    _counter_latin_draft_select_kernel[(len(field_values),)](
        processed_weights,
        token_ids,
        field_ids,
        output,
        seed=_counter_context_seed(source_factory, context_label),
        vocab_size=effective_vocab_size,
        n_tokens=int(token_ids.numel()),
        width=int(width),
        compact_logits=bool(compact_logits),
        BLOCK=triton.next_power_of_2(int(token_ids.numel())),
    )
    return output


@torch.no_grad()
def counter_latin_draft_select_batch_support(
    *,
    processed_logits: FloatTensor,
    supports: LongTensor,
    context_rows: Sequence[int],
    fields: Sequence[int],
    context_seeds: Sequence[int],
    width: int,
    vocab_size: int,
    processed_weights: Optional[FloatTensor] = None,
) -> LongTensor:
    """Fused compact Latin draft races for one complete tree depth."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter Latin PF requires Triton/CUDA")
    if tuple(processed_logits.shape) != tuple(supports.shape):
        raise ValueError("compact logits and supports must have equal shape")
    n_trajectories = len(fields)
    if not (
        n_trajectories > 0
        and len(context_rows) == n_trajectories
        and len(context_seeds) == n_trajectories
    ):
        raise ValueError("trajectory rows, fields, and seeds must align")
    if any(field < 0 or field >= int(width) for field in fields):
        raise ValueError("Latin draft fields must lie in [0,width)")
    device = processed_logits.device
    rows = torch.tensor(context_rows, device=device, dtype=torch.int64)
    field_ids = torch.tensor(fields, device=device, dtype=torch.int64)
    seeds = torch.tensor(context_seeds, device=device, dtype=torch.int64)
    output = torch.empty(n_trajectories, device=device, dtype=torch.int64)
    n_tokens = int(supports.shape[1])
    if processed_weights is None:
        logits_f = processed_logits.float()
        processed_weights = torch.exp(
            logits_f - logits_f.amax(dim=-1, keepdim=True)
        )
    _counter_latin_draft_select_batch_kernel[(n_trajectories,)](
        processed_weights.contiguous(),
        supports.to(device=device, dtype=torch.int64).contiguous(),
        rows,
        field_ids,
        seeds,
        output,
        vocab_size=int(vocab_size),
        n_tokens=n_tokens,
        width=int(width),
        BLOCK=triton.next_power_of_2(n_tokens),
    )
    return output


@torch.no_grad()
def counter_latin_target_select_support(
    *,
    processed_logits: FloatTensor,
    support: LongTensor,
    width: int,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    exact_pivot: bool = True,
    compact_logits: bool = False,
    vocab_size: Optional[int] = None,
    context_seed: Optional[int] = None,
) -> Tuple[int, FloatTensor]:
    """Fused counter RNG, independent Latin aggregate, and target race."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter Latin PF requires Triton/CUDA")
    if width <= 0:
        raise ValueError("width must be positive")
    token_ids = support.reshape(-1).to(
        device=processed_logits.device, dtype=torch.int64
    ).contiguous()
    output_token = torch.empty((), device=processed_logits.device, dtype=torch.int64)
    output_pivot = torch.empty((), device=processed_logits.device, dtype=torch.float32)
    effective_vocab_size = (
        int(vocab_size) if vocab_size is not None else int(processed_logits.shape[-1])
    )
    _counter_latin_target_select_kernel[(1,)](
        processed_logits,
        token_ids,
        output_token,
        output_pivot,
        seed=(
            int(context_seed)
            if context_seed is not None
            else _counter_context_seed(source_factory, context_label)
        ),
        vocab_size=effective_vocab_size,
        n_tokens=int(token_ids.numel()),
        width=int(width),
        compact_logits=bool(compact_logits),
        BLOCK=triton.next_power_of_2(int(token_ids.numel())),
    )
    token = int(output_token.item())
    if exact_pivot:
        token_uniforms = keyed_counter_latin_uniforms_on_support(
            source_factory=source_factory,
            context_label=context_label,
            fields=range(int(width)),
            width=int(width),
            support=torch.tensor([token], device=processed_logits.device),
            vocab_size=effective_vocab_size,
            device=processed_logits.device,
        )
        output_pivot = aggregate_latin_uniform(token_uniforms)[0]
    return token, output_pivot


@torch.no_grad()
def counter_target_select_support(
    *,
    processed_logits: FloatTensor,
    support: LongTensor,
    fields: Sequence[int],
    source_factory: PFRSourceFactory,
    context_label: bytes,
    exact_pivot: bool = True,
    compact_logits: bool = False,
    vocab_size: Optional[int] = None,
) -> Tuple[int, FloatTensor]:
    """Fused token-addressable PRF, max-order transform, and target race."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter PF requires Triton/CUDA")
    token_ids = support.reshape(-1).to(
        device=processed_logits.device, dtype=torch.int64
    ).contiguous()
    field_values = [int(field) for field in fields]
    if not field_values or field_values != list(
        range(field_values[0], field_values[0] + len(field_values))
    ):
        raise ValueError("fused target fields must be non-empty and contiguous")
    output_token = torch.empty((), device=processed_logits.device, dtype=torch.int64)
    output_pivot = torch.empty((), device=processed_logits.device, dtype=torch.float32)
    block = triton.next_power_of_2(int(token_ids.numel()))
    effective_vocab_size = (
        int(vocab_size) if vocab_size is not None else int(processed_logits.shape[-1])
    )
    _counter_target_select_kernel[(1,)](
        processed_logits,
        token_ids,
        output_token,
        output_pivot,
        seed=_counter_context_seed(source_factory, context_label),
        field_offset=field_values[0],
        vocab_size=effective_vocab_size,
        n_tokens=int(token_ids.numel()),
        n_fields=len(field_values),
        compact_logits=bool(compact_logits),
        BLOCK=block,
    )
    token = int(output_token.item())
    if exact_pivot:
        token_uniforms = keyed_counter_uniforms_on_support(
            source_factory=source_factory,
            context_label=context_label,
            fields=fields,
            support=torch.tensor([token], device=processed_logits.device),
            vocab_size=effective_vocab_size,
            device=processed_logits.device,
        )
        output_pivot = aggregate_min_uniform(token_uniforms)[0]
    return token, output_pivot


@torch.no_grad()
def counter_target_select_batch_support(
    *,
    processed_logits: FloatTensor,
    supports: LongTensor,
    context_seeds: Sequence[int],
    width: int,
) -> LongTensor:
    """Batch max-order PF target selection over known context rows.

    The returned token for each row is bit-identical to calling
    :func:`counter_target_select_support` with fields ``range(width)`` and
    ``exact_pivot=False`` on that row.  Pivots are intentionally not produced:
    the timed benchmark recovers them after generation, while the
    correctness-first path continues to use the scalar implementation.
    """
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter PF requires Triton/CUDA")
    if processed_logits.dim() != 2 or supports.dim() != 2:
        raise ValueError("processed_logits and supports must be rank-2")
    if processed_logits.shape[0] != supports.shape[0]:
        raise ValueError("processed logits and supports must have equal rows")
    if len(context_seeds) != int(processed_logits.shape[0]):
        raise ValueError("one context seed is required per row")
    if width <= 0:
        raise ValueError("width must be positive")
    token_ids = supports.to(
        device=processed_logits.device, dtype=torch.int64
    ).contiguous()
    seeds = torch.tensor(
        [int(seed) for seed in context_seeds],
        device=processed_logits.device,
        dtype=torch.int64,
    )
    output_tokens = torch.empty(
        (int(processed_logits.shape[0]),),
        device=processed_logits.device,
        dtype=torch.int64,
    )
    n_tokens = int(token_ids.shape[1])
    block = triton.next_power_of_2(n_tokens)
    _counter_target_select_batch_kernel[(int(processed_logits.shape[0]),)](
        processed_logits.contiguous(),
        token_ids,
        seeds,
        output_tokens,
        vocab_size=int(processed_logits.shape[1]),
        n_tokens=n_tokens,
        n_fields=int(width),
        BLOCK=block,
    )
    return output_tokens


@torch.no_grad()
def counter_target_traverse_support(
    *,
    processed_logits: FloatTensor,
    supports: LongTensor,
    context_seeds: Sequence[int],
    child_tokens: LongTensor,
    child_rows: LongTensor,
    width: int,
    draft_steps: int,
    max_outputs: int,
    eos_token: int,
) -> Tuple[List[int], List[int], int, int]:
    """Select and traverse one max-order target block with one synchronization.

    All model logits have already been produced by the ordinary batched target
    forward.  This function only fuses the same per-context PF reduction and
    the deterministic draft-membership traversal previously performed by
    Python.  It returns emitted tokens, their context-row indices, the accepted
    draft count, and the final accepted context row.
    """
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter PF requires Triton/CUDA")
    n_contexts = int(processed_logits.shape[0])
    if supports.shape[0] != n_contexts or len(context_seeds) != n_contexts:
        raise ValueError("target context batches are not aligned")
    if tuple(child_tokens.shape) != (n_contexts, int(width)):
        raise ValueError("child_tokens has the wrong shape")
    if tuple(child_rows.shape) != (n_contexts, int(width)):
        raise ValueError("child_rows has the wrong shape")
    if not (1 <= int(max_outputs) <= int(draft_steps) + 1):
        raise ValueError("max_outputs must be in [1, draft_steps + 1]")

    device = processed_logits.device
    token_ids = supports.to(device=device, dtype=torch.int64).contiguous()
    seeds = torch.tensor(context_seeds, device=device, dtype=torch.int64)
    child_tokens = child_tokens.to(device=device, dtype=torch.int64).contiguous()
    child_rows = child_rows.to(device=device, dtype=torch.int64).contiguous()
    packed = torch.empty(2 * int(max_outputs) + 3, device=device, dtype=torch.int64)
    n_tokens = int(token_ids.shape[1])
    _counter_target_traverse_kernel[(1,)](
        processed_logits.contiguous(),
        token_ids,
        seeds,
        child_tokens,
        child_rows,
        packed,
        eos_token=int(eos_token),
        vocab_size=int(processed_logits.shape[1]),
        n_tokens=n_tokens,
        n_fields=int(width),
        draft_steps=int(draft_steps),
        max_outputs=int(max_outputs),
        BLOCK=triton.next_power_of_2(n_tokens),
    )
    values = packed.detach().cpu().tolist()
    generated = int(values[2 * max_outputs])
    accepted = int(values[2 * max_outputs + 1])
    final_row = int(values[2 * max_outputs + 2])
    return (
        [int(token) for token in values[:generated]],
        [int(row) for row in values[max_outputs : max_outputs + generated]],
        accepted,
        final_row,
    )


@torch.no_grad()
def counter_latin_target_traverse_support(
    *,
    processed_logits: FloatTensor,
    supports: LongTensor,
    context_seeds: Sequence[int],
    child_tokens: LongTensor,
    child_rows: LongTensor,
    width: int,
    vocab_size: int,
    draft_steps: int,
    max_outputs: int,
    eos_token: int,
) -> Tuple[List[int], List[int], int, int]:
    """Fuse compact Latin target races and deterministic tree traversal."""
    if triton is None or processed_logits.device.type != "cuda":
        raise RuntimeError("fused counter Latin PF requires Triton/CUDA")
    n_contexts = int(processed_logits.shape[0])
    if tuple(processed_logits.shape) != tuple(supports.shape):
        raise ValueError("compact Latin logits and supports must have equal shape")
    if len(context_seeds) != n_contexts:
        raise ValueError("one context seed is required per target row")
    if tuple(child_tokens.shape) != (n_contexts, int(width)):
        raise ValueError("child_tokens has the wrong shape")
    if tuple(child_rows.shape) != (n_contexts, int(width)):
        raise ValueError("child_rows has the wrong shape")
    if not (1 <= int(max_outputs) <= int(draft_steps) + 1):
        raise ValueError("max_outputs must be in [1,draft_steps+1]")
    device = processed_logits.device
    token_ids = supports.to(device=device, dtype=torch.int64).contiguous()
    seeds = torch.tensor(context_seeds, device=device, dtype=torch.int64)
    child_tokens = child_tokens.to(device=device, dtype=torch.int64).contiguous()
    child_rows = child_rows.to(device=device, dtype=torch.int64).contiguous()
    packed = torch.empty(2 * int(max_outputs) + 3, device=device, dtype=torch.int64)
    n_tokens = int(token_ids.shape[1])
    _counter_latin_target_traverse_kernel[(1,)](
        processed_logits.contiguous(),
        token_ids,
        seeds,
        child_tokens,
        child_rows,
        packed,
        eos_token=int(eos_token),
        vocab_size=int(vocab_size),
        n_tokens=n_tokens,
        width=int(width),
        draft_steps=int(draft_steps),
        max_outputs=int(max_outputs),
        BLOCK=triton.next_power_of_2(n_tokens),
    )
    values = packed.detach().cpu().tolist()
    generated = int(values[2 * max_outputs])
    accepted = int(values[2 * max_outputs + 1])
    final_row = int(values[2 * max_outputs + 2])
    return (
        [int(token) for token in values[:generated]],
        [int(row) for row in values[max_outputs : max_outputs + generated]],
        accepted,
        final_row,
    )


@torch.no_grad()
def _keyed_uniform_fields_with_cache(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    vocab_size: int,
    device,
    generator: torch.Generator,
    cached_rows: Optional[Dict[int, FloatTensor]],
) -> FloatTensor:
    """Assemble fields while regenerating only rows absent from draft cache."""
    field_ids = [int(field) for field in fields]
    cached_rows = cached_rows or {}
    valid_cached = {
        field: row
        for field, row in cached_rows.items()
        if row.device == torch.device(device)
        and int(row.shape[-1]) == int(vocab_size)
    }
    missing = [
        field
        for field in field_ids
        if field not in valid_cached
    ]
    generated = keyed_uniform_fields(
        source_factory=source_factory,
        context_label=context_label,
        fields=missing,
        vocab_size=vocab_size,
        device=device,
        generator=generator,
    )
    generated_by_field = {
        field: generated[pos] for pos, field in enumerate(missing)
    }
    return torch.stack(
        [
            valid_cached[field]
            if field in valid_cached
            else generated_by_field[field]
            for field in field_ids
        ],
        dim=0,
    )


@torch.no_grad()
def _keyed_uniforms_on_support(
    *,
    source_factory: PFRSourceFactory,
    context_label: bytes,
    fields: Sequence[int],
    support: LongTensor,
    vocab_size: int,
    device,
    generator: torch.Generator,
    cached_rows: Optional[Dict[int, FloatTensor]] = None,
    retain_generated_rows: bool = False,
) -> Tuple[FloatTensor, Dict[int, FloatTensor]]:
    """Generate the original RNG rows but only materialise top-k race input.

    This avoids the otherwise unnecessary ``stack`` copy into a dense
    ``[B, |V|]`` tensor.  Returned full rows are optional and are used only by
    the draft phase so target verification can reuse the exact same fields.
    """
    token_ids = support.reshape(-1).to(device=device, dtype=torch.long)
    if token_ids.numel() == 0:
        raise ValueError("PF support cannot be empty")
    cached_rows = cached_rows or {}
    support_rows: List[FloatTensor] = []
    retained: Dict[int, FloatTensor] = {}
    for field_value in fields:
        field = int(field_value)
        row = cached_rows.get(field)
        if (
            row is None
            or row.device != torch.device(device)
            or int(row.shape[-1]) != int(vocab_size)
        ):
            row = _field_source(
                source_factory, context_label, field
            ).uniform_noise(
                (1, int(vocab_size)), device=device, generator=generator
            )[0]
            if retain_generated_rows:
                retained[field] = row
        elif retain_generated_rows:
            retained[field] = row
        support_rows.append(row.index_select(0, token_ids))
    return torch.stack(support_rows, dim=0), retained


def _normalized_pf_weights(processed_logits: FloatTensor) -> FloatTensor:
    """Return alpha=exp(logit-max(logit)) with masked entries equal to zero."""
    if processed_logits.dim() != 1:
        raise ValueError("processed_logits must be one-dimensional")
    finite = torch.isfinite(processed_logits)
    if not bool(finite.any()):
        raise ValueError("processed logits have empty finite support")
    max_logit = processed_logits[finite].float().max()
    weights = torch.zeros_like(processed_logits, dtype=torch.float32)
    weights[finite] = torch.exp(processed_logits[finite].float() - max_logit)
    return weights


@torch.no_grad()
def uniform_race_select(
    processed_logits: FloatTensor, uniforms: FloatTensor
) -> LongTensor:
    """Select one PF token per uniform row using ``argmin U/alpha``."""
    if uniforms.dim() == 1:
        uniforms = uniforms.unsqueeze(0)
    if uniforms.dim() != 2:
        raise ValueError("uniforms must have shape [fields, vocab]")
    if uniforms.shape[-1] != processed_logits.shape[-1]:
        raise ValueError("uniform and logits vocabulary sizes differ")
    weights = _normalized_pf_weights(processed_logits)
    ratios = torch.where(
        weights.unsqueeze(0) > 0,
        uniforms.float() / weights.unsqueeze(0),
        torch.full_like(uniforms, float("inf"), dtype=torch.float32),
    )
    return ratios.argmin(dim=-1).to(torch.long)


@torch.no_grad()
def uniform_race_select_support(
    processed_logits: FloatTensor,
    uniforms: FloatTensor,
    support: Optional[LongTensor],
) -> LongTensor:
    """Exact PF race restricted to the finite decoding support.

    ``process_logits_exact(..., return_support=True)`` returns the indices
    retained by top-k.  Masked tokens have zero PF weight, so materialising
    their ratios cannot affect the winner.  We still generate the original
    vocabulary-sized keyed uniform rows, preserving the RNG stream and hence
    bit-exact winners/pivots; only the subsequent arithmetic is sparse.
    """
    if support is None:
        return uniform_race_select(processed_logits, uniforms)
    if uniforms.dim() == 1:
        uniforms = uniforms.unsqueeze(0)
    token_ids = support.reshape(-1).to(device=uniforms.device, dtype=torch.long)
    if token_ids.numel() == 0:
        raise ValueError("PF support cannot be empty")
    support_logits = processed_logits.index_select(0, token_ids).float()
    max_logit = support_logits.max()
    weights = torch.exp(support_logits - max_logit)
    support_uniforms = uniforms.index_select(1, token_ids).float()
    winners = (support_uniforms / weights.unsqueeze(0)).argmin(dim=-1)
    return token_ids.index_select(0, winners).to(torch.long)


@torch.no_grad()
def _uniform_race_select_compact(
    processed_logits: FloatTensor,
    support_uniforms: FloatTensor,
    support: LongTensor,
    *,
    compact_logits: bool = False,
) -> LongTensor:
    """PF winners from uniforms already gathered on ``support``."""
    token_ids = support.reshape(-1).to(
        device=support_uniforms.device, dtype=torch.long
    )
    support_logits = (
        processed_logits.float()
        if compact_logits
        else processed_logits.index_select(0, token_ids).float()
    )
    if support_logits.numel() != token_ids.numel():
        raise ValueError("compact logits and support must have equal length")
    weights = torch.exp(support_logits - support_logits.max())
    winners = (support_uniforms.float() / weights.unsqueeze(0)).argmin(dim=-1)
    return token_ids.index_select(0, winners).to(torch.long)


@torch.no_grad()
def _aggregate_select_support(
    processed_logits: FloatTensor,
    uniforms: FloatTensor,
    support: Optional[LongTensor],
) -> Tuple[int, FloatTensor]:
    """Select target PF token and return only its aggregate detector pivot."""
    if support is None:
        token, aggregate = max_order_pf_select(processed_logits, uniforms)
        return token, aggregate[token]
    token_ids = support.reshape(-1).to(device=uniforms.device, dtype=torch.long)
    support_logits = processed_logits.index_select(0, token_ids).float()
    support_uniforms = uniforms.index_select(1, token_ids).float()
    aggregate = aggregate_min_uniform(support_uniforms)
    max_logit = support_logits.max()
    weights = torch.exp(support_logits - max_logit)
    winner = int((aggregate / weights).argmin().item())
    return int(token_ids[winner].item()), aggregate[winner]


@torch.no_grad()
def _aggregate_select_compact(
    processed_logits: FloatTensor,
    support_uniforms: FloatTensor,
    support: LongTensor,
) -> Tuple[int, FloatTensor]:
    """Target PF winner from compact ``[B, k]`` keyed uniforms."""
    token_ids = support.reshape(-1).to(
        device=support_uniforms.device, dtype=torch.long
    )
    support_logits = processed_logits.index_select(0, token_ids).float()
    aggregate = aggregate_min_uniform(support_uniforms)
    weights = torch.exp(support_logits - support_logits.max())
    winner = int((aggregate / weights).argmin().item())
    return int(token_ids[winner].item()), aggregate[winner]


def aggregate_min_uniform(uniforms: FloatTensor, width: Optional[int] = None) -> FloatTensor:
    """Probability-integral transform of the fieldwise minimum.

    If ``R=min_b U_b`` for B iid uniforms, then
    ``1-(1-R)^B`` is exactly uniform.  The direct integer-power expression is
    cheap on GPU and stable in the range relevant to vocabulary-sized races;
    clamping only protects the downstream division from exact zero.
    """
    if uniforms.dim() != 2 or uniforms.shape[0] <= 0:
        raise ValueError("uniforms must have non-empty shape [fields, vocab]")
    b = int(uniforms.shape[0] if width is None else width)
    if b != int(uniforms.shape[0]):
        raise ValueError("width must equal the number of uniform fields")
    minimum = uniforms.float().amin(dim=0)
    aggregate = 1.0 - torch.pow(1.0 - minimum, b)
    return aggregate.clamp(min=torch.finfo(torch.float32).tiny, max=1.0)


def aggregate_latin_uniform(
    uniforms: FloatTensor, width: Optional[int] = None
) -> FloatTensor:
    """Exact target pivot for a complete Latin-hypercube field set."""
    if uniforms.dim() != 2 or uniforms.shape[0] <= 0:
        raise ValueError("uniforms must have non-empty shape [fields, vocab]")
    b = int(uniforms.shape[0] if width is None else width)
    if b != int(uniforms.shape[0]):
        raise ValueError("Latin aggregation requires all width fields")
    aggregate = float(b) * uniforms.float().amin(dim=0)
    return aggregate.clamp(min=torch.finfo(torch.float32).tiny, max=1.0)


@torch.no_grad()
def _latin_select_compact(
    processed_logits: FloatTensor,
    support_uniforms: FloatTensor,
    support: LongTensor,
    *,
    compact_logits: bool = False,
) -> Tuple[int, FloatTensor]:
    token_ids = support.reshape(-1).to(
        device=support_uniforms.device, dtype=torch.long
    )
    support_logits = (
        processed_logits.float()
        if compact_logits
        else processed_logits.index_select(0, token_ids).float()
    )
    if support_logits.numel() != token_ids.numel():
        raise ValueError("compact logits and support must have equal length")
    aggregate = aggregate_latin_uniform(support_uniforms)
    weights = torch.exp(support_logits - support_logits.max())
    winner = int((aggregate / weights).argmin().item())
    return int(token_ids[winner].item()), aggregate[winner]


@torch.no_grad()
def max_order_pf_select(
    processed_logits: FloatTensor, uniforms: FloatTensor
) -> Tuple[int, FloatTensor]:
    """Select the fixed-B target PF token and return its aggregate field."""
    aggregate = aggregate_min_uniform(uniforms)
    token = int(uniform_race_select(processed_logits, aggregate)[0].item())
    return token, aggregate


@dataclass
class MaxOrderPFTree:
    levels: List[List[ContextKey]]
    active_fields: Dict[ContextKey, Tuple[int, ...]]
    draft_tokens: Dict[ContextKey, set[int]]
    draft_tree_size: int
    draft_forward_calls: int
    attempted_draft_tokens: int
    # Dense rows are retained only until target verification.  Reusing them
    # avoids regenerating identical keyed fields on accepted draft contexts.
    uniforms_by_context: Dict[ContextKey, Dict[int, FloatTensor]]
    # Context labels are reused by target selection.  Cache the exact bytes
    # produced during drafting instead of serializing the full prefix twice.
    labels_by_context: Dict[ContextKey, bytes]


@dataclass
class MaxOrderPFBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    output_labels: List[bytes]
    output_pivots: FloatTensor
    accepted_count: int
    attempted_draft_tokens: int
    draft_tree_size: int
    target_context_count: int
    target_forward_calls: int
    draft_forward_calls: int
    got_eos: bool
    target_past_key_values: Any
    draft_past_key_values: Any = None


def _as_eos_set(eos_token_id) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token) for token in eos_token_id}
    return {int(eos_token_id)}


@torch.no_grad()
def build_max_order_pf_tree_cached(
    *,
    ref_model,
    root: ContextKey,
    lookahead: int,
    width: int,
    source_factory: PFRSourceFactory,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
    past_key_values: Any = None,
    rng_backend: str = "torch_dense",
    field_coupling: str = "iid",
    fuse_latin_sampling: bool = True,
    batch_latin_draft_sampling: bool = False,
) -> Tuple[MaxOrderPFTree, Dict[ContextKey, Any]]:
    """Build B PF trajectories, merging identical prefixes."""
    if lookahead <= 0 or width <= 0:
        raise ValueError("lookahead and width must be positive")

    draft_device = _model_device(ref_model)
    draft_temp = _temperature_for_draft(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)
    cached_n = cache_len(past_key_values)
    generator = torch.Generator(device=draft_device)
    if rng_backend not in {"torch_dense", "counter_philox"}:
        raise ValueError(f"unknown rng_backend: {rng_backend}")
    if field_coupling not in {"iid", "latin_hypercube", "latin_reverse"}:
        raise ValueError(f"unknown field_coupling: {field_coupling}")

    levels: List[List[ContextKey]] = [[root]]
    active_fields: Dict[ContextKey, Tuple[int, ...]] = {
        root: tuple(range(int(width)))
    }
    draft_tokens: Dict[ContextKey, set[int]] = {}
    past_by_context: Dict[ContextKey, Any] = {}
    draft_tree_size = 0
    draft_forward_calls = 0
    attempted_draft_tokens = 0
    uniforms_by_context: Dict[ContextKey, Dict[int, FloatTensor]] = {}
    labels_by_context: Dict[ContextKey, bytes] = {}
    level_cache: Any = None

    for depth in range(1, int(lookahead) + 1):
        prev_level = levels[-1]
        if len(prev_level) > width:
            raise RuntimeError("merged PF level unexpectedly exceeds width")
        if depth == 1:
            if cached_n > 0:
                batch_past = _repeat_cache(past_key_values, len(prev_level))
                input_tokens = _batch_from_contexts(
                    [context[cached_n:] for context in prev_level], draft_device
                )
            else:
                batch_past = None
                input_tokens = _batch_from_contexts(prev_level, draft_device)
        else:
            previous_parents = levels[-2]
            parent_rows = {context: row for row, context in enumerate(previous_parents)}
            parent_indices = [
                parent_rows[context[:-1]] for context in prev_level
            ]
            # Once B trajectories have split, their row order remains aligned
            # with their parents unless a merged prefix splits again.  Avoid a
            # full-layer KV ``index_select`` for the common identity mapping.
            # Reusing the same immutable legacy cache is bit-exact.
            if parent_indices == list(range(len(prev_level))):
                batch_past = level_cache
            else:
                parent_idx = torch.tensor(
                    parent_indices,
                    device=draft_device,
                    dtype=torch.long,
                )
                batch_past = _gather_cache_rows(level_cache, parent_idx)
            input_tokens = torch.tensor(
                [[int(context[-1])] for context in prev_level],
                device=draft_device,
                dtype=torch.long,
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
        for row, context in enumerate(prev_level):
            past_by_context[context] = _select_cache_row(level_cache, row)

        raw_logits = out.logits[:, -1, :]
        if max_vocab_size is not None and max_vocab_size < raw_logits.shape[-1]:
            raw_logits = raw_logits[..., :max_vocab_size]
        compact_counter = (
            rng_backend == "counter_philox"
            and 0 < int(top_k) < int(raw_logits.shape[-1])
        )
        if compact_counter:
            processed, supports = _compact_topk_logits(
                raw_logits, top_k=top_k, temperature=draft_temp
            )
        else:
            processed, supports = process_logits_exact(
                raw_logits,
                temperature=draft_temp,
                top_k=top_k,
                top_p=top_p,
                return_support=True,
            )
        latin_processed_weights: Optional[FloatTensor] = None
        if (
            field_coupling == "latin_hypercube"
            and fuse_latin_sampling
            and rng_backend == "counter_philox"
            and supports is not None
        ):
            processed_f = processed.float()
            latin_processed_weights = torch.exp(
                processed_f - processed_f.amax(dim=-1, keepdim=True)
            )

        children: Dict[ContextKey, List[int]] = {}
        pending_winners: List[
            Tuple[ContextKey, Tuple[int, ...], LongTensor]
        ] = []
        latin_batch_pending: List[
            Tuple[ContextKey, Tuple[int, ...], int, bytes]
        ] = []
        for row, context in enumerate(prev_level):
            selected: Optional[LongTensor] = None
            fields = active_fields[context]
            attempted_draft_tokens += len(fields)
            context_label = max_order_context_label(context, width)
            labels_by_context[context] = context_label
            support = None if supports is None else supports[row]
            if field_coupling in {"latin_hypercube", "latin_reverse"}:
                dense_latin = (
                    keyed_reverse_latin_uniform_fields
                    if field_coupling == "latin_reverse"
                    else keyed_latin_uniform_fields
                )
                counter_latin = (
                    keyed_counter_reverse_latin_uniforms_on_support
                    if field_coupling == "latin_reverse"
                    else keyed_counter_latin_uniforms_on_support
                )
                if support is None:
                    uniforms = dense_latin(
                        source_factory=source_factory,
                        context_label=context_label,
                        fields=fields,
                        width=width,
                        vocab_size=int(processed.shape[-1]),
                        device=draft_device,
                        generator=generator,
                    )
                    selected = uniform_race_select(processed[row], uniforms)
                elif rng_backend == "counter_philox":
                    if (
                        field_coupling == "latin_hypercube"
                        and fuse_latin_sampling
                        and compact_counter
                        and batch_latin_draft_sampling
                    ):
                        latin_batch_pending.append(
                            (context, fields, row, context_label)
                        )
                    elif field_coupling == "latin_hypercube" and fuse_latin_sampling:
                        selected = counter_latin_draft_select_support(
                            processed_logits=processed[row], support=support,
                            fields=fields, width=width,
                            source_factory=source_factory,
                            context_label=context_label,
                            compact_logits=compact_counter,
                            vocab_size=max_vocab_size,
                            processed_weights=latin_processed_weights[row],
                        )
                    else:
                        support_uniforms = counter_latin(
                            source_factory=source_factory,
                            context_label=context_label,
                            fields=fields,
                            width=width,
                            support=support,
                            vocab_size=int(max_vocab_size or processed.shape[-1]),
                            device=draft_device,
                        )
                        selected = _uniform_race_select_compact(
                            processed[row],
                            support_uniforms,
                            support,
                            compact_logits=compact_counter,
                        )
                else:
                    uniforms = dense_latin(
                        source_factory=source_factory,
                        context_label=context_label,
                        fields=fields,
                        width=width,
                        vocab_size=int(processed.shape[-1]),
                        device=draft_device,
                        generator=generator,
                    )
                    selected = uniform_race_select_support(
                        processed[row], uniforms, support
                    )
                # Latin target aggregation requires all fields, so partial
                # active-field rows are deliberately regenerated at verify.
                uniforms_by_context[context] = {}
            elif support is None:
                uniforms = keyed_uniform_fields(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    vocab_size=int(processed.shape[-1]),
                    device=draft_device,
                    generator=generator,
                )
                uniforms_by_context[context] = {
                    int(field): uniforms[pos]
                    for pos, field in enumerate(fields)
                }
                selected = uniform_race_select(processed[row], uniforms)
            else:
                if rng_backend == "counter_philox":
                    selected = counter_draft_select_support(
                        processed_logits=processed[row],
                        support=support,
                        fields=fields,
                        source_factory=source_factory,
                        context_label=context_label,
                        compact_logits=compact_counter,
                        vocab_size=max_vocab_size,
                    )
                    retained = {}
                else:
                    support_uniforms, retained = _keyed_uniforms_on_support(
                        source_factory=source_factory,
                        context_label=context_label,
                        fields=fields,
                        support=support,
                        vocab_size=int(processed.shape[-1]),
                        device=draft_device,
                        generator=generator,
                        retain_generated_rows=True,
                    )
                    selected = _uniform_race_select_compact(
                        processed[row], support_uniforms, support
                    )
                uniforms_by_context[context] = retained
            if selected is not None:
                pending_winners.append((context, fields, selected))

        if latin_batch_pending:
            flat_rows: List[int] = []
            flat_fields: List[int] = []
            flat_seeds: List[int] = []
            for _context, fields, row, context_label in latin_batch_pending:
                for field in fields:
                    flat_rows.append(int(row))
                    flat_fields.append(int(field))
                    flat_seeds.append(
                        _counter_context_seed(source_factory, context_label)
                    )
            flat_selected = counter_latin_draft_select_batch_support(
                processed_logits=processed,
                supports=supports,
                context_rows=flat_rows,
                fields=flat_fields,
                context_seeds=flat_seeds,
                width=width,
                vocab_size=int(max_vocab_size or processed.shape[-1]),
                processed_weights=latin_processed_weights,
            )
            offset = 0
            for context, fields, _row, _label in latin_batch_pending:
                count = len(fields)
                pending_winners.append(
                    (context, fields, flat_selected[offset : offset + count])
                )
                offset += count

        # One device synchronization per draft depth.  Calling `.item()` for
        # every context/field serialised the otherwise batched draft path.
        flat_tokens = torch.cat(
            [selected.reshape(-1) for _, _, selected in pending_winners]
        ).detach().cpu().tolist()
        offset = 0
        for context, fields, selected in pending_winners:
            token_set: set[int] = set()
            n_selected = int(selected.numel())
            for field, token_value in zip(
                fields, flat_tokens[offset : offset + n_selected]
            ):
                token = int(token_value)
                token_set.add(token)
                children.setdefault(context + (token,), []).append(int(field))
            draft_tokens[context] = token_set
            offset += n_selected

        next_level = list(children.keys())
        for child, fields in children.items():
            active_fields[child] = tuple(fields)
        if sum(len(active_fields[child]) for child in next_level) != width:
            raise RuntimeError("active fields no longer partition [B]")
        levels.append(next_level)

    total_contexts = sum(len(level) for level in levels)
    if total_contexts > 1 + int(width) * int(lookahead):
        raise RuntimeError("max-order PF tree violated its linear size bound")
    return (
        MaxOrderPFTree(
            levels=levels,
            active_fields=active_fields,
            draft_tokens=draft_tokens,
            draft_tree_size=draft_tree_size,
            draft_forward_calls=draft_forward_calls,
            attempted_draft_tokens=attempted_draft_tokens,
            uniforms_by_context=uniforms_by_context,
            labels_by_context=labels_by_context,
        ),
        past_by_context,
    )


@torch.no_grad()
def _target_processed_from_leaves_cached(
    *,
    model,
    leaves: List[ContextKey],
    root_len: int,
    lookahead: int,
    target_past_key_values: Any,
    process_logits_kwargs: Optional[Dict[str, Any]],
    max_vocab_size: Optional[int],
    compact_counter: bool = False,
) -> Tuple[
    Dict[ContextKey, Tuple[FloatTensor, Optional[LongTensor]]],
    Any,
    List[ContextKey],
    FloatTensor,
    Optional[LongTensor],
]:
    """Evaluate at most B leaves once and recover every unique tree context."""
    if not leaves:
        raise ValueError("cannot evaluate an empty max-order PF tree")
    device = _model_device(model)
    cached_n = cache_len(target_past_key_values)
    if cached_n > 0:
        target_input_ids = _batch_from_contexts(
            [leaf[cached_n:] for leaf in leaves], device
        )
    else:
        target_input_ids = _batch_from_contexts(leaves, device)
    n_leaves = int(target_input_ids.shape[0])
    repeated_cache = (
        _repeat_cache(target_past_key_values, n_leaves)
        if target_past_key_values is not None
        else None
    )
    if cached_n > root_len - 1:
        raise RuntimeError(
            f"target cache length {cached_n} exceeds root_len-1={root_len - 1}"
        )

    full_input_length = int(target_input_ids.shape[1])
    out = _model_forward(
        model,
        num_logits_to_keep=lookahead + 1,
        input_ids=target_input_ids,
        past_key_values=repeated_cache,
        use_cache=True,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    logits_all = out.logits
    kept_start = full_input_length - int(logits_all.shape[1])
    if max_vocab_size is not None and max_vocab_size < logits_all.shape[-1]:
        logits_all = logits_all[..., :max_vocab_size]

    target_temp = _temperature_for_target(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)
    # Select one representative (leaf row, logit position) for every merged
    # context, then process all unique rows in one GPU call.  The previous
    # implementation launched top-k/temperature kernels once per context.
    context_positions: List[Tuple[ContextKey, int, int]] = []
    seen: set[ContextKey] = set()
    for row, leaf in enumerate(leaves):
        for depth in range(int(lookahead) + 1):
            context = leaf[: root_len + depth]
            if context in seen:
                continue
            seen.add(context)
            pos = root_len + depth - 1 - cached_n - kept_start
            if pos < 0:
                raise RuntimeError(f"invalid target-logit position {pos}")
            context_positions.append((context, row, pos))

    row_idx = torch.tensor(
        [item[1] for item in context_positions], device=device, dtype=torch.long
    )
    pos_idx = torch.tensor(
        [item[2] for item in context_positions], device=device, dtype=torch.long
    )
    unique_logits = logits_all[row_idx, pos_idx, :]
    if compact_counter and 0 < int(top_k) < int(unique_logits.shape[-1]):
        processed_all, supports_all = _compact_topk_logits(
            unique_logits, top_k=top_k, temperature=target_temp
        )
    else:
        processed_all, supports_all = process_logits_exact(
            unique_logits,
            temperature=target_temp,
            top_k=top_k,
            top_p=top_p,
            return_support=True,
        )
    processed_by_context: Dict[
        ContextKey, Tuple[FloatTensor, Optional[LongTensor]]
    ] = {}
    for idx, (context, _row, _pos) in enumerate(context_positions):
        support = None if supports_all is None else supports_all[idx]
        processed_by_context[context] = (processed_all[idx], support)
    return (
        processed_by_context,
        out.past_key_values,
        [item[0] for item in context_positions],
        processed_all,
        supports_all,
    )


@torch.no_grad()
def speculative_max_order_pf_block(
    *,
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    width: int,
    source_factory: PFRSourceFactory,
    max_new_tokens: int,
    target_past_key_values: Any = None,
    ref_past_key_values: Any = None,
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    target_coupling: str = "max_order",
    rng_backend: str = "torch_dense",
    return_logprobs: bool = True,
    record_pivots: bool = True,
    batch_target_selection: bool = False,
    fuse_latin_sampling: bool = True,
    batch_latin_draft_sampling: bool = False,
    root_context: Optional[ContextKey] = None,
) -> MaxOrderPFBlock:
    """One speculative block with a fixed-B target PF path."""
    device = _model_device(model)
    input_ids = input_ids.to(device)
    root = root_context if root_context is not None else _context_key(input_ids)
    root_len = len(root)
    block_len = min(int(lookahead), int(max_new_tokens))
    max_vocab_size = min(
        int(getattr(model.config, "vocab_size")),
        int(getattr(ref_model.config, "vocab_size")),
    )

    tree, draft_past_by_context = build_max_order_pf_tree_cached(
        ref_model=ref_model,
        root=root,
        lookahead=block_len,
        width=width,
        source_factory=source_factory,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
        past_key_values=ref_past_key_values,
        rng_backend=rng_backend,
        field_coupling=(
            target_coupling
            if target_coupling in {"latin_hypercube", "latin_reverse"}
            else "iid"
        ),
        fuse_latin_sampling=fuse_latin_sampling,
        batch_latin_draft_sampling=batch_latin_draft_sampling,
    )
    leaves = tree.levels[block_len]
    use_compact_counter = (
        rng_backend == "counter_philox"
        and not return_logprobs
        and 0 < int(_top_k(process_logits_kwargs)) < int(max_vocab_size)
    )
    (
        target_logits,
        repeated_target_cache,
        target_contexts,
        target_processed_batch,
        target_supports_batch,
    ) = _target_processed_from_leaves_cached(
        model=model,
        leaves=leaves,
        root_len=root_len,
        lookahead=block_len,
        target_past_key_values=target_past_key_values,
        process_logits_kwargs=process_logits_kwargs,
        max_vocab_size=max_vocab_size,
        compact_counter=use_compact_counter,
    )

    generator = torch.Generator(device=device)
    current = root
    output_tokens: List[int] = []
    output_processed_logits: List[FloatTensor] = []
    output_labels: List[bytes] = []
    output_pivots: List[FloatTensor] = []
    accepted_count = 0
    got_eos = False
    covered = True
    eos_tokens = _as_eos_set(getattr(model.config, "eos_token_id", None))

    if target_coupling not in {
        "max_order", "random_anchor", "latin_hypercube", "latin_reverse"
    }:
        raise ValueError(f"unknown target_coupling: {target_coupling}")

    # All target logits are already available after one batched model forward.
    # In the timed counter path, perform only the actually visited PF
    # reductions and the draft-membership traversal in one GPU program.  This
    # removes per-token host synchronizations without precomputing unused
    # winners and without changing any PRF address or coupling operation.
    used_fused_traversal = False
    if (
        rng_backend == "counter_philox"
        and target_coupling == "max_order"
        and not record_pivots
        and not return_logprobs
        and batch_target_selection
        and target_supports_batch is not None
        and len(eos_tokens) <= 1
        and not use_compact_counter
    ):
        labels = [
            max_order_context_label(context, width)
            for context in target_contexts
        ]
        seeds = [
            _counter_context_seed(source_factory, label)
            for label in labels
        ]
        row_by_context = {
            context: row for row, context in enumerate(target_contexts)
        }
        children_by_parent: Dict[ContextKey, List[ContextKey]] = {}
        for level in tree.levels[1:]:
            for child in level:
                children_by_parent.setdefault(child[:-1], []).append(child)
        child_token_rows = [
            [-1 for _ in range(width)] for _ in target_contexts
        ]
        child_context_rows = [
            [-1 for _ in range(width)] for _ in target_contexts
        ]
        for parent, children in children_by_parent.items():
            parent_row = row_by_context[parent]
            for slot, child in enumerate(children):
                child_token_rows[parent_row][slot] = int(child[-1])
                child_context_rows[parent_row][slot] = row_by_context[child]

        max_block_outputs = min(block_len + 1, int(max_new_tokens))
        (
            output_tokens,
            output_context_rows,
            accepted_count,
            final_context_row,
        ) = counter_target_traverse_support(
            processed_logits=target_processed_batch,
            supports=target_supports_batch,
            context_seeds=seeds,
            child_tokens=torch.tensor(child_token_rows, dtype=torch.int64),
            child_rows=torch.tensor(child_context_rows, dtype=torch.int64),
            width=width,
            draft_steps=block_len,
            max_outputs=max_block_outputs,
            eos_token=(next(iter(eos_tokens)) if eos_tokens else -1),
        )
        output_labels = [labels[row] for row in output_context_rows]
        current = target_contexts[final_context_row]
        got_eos = bool(output_tokens and output_tokens[-1] in eos_tokens)
        covered = False
        used_fused_traversal = True

    if (
        not used_fused_traversal
        and rng_backend == "counter_philox"
        and target_coupling == "latin_hypercube"
        and fuse_latin_sampling
        and not record_pivots
        and not return_logprobs
        and batch_target_selection
        and target_supports_batch is not None
        and len(eos_tokens) <= 1
        and use_compact_counter
    ):
        labels = [
            max_order_context_label(context, width)
            for context in target_contexts
        ]
        seeds = [_counter_context_seed(source_factory, label) for label in labels]
        row_by_context = {
            context: row for row, context in enumerate(target_contexts)
        }
        children_by_parent: Dict[ContextKey, List[ContextKey]] = {}
        for level in tree.levels[1:]:
            for child in level:
                children_by_parent.setdefault(child[:-1], []).append(child)
        child_token_rows = [[-1 for _ in range(width)] for _ in target_contexts]
        child_context_rows = [[-1 for _ in range(width)] for _ in target_contexts]
        for parent, children in children_by_parent.items():
            parent_row = row_by_context[parent]
            for slot, child in enumerate(children):
                child_token_rows[parent_row][slot] = int(child[-1])
                child_context_rows[parent_row][slot] = row_by_context[child]

        max_block_outputs = min(block_len + 1, int(max_new_tokens))
        (
            output_tokens,
            output_context_rows,
            accepted_count,
            final_context_row,
        ) = counter_latin_target_traverse_support(
            processed_logits=target_processed_batch,
            supports=target_supports_batch,
            context_seeds=seeds,
            child_tokens=torch.tensor(child_token_rows, dtype=torch.int64),
            child_rows=torch.tensor(child_context_rows, dtype=torch.int64),
            width=width,
            vocab_size=max_vocab_size,
            draft_steps=block_len,
            max_outputs=max_block_outputs,
            eos_token=(next(iter(eos_tokens)) if eos_tokens else -1),
        )
        output_labels = [labels[row] for row in output_context_rows]
        current = target_contexts[final_context_row]
        got_eos = bool(output_tokens and output_tokens[-1] in eos_tokens)
        covered = False
        used_fused_traversal = True

    def select_target(
        context: ContextKey,
        processed: FloatTensor,
        support: Optional[LongTensor],
    ) -> Tuple[int, FloatTensor, bytes]:
        context_label = tree.labels_by_context.get(context)
        if context_label is None:
            context_label = max_order_context_label(context, width)
        if target_coupling in {"max_order", "latin_hypercube", "latin_reverse"}:
            fields = tuple(range(int(width)))
        else:
            fields = (
                random_anchor_field(source_factory, context_label, width),
            )
        cached = tree.uniforms_by_context.get(context)
        if target_coupling in {"latin_hypercube", "latin_reverse"}:
            dense_latin = (
                keyed_reverse_latin_uniform_fields
                if target_coupling == "latin_reverse"
                else keyed_latin_uniform_fields
            )
            counter_latin = (
                keyed_counter_reverse_latin_uniforms_on_support
                if target_coupling == "latin_reverse"
                else keyed_counter_latin_uniforms_on_support
            )
            if support is None:
                uniforms = dense_latin(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    width=width,
                    vocab_size=int(processed.shape[-1]),
                    device=device,
                    generator=generator,
                )
                aggregate = aggregate_latin_uniform(uniforms)
                token = int(uniform_race_select(processed, aggregate)[0].item())
                pivot = aggregate[token]
            elif rng_backend == "counter_philox":
                if target_coupling == "latin_hypercube" and fuse_latin_sampling:
                    token, pivot = counter_latin_target_select_support(
                        processed_logits=processed,
                        support=support,
                        width=width,
                        source_factory=source_factory,
                        context_label=context_label,
                        exact_pivot=record_pivots,
                        compact_logits=use_compact_counter,
                        vocab_size=max_vocab_size,
                    )
                else:
                    support_uniforms = counter_latin(
                        source_factory=source_factory,
                        context_label=context_label,
                        fields=fields,
                        width=width,
                        support=support,
                        vocab_size=max_vocab_size,
                        device=device,
                    )
                    token, pivot = _latin_select_compact(
                        processed,
                        support_uniforms,
                        support,
                        compact_logits=use_compact_counter,
                    )
            else:
                uniforms = dense_latin(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    width=width,
                    vocab_size=int(processed.shape[-1]),
                    device=device,
                    generator=generator,
                )
                support_uniforms = uniforms.index_select(
                    1, support.reshape(-1).to(device=device, dtype=torch.long)
                )
                token, pivot = _latin_select_compact(
                    processed, support_uniforms, support
                )
        elif support is None:
            uniforms = _keyed_uniform_fields_with_cache(
                source_factory=source_factory,
                context_label=context_label,
                fields=fields,
                vocab_size=int(processed.shape[-1]),
                device=device,
                generator=generator,
                cached_rows=cached,
            )
            token, pivot = _aggregate_select_support(processed, uniforms, None)
        else:
            if rng_backend == "counter_philox":
                token, pivot = counter_target_select_support(
                    processed_logits=processed,
                    support=support,
                    fields=fields,
                    source_factory=source_factory,
                    context_label=context_label,
                    exact_pivot=record_pivots,
                    compact_logits=use_compact_counter,
                    vocab_size=max_vocab_size,
                )
            else:
                support_uniforms, _ = _keyed_uniforms_on_support(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    support=support,
                    vocab_size=int(processed.shape[-1]),
                    device=device,
                    generator=generator,
                    cached_rows=cached,
                )
                token, pivot = _aggregate_select_compact(
                    processed, support_uniforms, support
                )
        return token, pivot, context_label

    if not used_fused_traversal:
        for _ in range(block_len):
            processed, support = target_logits[current]
            token, pivot, context_label = select_target(current, processed, support)
            output_tokens.append(token)
            if return_logprobs:
                output_processed_logits.append(processed)
            output_labels.append(context_label)
            if record_pivots:
                output_pivots.append(pivot)

            if token in eos_tokens:
                got_eos = True
                covered = False
                break
            if token not in tree.draft_tokens.get(current, set()):
                covered = False
                break
            accepted_count += 1
            current = current + (token,)
            if len(output_tokens) >= max_new_tokens:
                covered = False
                break

    if (
        not used_fused_traversal
        and covered
        and not got_eos
        and len(output_tokens) < max_new_tokens
    ):
        processed, support = target_logits[current]
        token, pivot, context_label = select_target(current, processed, support)
        output_tokens.append(token)
        if return_logprobs:
            output_processed_logits.append(processed)
        output_labels.append(context_label)
        if record_pivots:
            output_pivots.append(pivot)
        got_eos = token in eos_tokens

    output_ids = torch.tensor([output_tokens], device=device, dtype=torch.long)
    # One batched log-softmax replaces one vocabulary-wide launch per emitted
    # token without changing the returned dense log-probabilities.
    if return_logprobs:
        output_logprobs_tensor = F.log_softmax(
            torch.stack(output_processed_logits, dim=0).float(), dim=-1
        ).unsqueeze(0)
    else:
        output_logprobs_tensor = torch.empty(
            (1, len(output_tokens), 0), device=device, dtype=torch.float32
        )
    output_pivots_tensor = (
        torch.stack(output_pivots).to(torch.float32)
        if output_pivots
        else torch.empty(0, device=device, dtype=torch.float32)
    )

    alive_row_idx = 0
    current_len = len(current)
    for row_idx, leaf in enumerate(leaves):
        if leaf[:current_len] == current:
            alive_row_idx = row_idx
            break
    new_cache_len = root_len + accepted_count
    target_cache = _select_and_truncate_cache(
        repeated_target_cache,
        alive_row_idx,
        new_cache_len,
        make_contiguous=False,
    )
    draft_cache = draft_past_by_context.get(current)
    if draft_cache is None and len(current) > root_len:
        draft_cache = draft_past_by_context.get(current[:-1])

    return MaxOrderPFBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs_tensor,
        output_labels=output_labels,
        output_pivots=output_pivots_tensor,
        accepted_count=accepted_count,
        attempted_draft_tokens=tree.attempted_draft_tokens,
        draft_tree_size=tree.draft_tree_size,
        target_context_count=len(target_logits),
        target_forward_calls=1,
        draft_forward_calls=tree.draft_forward_calls,
        got_eos=got_eos,
        target_past_key_values=target_cache,
        draft_past_key_values=draft_cache,
    )


@torch.no_grad()
def speculative_max_order_pf_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    lookahead: int,
    width: int,
    max_length: int,
    private_key: bytes | str = b"1234",
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    target_coupling: str = "max_order",
    rng_backend: str = "torch_dense",
    return_logprobs: bool = True,
    record_pivots: bool = True,
    batch_target_selection: bool = False,
    fuse_latin_sampling: bool = True,
    batch_latin_draft_sampling: bool = False,
) -> Iterable:
    """Generate with watermarkable max-order PF speculative decoding."""
    model.eval()
    ref_model.eval()
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")
    source_factory = PFRSourceFactory(private_key=bytes(private_key))
    device = _model_device(model)
    input_ids = input_ids.to(device)
    target_cache = None
    draft_cache = None
    generated = 0
    root_context = _context_key(input_ids)

    while generated < int(max_length):
        block = speculative_max_order_pf_block(
            model=model,
            ref_model=ref_model,
            input_ids=input_ids,
            lookahead=lookahead,
            width=width,
            source_factory=source_factory,
            max_new_tokens=int(max_length) - generated,
            target_past_key_values=target_cache,
            ref_past_key_values=draft_cache,
            process_logits_kwargs=process_logits_kwargs or {},
            target_coupling=target_coupling,
            rng_backend=rng_backend,
            return_logprobs=return_logprobs,
            record_pivots=record_pivots,
            batch_target_selection=batch_target_selection,
            fuse_latin_sampling=fuse_latin_sampling,
            batch_latin_draft_sampling=batch_latin_draft_sampling,
            root_context=root_context,
        )
        meta = {
            "accepted_count": block.accepted_count,
            "attempted_draft_tokens": block.attempted_draft_tokens,
            "draft_tree_size": block.draft_tree_size,
            "target_context_count": block.target_context_count,
            "target_forward_calls": block.target_forward_calls,
            "draft_forward_calls": block.draft_forward_calls,
            "width": int(width),
            "proposal": target_coupling + "_pf",
            "source_labels": block.output_labels,
            # Keep pivots on device. Benchmark/detector code may transfer or
            # regenerate them after the generation timing window.
            "aggregate_pivots": (
                block.output_pivots if block.output_pivots.numel() else None
            ),
            "rng_backend": rng_backend,
        }
        if return_meta:
            yield block.output_ids, block.output_logprobs, meta
        else:
            yield block.output_ids, block.output_logprobs

        input_ids = torch.cat([input_ids, block.output_ids.to(device)], dim=1)
        root_context = root_context + tuple(
            int(token) for token in block.output_ids[0].detach().cpu().tolist()
        )
        target_cache = block.target_past_key_values
        draft_cache = block.draft_past_key_values
        generated += int(block.output_ids.shape[-1])
        if block.got_eos:
            break


@torch.no_grad()
def max_order_pf_generator(
    model,
    input_ids: LongTensor,
    width: int,
    max_length: int,
    private_key: bytes | str = b"1234",
    process_logits_kwargs: Optional[Dict[str, Any]] = None,
    return_meta: bool = False,
    max_vocab_size: Optional[int] = None,
    target_coupling: str = "max_order",
    rng_backend: str = "torch_dense",
    return_logprobs: bool = True,
    record_pivots: bool = True,
) -> Iterable:
    """Autoregressive fixed-B max-order PF baseline for exactness checks."""
    model.eval()
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")
    source_factory = PFRSourceFactory(private_key=bytes(private_key))
    device = _model_device(model)
    input_ids = input_ids.to(device)
    cache = None
    generated = 0
    eos_tokens = _as_eos_set(getattr(model.config, "eos_token_id", None))
    model_vocab = int(getattr(model.config, "vocab_size"))
    vocab_size = min(model_vocab, int(max_vocab_size or model_vocab))
    temperature = _temperature_for_target(process_logits_kwargs)
    top_k = _top_k(process_logits_kwargs)
    top_p = _top_p(process_logits_kwargs)
    generator = torch.Generator(device=device)
    if target_coupling not in {
        "max_order", "random_anchor", "latin_hypercube", "latin_reverse"
    }:
        raise ValueError(f"unknown target_coupling: {target_coupling}")
    if rng_backend not in {"torch_dense", "counter_philox"}:
        raise ValueError(f"unknown rng_backend: {rng_backend}")

    while generated < int(max_length):
        cached_n = cache_len(cache)
        model_input = input_ids[:, cached_n:] if cached_n > 0 else input_ids
        out = _model_forward(
            model,
            num_logits_to_keep=1,
            input_ids=model_input,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        processed, support = process_logits_exact(
            out.logits[0, -1, :vocab_size],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_support=True,
        )
        context = _context_key(input_ids)
        context_label = max_order_context_label(context, width)
        fields = tuple(range(int(width)))
        if target_coupling == "random_anchor":
            fields = (random_anchor_field(source_factory, context_label, width),)
        if target_coupling in {"latin_hypercube", "latin_reverse"}:
            fields = tuple(range(int(width)))
            dense_latin = (
                keyed_reverse_latin_uniform_fields
                if target_coupling == "latin_reverse"
                else keyed_latin_uniform_fields
            )
            counter_latin = (
                keyed_counter_reverse_latin_uniforms_on_support
                if target_coupling == "latin_reverse"
                else keyed_counter_latin_uniforms_on_support
            )
            if rng_backend == "counter_philox" and support is not None:
                support_uniforms = counter_latin(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    width=width,
                    support=support,
                    vocab_size=vocab_size,
                    device=device,
                )
                token, pivot = _latin_select_compact(
                    processed, support_uniforms, support
                )
            else:
                uniforms = dense_latin(
                    source_factory=source_factory,
                    context_label=context_label,
                    fields=fields,
                    width=width,
                    vocab_size=vocab_size,
                    device=device,
                    generator=generator,
                )
                if support is None:
                    aggregate = aggregate_latin_uniform(uniforms)
                    token = int(
                        uniform_race_select(processed, aggregate)[0].item()
                    )
                    pivot = aggregate[token]
                else:
                    support_uniforms = uniforms.index_select(
                        1, support.reshape(-1).to(device=device, dtype=torch.long)
                    )
                    token, pivot = _latin_select_compact(
                        processed, support_uniforms, support
                    )
        elif rng_backend == "counter_philox" and support is not None:
            token, pivot = counter_target_select_support(
                processed_logits=processed,
                support=support,
                fields=fields,
                source_factory=source_factory,
                context_label=context_label,
                exact_pivot=record_pivots,
            )
        else:
            uniforms = keyed_uniform_fields(
                source_factory=source_factory,
                context_label=context_label,
                fields=fields,
                vocab_size=vocab_size,
                device=device,
                generator=generator,
            )
            token, pivot = _aggregate_select_support(processed, uniforms, support)
        token_ids = torch.tensor([[token]], device=device, dtype=torch.long)
        logprobs = (
            F.log_softmax(processed.float(), dim=-1).unsqueeze(0)
            if return_logprobs
            else torch.empty((1, 1, 0), device=device, dtype=torch.float32)
        )
        meta = {
            "source_labels": [context_label],
            "aggregate_pivots": pivot.reshape(1) if record_pivots else None,
            "width": int(width),
            "target_coupling": target_coupling,
            "rng_backend": rng_backend,
        }
        if return_meta:
            yield token_ids, logprobs, meta
        else:
            yield token_ids, logprobs
        input_ids = torch.cat([input_ids, token_ids], dim=1)
        cache = out.past_key_values
        generated += 1
        if token in eos_tokens:
            break


@torch.no_grad()
def recover_max_order_pivots(
    *,
    out_ids: LongTensor,
    context_labels: Sequence[bytes],
    width: int,
    private_key: bytes | str,
    vocab_size: int,
    device: torch.device | str = "cpu",
    target_coupling: str = "max_order",
    rng_backend: str = "torch_dense",
) -> FloatTensor:
    """Regenerate the aggregate ``U_{c,B}(w)`` detector pivots."""
    if isinstance(private_key, str):
        private_key = private_key.encode("utf-8")
    tokens = out_ids.reshape(-1).detach().cpu().tolist()
    if len(tokens) != len(context_labels):
        raise ValueError("one context label is required per output token")
    source_factory = PFRSourceFactory(private_key=bytes(private_key))
    device = torch.device(device)
    generator = torch.Generator(device=device)
    if target_coupling not in {
        "max_order", "random_anchor", "latin_hypercube", "latin_reverse"
    }:
        raise ValueError(f"unknown target_coupling: {target_coupling}")
    if rng_backend not in {"torch_dense", "counter_philox"}:
        raise ValueError(f"unknown rng_backend: {rng_backend}")
    pivots: List[FloatTensor] = []
    for token, context_label in zip(tokens, context_labels):
        fields = tuple(range(int(width)))
        if target_coupling == "random_anchor":
            fields = (random_anchor_field(source_factory, context_label, width),)
        if target_coupling in {"latin_hypercube", "latin_reverse"} and rng_backend == "counter_philox":
            counter_latin = (
                keyed_counter_reverse_latin_uniforms_on_support
                if target_coupling == "latin_reverse"
                else keyed_counter_latin_uniforms_on_support
            )
            uniforms = counter_latin(
                source_factory=source_factory,
                context_label=bytes(context_label),
                fields=fields,
                width=width,
                support=torch.tensor([int(token)], device=device),
                vocab_size=int(vocab_size),
                device=device,
            )
        elif target_coupling in {"latin_hypercube", "latin_reverse"}:
            dense_latin = (
                keyed_reverse_latin_uniform_fields
                if target_coupling == "latin_reverse"
                else keyed_latin_uniform_fields
            )
            uniforms = dense_latin(
                source_factory=source_factory,
                context_label=bytes(context_label),
                fields=fields,
                width=width,
                vocab_size=int(vocab_size),
                device=device,
                generator=generator,
            )
        elif rng_backend == "counter_philox":
            uniforms = keyed_counter_uniforms_on_support(
                source_factory=source_factory,
                context_label=bytes(context_label),
                fields=fields,
                support=torch.tensor([int(token)], device=device),
                vocab_size=int(vocab_size),
                device=device,
            )
        else:
            uniforms = keyed_uniform_fields(
                source_factory=source_factory,
                context_label=bytes(context_label),
                fields=fields,
                vocab_size=int(vocab_size),
                device=device,
                generator=generator,
            )
        # Generation only needs the emitted token's pivot as well.  Preserve
        # the full RNG draw, then avoid vocabulary-wide aggregate arithmetic.
        token_uniforms = (
            uniforms
            if rng_backend == "counter_philox"
            else uniforms[:, int(token) : int(token) + 1]
        )
        aggregate_fn = (
            aggregate_latin_uniform
            if target_coupling in {"latin_hypercube", "latin_reverse"}
            else aggregate_min_uniform
        )
        pivots.append(aggregate_fn(token_uniforms)[0])
    return torch.stack(pivots).to(torch.float32)


def pf_power_law_score(pivots: FloatTensor, eps: float) -> FloatTensor:
    """Centered adaptive PF power-law score from Equation (10)."""
    if not 0.0 < float(eps) < 1.0:
        raise ValueError("eps must lie in (0,1)")
    pivots = pivots.float().clamp(min=torch.finfo(torch.float32).tiny, max=1.0)
    cap = float(eps) ** -0.5
    return torch.minimum(pivots.rsqrt(), torch.full_like(pivots, cap)) - (
        2.0 - math.sqrt(float(eps))
    )


def pf_li_score(pivots: FloatTensor, rho: float) -> FloatTensor:
    """Per-token Li-style PF log likelihood-ratio score from Equation (9)."""
    if not 0.0 < float(rho) < 1.0:
        raise ValueError("rho must lie in (0,1)")
    v = pivots.float().clamp(min=torch.finfo(torch.float32).tiny, max=1.0)
    density = 1.0 - float(rho) * v
    density = density + torch.where(
        v <= float(rho), 1.0 - v / float(rho), torch.zeros_like(v)
    )
    return torch.log(density.clamp(min=torch.finfo(torch.float32).tiny))
