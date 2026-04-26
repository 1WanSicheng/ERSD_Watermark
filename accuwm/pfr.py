"""
Shared-PFR speculative decoding primitives.

This module implements the algorithm in `pfr_algorithm.md` as a separate,
clear abstraction layer without modifying the existing ERSD code.

The core split is:

1. Labeler `Lambda`: context -> label
2. Optional repeated-context masking on top of the labeler
3. `PFRSource(label, key)`: deterministic shared random source
4. `PFRWin(dist, source, mu)`: winner under a shared exponential race
5. Draft / verify / bonus-token speculative decoding using the shared source

The implementation below targets the discrete-vocabulary case where the common
reference measure `mu` is the counting measure on the vocabulary.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .utils import cache_len, process_logits, truncate_cache


def _safe_log(x: FloatTensor, eps: float = 1e-20) -> FloatTensor:
    return torch.log(x.clamp(min=eps))


def _as_single_bytes(value) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _as_single_bytes(value.item())
        return _as_single_bytes(value.reshape(-1)[0].item())
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, int):
        return int(value).to_bytes(8, byteorder="big", signed=True)
    raise TypeError(f"unsupported label value type: {type(value)!r}")


def _noise_to_winner(probs: FloatTensor, exp_noise: FloatTensor) -> LongTensor:
    ratios = torch.where(
        probs > 0,
        exp_noise / probs,
        torch.full_like(probs, float("inf")),
    )
    return ratios.argmin(dim=-1)


@dataclass(frozen=True)
class LabelInfo:
    base_label: bytes
    source_label: bytes
    masked: bool


class AbstractLabeler(ABC):
    @abstractmethod
    def label(self, context: LongTensor) -> bytes:
        """Return a deterministic byte label for `context`."""


class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: LongTensor) -> np.ndarray:
        """Return context-code bytes used to initialize the keyed source."""


def _peel_ndarray(a, last_n_dim=1):
    ra = np.empty(a.shape[:-last_n_dim], dtype=object)
    for index in np.ndindex(ra.shape):
        ra[index] = a[index].copy()
    return ra


@dataclass(frozen=True)
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    n: int

    def extract(self, context: LongTensor) -> np.ndarray:
        c = context[..., -self.n :].detach().cpu().numpy()
        c = _peel_ndarray(c, last_n_dim=1)
        return np.vectorize(lambda x: x.tobytes())(c)


@dataclass(frozen=True)
class PrefixLabeler(AbstractLabeler):
    include_length: bool = True

    def label(self, context: LongTensor) -> bytes:
        tokens = context.detach().cpu().numpy().astype(np.int32, copy=False)
        payload = tokens.tobytes()
        if not self.include_length:
            return payload
        length = context.shape[-1].to_bytes(8, byteorder="big", signed=False)
        return length + payload


@dataclass(frozen=True)
class ContextCodeLabeler(AbstractLabeler):
    cc_extractor: AbstractContextCodeExtractor

    def label(self, context: LongTensor) -> bytes:
        cc = self.cc_extractor.extract(context)
        return _as_single_bytes(cc)


class RepeatedContextMaskingLabeler:
    """
    Stateful labeler wrapper that annotates repeated contexts.

    The default behavior changes the PFR source label for repeated contexts by
    prefixing the original label with `b"repeat::"`. This keeps the interface
    explicit: repeated-context masking is part of the labeler layer, not an
    afterthought in the scoring layer.
    """

    def __init__(
        self,
        base_labeler: AbstractLabeler,
        mask_transform: Callable[[bytes], bytes] | None = None,
    ):
        self.base_labeler = base_labeler
        self.mask_transform = (
            mask_transform if mask_transform is not None else self._default_transform
        )
        self._seen: set[bytes] = set()

    @staticmethod
    def _default_transform(label: bytes) -> bytes:
        return b"repeat::" + label

    def reset(self):
        self._seen.clear()

    def snapshot(self) -> set[bytes]:
        return set(self._seen)

    def restore(self, snapshot: set[bytes]):
        self._seen = set(snapshot)

    def fork(self) -> "RepeatedContextMaskingLabeler":
        child = RepeatedContextMaskingLabeler(
            base_labeler=self.base_labeler,
            mask_transform=self.mask_transform,
        )
        child.restore(self.snapshot())
        return child

    def commit(self, labels: list[LabelInfo]):
        for label in labels:
            self._seen.add(label.base_label)

    def label_info(self, context: LongTensor) -> LabelInfo:
        base_label = self.base_labeler.label(context)
        masked = base_label in self._seen
        self._seen.add(base_label)
        source_label = self.mask_transform(base_label) if masked else base_label
        return LabelInfo(
            base_label=base_label,
            source_label=source_label,
            masked=masked,
        )


def build_default_labeler(
    mode: str = "context_code",
    cc_extractor: AbstractContextCodeExtractor | None = None,
    mask_transform: Callable[[bytes], bytes] | None = None,
) -> RepeatedContextMaskingLabeler:
    """
    Convenience factory so callers do not need to manually assemble the
    labeler stack.

    Args:
        mode:
            - ``"prefix"``: use the full prefix bytes as the label
            - ``"context_code"``: use a context-code extractor
        cc_extractor:
            Context-code extractor used when ``mode="context_code"``.
            Defaults to ``PrevN_ContextCodeExtractor(n=3)``.
        mask_transform:
            Optional custom repeated-context masking transform.
    """
    if mode == "prefix":
        base_labeler: AbstractLabeler = PrefixLabeler()
    elif mode == "context_code":
        if cc_extractor is None:
            cc_extractor = PrevN_ContextCodeExtractor(n=3)
        base_labeler = ContextCodeLabeler(cc_extractor=cc_extractor)
    else:
        raise ValueError(f"unknown labeler mode: {mode}")
    return RepeatedContextMaskingLabeler(
        base_labeler=base_labeler,
        mask_transform=mask_transform,
    )


def _get_seed(*bs: bytes) -> int:
    m = hashlib.sha256()
    for b in bs:
        m.update(b)
    return int.from_bytes(m.digest(), "big") % (2**32 - 1)


def _get_rng(*bs: bytes):
    return np.random.default_rng(_get_seed(*bs))


@dataclass(frozen=True)
class SharedPFRSource:
    label: bytes
    private_key: bytes

    def numpy_rng(self):
        return _get_rng(self.label, self.private_key)

    def uniform_noise(self, shape, device) -> FloatTensor:
        rng = self.numpy_rng()
        uniform_noise = rng.random(shape).astype(np.float32)
        return torch.from_numpy(uniform_noise).to(device)

    def exponential_noise(self, vocab_size: int, device) -> FloatTensor:
        # Keep generation-side keyed randomness aligned with the detector,
        # which reconstructs tokenwise uniforms from the same keyed hash.
        uniform_noise = self.uniform_noise((1, vocab_size), device)
        return -_safe_log(uniform_noise)


@dataclass(frozen=True)
class PFRSourceFactory:
    private_key: bytes

    def build(self, label: bytes) -> SharedPFRSource:
        return SharedPFRSource(label=label, private_key=self.private_key)


@dataclass
class PFRDraftBlock:
    draft_tokens: LongTensor
    draft_logprobs: FloatTensor
    labels: list[LabelInfo]
    sources: list[SharedPFRSource]
    draft_past_key_values: any
    got_eos: bool


@dataclass
class PFRVerifyBlock:
    output_ids: LongTensor
    output_logprobs: FloatTensor
    labels: list[LabelInfo]
    accepted_count: int
    target_past_key_values: any
    got_eos: bool


def pfr_win_from_logits(
    logits: FloatTensor,
    source: SharedPFRSource,
    device,
) -> tuple[LongTensor, FloatTensor, FloatTensor]:
    logprobs = F.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    vocab_size = probs.shape[-1]
    exp_noise = source.exponential_noise(vocab_size, device)
    winner = _noise_to_winner(probs, exp_noise)
    return winner, logprobs, exp_noise


@torch.no_grad()
def draft_pfr_block(
    model,
    input_ids: LongTensor,
    lookahead: int,
    labeler: RepeatedContextMaskingLabeler,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs: dict | None = None,
) -> PFRDraftBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    cached_n = cache_len(past_key_values)
    input_tokens = input_ids[:, cached_n:] if cached_n > 0 else input_ids
    running_ids = input_ids
    draft_tokens = []
    draft_logprobs = []
    labels: list[LabelInfo] = []
    sources: list[SharedPFRSource] = []
    got_eos = False

    for _ in range(lookahead):
        output = model(input_tokens, past_key_values=past_key_values)
        logits = output.logits[:, -1, :]
        logits = process_logits(running_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]

        label_info = labeler.label_info(running_ids)
        source = source_factory.build(label_info.source_label)
        new_token, logprobs, _ = pfr_win_from_logits(
            logits, source, model.device
        )

        new_token = new_token.unsqueeze(1)
        draft_tokens.append(new_token)
        draft_logprobs.append(logprobs)
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
    else:
        vocab_size = min(model.config.vocab_size, max_vocab_size or model.config.vocab_size)
        draft_tokens_tensor = torch.empty((1, 0), device=input_ids.device, dtype=torch.long)
        draft_logprobs_tensor = torch.empty(
            (1, 0, vocab_size), device=input_ids.device, dtype=torch.float32
        )

    return PFRDraftBlock(
        draft_tokens=draft_tokens_tensor,
        draft_logprobs=draft_logprobs_tensor,
        labels=labels,
        sources=sources,
        draft_past_key_values=past_key_values,
        got_eos=got_eos,
    )


@torch.no_grad()
def verify_pfr_block(
    model,
    input_ids: LongTensor,
    draft_block: PFRDraftBlock,
    labeler: RepeatedContextMaskingLabeler,
    source_factory: PFRSourceFactory,
    past_key_values=None,
    process_logits_kwargs: dict | None = None,
) -> PFRVerifyBlock:
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    assert input_ids.shape[0] == 1, "only batch_size=1 is supported"

    draft_len = draft_block.draft_tokens.shape[1]
    if draft_len == 0:
        raise ValueError("verify_pfr_block requires at least one draft token")

    cached_n = cache_len(past_key_values)
    if cached_n > 0:
        input_tokens = torch.cat([input_ids[:, cached_n:], draft_block.draft_tokens], dim=1)
    else:
        input_tokens = torch.cat([input_ids, draft_block.draft_tokens], dim=1)

    full_ids = torch.cat([input_ids, draft_block.draft_tokens], dim=1)
    output = model(input_tokens, past_key_values=past_key_values)
    logits = output.logits[:, -draft_len - 1 :, :]

    for i in range(draft_len + 1):
        logits[:, i, :] = process_logits(
            full_ids[:, : input_ids.shape[1] + i],
            logits[:, i, :],
            **process_logits_kwargs,
        )

    target_logprobs = F.log_softmax(logits, dim=-1)
    target_probs = target_logprobs.exp()
    accepted_tokens: list[int] = []
    labels: list[LabelInfo] = []
    accepted_count = 0

    for s in range(draft_len):
        source = draft_block.sources[s]
        winner = _noise_to_winner(
            target_probs[0, s, :].unsqueeze(0),
            source.exponential_noise(target_probs.shape[-1], model.device),
        ).item()
        draft_token = draft_block.draft_tokens[0, s].item()
        if winner != draft_token:
            accepted_prefix = (
                torch.cat([input_ids, draft_block.draft_tokens[:, :s]], dim=1)
                if s > 0
                else input_ids
            )
            label_info = labeler.label_info(accepted_prefix)
            labels.append(label_info)
            accepted_tokens.append(winner)
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
        bonus_winner = _noise_to_winner(
            target_probs[0, draft_len, :].unsqueeze(0),
            bonus_source.exponential_noise(target_probs.shape[-1], model.device),
        ).item()
        accepted_tokens.append(bonus_winner)
        labels.append(bonus_label)

    output_ids = torch.tensor([accepted_tokens], device=input_ids.device, dtype=torch.long)
    output_logprobs = target_logprobs[:, : len(accepted_tokens), :]
    got_eos = bool((output_ids == model.config.eos_token_id).any())
    new_cache_len = input_ids.shape[1] + output_ids.shape[1] - 1
    new_past = truncate_cache(output.past_key_values, new_cache_len)

    return PFRVerifyBlock(
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        labels=labels,
        accepted_count=accepted_count,
        target_past_key_values=new_past,
        got_eos=got_eos,
    )


@torch.no_grad()
def pfr_speculative_generator(
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
):
    """
    Generator form of the algorithm in `pfr_algorithm.md`.

    Yields:
        (output_ids, output_logprobs, meta)

    Meta contains the per-output labels and accepted draft count for the block.
    """

    if process_logits_kwargs is None:
        process_logits_kwargs = {}

    source_factory = PFRSourceFactory(private_key=private_key)
    initial_len = input_ids.shape[1]

    while input_ids.shape[1] < initial_len + max_length:
        remaining = initial_len + max_length - input_ids.shape[1]
        block_len = min(lookahead, remaining)
        draft_labeler = labeler.fork()
        verify_labeler = labeler.fork()
        draft_block = draft_pfr_block(
            model=ref_model,
            input_ids=input_ids,
            lookahead=block_len,
            labeler=draft_labeler,
            source_factory=source_factory,
            past_key_values=ref_past_key_values,
            process_logits_kwargs=process_logits_kwargs,
            max_vocab_size=model.config.vocab_size,
        )
        verify_block = verify_pfr_block(
            model=model,
            input_ids=input_ids,
            draft_block=draft_block,
            labeler=verify_labeler,
            source_factory=source_factory,
            past_key_values=past_key_values,
            process_logits_kwargs=process_logits_kwargs,
        )
        labeler.commit(verify_block.labels)

        yield (
            verify_block.output_ids,
            verify_block.output_logprobs,
            {
                "accepted_count": verify_block.accepted_count,
                "labels": verify_block.labels,
                "draft_len": draft_block.draft_tokens.shape[1],
            },
        )

        input_ids = torch.cat([input_ids, verify_block.output_ids], dim=1)
        past_key_values = verify_block.target_past_key_values

        # Rebuild the draft cache on the true emitted sequence so the next block
        # starts from a consistent state.
        ref_update_out = ref_model(
            verify_block.output_ids.to(ref_model.device),
            past_key_values=draft_block.draft_past_key_values,
        )
        ref_past_key_values = truncate_cache(
            ref_update_out.past_key_values,
            input_ids.shape[1] - 1,
        )

        if verify_block.got_eos:
            break


def pfr_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    max_length: int,
    private_key: bytes,
    labeler: RepeatedContextMaskingLabeler | None = None,
    labeler_mode: str = "context_code",
    cc_extractor: AbstractContextCodeExtractor | None = None,
    process_logits_kwargs: dict | None = None,
):
    """
    Drop-in style wrapper matching the repository's existing generator pattern.

    Args:
        n: lookahead K in the PFR algorithm.
    """
    if labeler is None:
        labeler = build_default_labeler(
            mode=labeler_mode,
            cc_extractor=cc_extractor,
        )
    yield from pfr_speculative_generator(
        model=model,
        ref_model=ref_model,
        input_ids=input_ids,
        lookahead=n,
        max_length=max_length,
        labeler=labeler,
        private_key=private_key,
        process_logits_kwargs=process_logits_kwargs,
    )
