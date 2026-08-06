from __future__ import annotations

import torch

from PF_sd.frozen_pf_decoder.core.max_order_pf import (
    _compact_topk_logits,
    _counter_context_seed,
    _latin_select_compact,
    _uniform_race_select_compact,
    aggregate_latin_uniform,
    counter_latin_draft_select_support,
    counter_latin_draft_select_batch_support,
    counter_latin_target_select_support,
    keyed_counter_latin_uniforms_on_support,
    max_order_pf_generator,
    recover_max_order_pivots,
    reverse_stratify_uniform_fields,
    speculative_max_order_pf_generator,
    stratify_uniform_fields,
)
from PF_sd.frozen_pf_decoder.core.test_core import ToyLM, _drain
from accuwm.pfr import PFRSourceFactory


def test_latin_fields_and_target_pivot_are_uniform():
    width = 2
    n = 300_000
    generator = torch.Generator().manual_seed(7)
    raw = torch.rand(width, n, generator=generator)
    shift = torch.rand(n, generator=generator)
    fields = stratify_uniform_fields(
        raw, shift, fields=range(width), width=width
    )
    pivot = aggregate_latin_uniform(fields)
    for values in (fields[0], fields[1], pivot):
        assert abs(float(values.mean()) - 0.5) < 0.003
        assert abs(float(values.square().mean()) - 1.0 / 3.0) < 0.003
    strata = torch.floor(fields * width).to(torch.long).sort(dim=0).values
    assert torch.equal(strata[0], torch.zeros(n, dtype=torch.long))
    assert torch.equal(strata[1], torch.ones(n, dtype=torch.long))


def test_reverse_latin_fields_and_target_pivot_are_uniform():
    width = 2
    n = 300_000
    generator = torch.Generator().manual_seed(17)
    base = torch.rand(n, generator=generator)
    shift = torch.rand(n, generator=generator)
    fields = reverse_stratify_uniform_fields(
        base, shift, fields=range(width), width=width
    )
    pivot = aggregate_latin_uniform(fields)
    for values in (fields[0], fields[1], pivot):
        assert abs(float(values.mean()) - 0.5) < 0.003
        assert abs(float(values.square().mean()) - 1.0 / 3.0) < 0.003
    assert torch.allclose(fields.sum(dim=0), torch.ones(n))


def test_latin_speculation_matches_target_and_is_drafter_invariant():
    target = ToyLM(shift=0, scale=0.7)
    draft_a = ToyLM(shift=2, scale=0.4)
    draft_b = ToyLM(shift=6, scale=1.4)
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long)
    common = dict(
        input_ids=prompt.clone(),
        width=2,
        max_length=24,
        private_key=b"latin-fixed-target",
        process_logits_kwargs={"temperature": 1.0, "top_k": 9, "top_p": 1.0},
        return_meta=True,
        target_coupling="latin_hypercube",
    )
    target_tokens, labels, pivots, _ = _drain(
        max_order_pf_generator(target, **common)
    )
    draft_a_tokens, _, _, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_a, lookahead=4, **common
        )
    )
    draft_b_tokens, _, _, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_b, lookahead=4, **common
        )
    )
    assert draft_a_tokens == target_tokens
    assert draft_b_tokens == target_tokens
    recovered = recover_max_order_pivots(
        out_ids=torch.tensor([target_tokens]),
        context_labels=labels,
        width=2,
        private_key=b"latin-fixed-target",
        vocab_size=target.config.vocab_size,
        target_coupling="latin_hypercube",
    )
    assert torch.equal(recovered, pivots)


def test_reverse_latin_matches_target_and_is_drafter_invariant():
    target = ToyLM(shift=0, scale=0.7)
    draft_a = ToyLM(shift=2, scale=0.4)
    draft_b = ToyLM(shift=6, scale=1.4)
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long)
    common = dict(
        input_ids=prompt.clone(),
        width=2,
        max_length=24,
        private_key=b"reverse-latin-fixed-target",
        process_logits_kwargs={"temperature": 1.0, "top_k": 9, "top_p": 1.0},
        return_meta=True,
        target_coupling="latin_reverse",
    )
    target_tokens, labels, pivots, _ = _drain(
        max_order_pf_generator(target, **common)
    )
    for draft in (draft_a, draft_b):
        draft_tokens, _, _, _ = _drain(
            speculative_max_order_pf_generator(
                target, draft, lookahead=4, **common
            )
        )
        assert draft_tokens == target_tokens
    recovered = recover_max_order_pivots(
        out_ids=torch.tensor([target_tokens]),
        context_labels=labels,
        width=2,
        private_key=b"reverse-latin-fixed-target",
        vocab_size=target.config.vocab_size,
        target_coupling="latin_reverse",
    )
    assert torch.equal(recovered, pivots)


def test_latin_compact_logits_match_dense_support_selection():
    logits = torch.randn(41, device="cuda", dtype=torch.float16)
    compact, support = _compact_topk_logits(
        logits, top_k=13, temperature=0.9
    )
    dense = torch.full_like(logits, float("-inf"))
    dense.scatter_(0, support, compact)
    generator = torch.Generator(device="cuda").manual_seed(107)
    raw = torch.rand(2, support.numel(), generator=generator, device="cuda")
    shift = torch.rand(support.numel(), generator=generator, device="cuda")
    uniforms = stratify_uniform_fields(
        raw, shift, fields=range(2), width=2
    )
    dense_token, dense_pivot = _latin_select_compact(
        dense, uniforms, support
    )
    compact_token, compact_pivot = _latin_select_compact(
        compact, uniforms, support, compact_logits=True
    )
    assert dense_token == compact_token
    assert torch.equal(dense_pivot, compact_pivot)


def test_fused_latin_primitives_match_unfused_counter_path():
    device = torch.device("cuda")
    width, vocab_size = 4, 97
    generator = torch.Generator(device=device).manual_seed(207)
    support = torch.randperm(vocab_size, generator=generator, device=device)[:31]
    compact = torch.randn(
        support.numel(), generator=generator, device=device, dtype=torch.float16
    )
    factory = PFRSourceFactory(b"fused-latin-primitives")
    label = b"context"
    uniforms = keyed_counter_latin_uniforms_on_support(
        source_factory=factory,
        context_label=label,
        fields=range(width),
        width=width,
        support=support,
        vocab_size=vocab_size,
        device=device,
    )
    expected_draft = _uniform_race_select_compact(
        compact, uniforms, support, compact_logits=True
    )
    fused_draft = counter_latin_draft_select_support(
        processed_logits=compact,
        support=support,
        fields=range(width),
        width=width,
        source_factory=factory,
        context_label=label,
        compact_logits=True,
        vocab_size=vocab_size,
    )
    expected_token, expected_pivot = _latin_select_compact(
        compact, uniforms, support, compact_logits=True
    )
    fused_token, fused_pivot = counter_latin_target_select_support(
        processed_logits=compact,
        support=support,
        width=width,
        source_factory=factory,
        context_label=label,
        exact_pivot=True,
        compact_logits=True,
        vocab_size=vocab_size,
    )
    assert torch.equal(fused_draft, expected_draft)
    assert fused_token == expected_token
    assert torch.equal(fused_pivot, expected_pivot)

    batch_draft = counter_latin_draft_select_batch_support(
        processed_logits=compact.unsqueeze(0),
        supports=support.unsqueeze(0),
        context_rows=[0] * width,
        fields=list(range(width)),
        context_seeds=[_counter_context_seed(factory, label)] * width,
        width=width,
        vocab_size=vocab_size,
    )
    assert torch.equal(batch_draft, expected_draft)


def test_fused_latin_generator_matches_unfused_path():
    target = ToyLM(shift=0, scale=0.7).cuda()
    draft = ToyLM(shift=2, scale=0.4).cuda()
    common = dict(
        input_ids=torch.tensor([[1, 4, 2]], device="cuda"),
        lookahead=4,
        width=2,
        max_length=32,
        private_key=b"fused-latin-generator",
        process_logits_kwargs={"temperature": 1.0, "top_k": 9, "top_p": 1.0},
        return_meta=True,
        return_logprobs=False,
        record_pivots=False,
        target_coupling="latin_hypercube",
        rng_backend="counter_philox",
    )
    def drain_without_pivots(generator):
        tokens, labels, metas = [], [], []
        for ids, _logprobs, meta in generator:
            tokens.extend(int(token) for token in ids[0].tolist())
            labels.extend(meta["source_labels"])
            metas.append(meta)
        return tokens, labels, metas

    reference, labels_ref, meta_ref = drain_without_pivots(
        speculative_max_order_pf_generator(
            target, draft, fuse_latin_sampling=False,
            batch_target_selection=False, **common
        )
    )
    fused, labels_fused, meta_fused = drain_without_pivots(
        speculative_max_order_pf_generator(
            target, draft, fuse_latin_sampling=True,
            batch_target_selection=False, **common
        )
    )
    assert fused == reference
    assert labels_fused == labels_ref
    assert [m["accepted_count"] for m in meta_fused] == [
        m["accepted_count"] for m in meta_ref
    ]
