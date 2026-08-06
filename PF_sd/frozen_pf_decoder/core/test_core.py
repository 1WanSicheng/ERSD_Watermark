from __future__ import annotations

from types import SimpleNamespace

import torch

from accuwm.pfr import PFRSourceFactory

from .max_order_pf import (
    _counter_context_seed,
    _compact_topk_logits,
    aggregate_min_uniform,
    build_max_order_pf_tree_cached,
    counter_target_select_batch_support,
    counter_target_select_support,
    counter_draft_select_support,
    keyed_uniform_fields,
    max_order_context_label,
    max_order_field_label,
    max_order_pf_generator,
    pf_power_law_score,
    recover_max_order_pivots,
    speculative_max_order_pf_generator,
    uniform_race_select,
    uniform_race_select_support,
)
from MPFR_spec.mpfr_direct_optimized import process_logits_exact


class ToyLM(torch.nn.Module):
    """Small cache-free Markov LM for exact decoder tests."""

    def __init__(self, vocab_size: int = 11, shift: int = 0, scale: float = 1.0):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=vocab_size, eos_token_id=-1)
        self.shift = int(shift)
        self.scale = float(scale)

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, past_key_values=None, **_):
        del past_key_values
        vocab = torch.arange(
            self.config.vocab_size, device=input_ids.device, dtype=torch.float32
        )
        centers = (input_ids + self.shift + 1) % self.config.vocab_size
        logits = -self.scale * torch.abs(
            vocab.view(1, 1, -1) - centers.unsqueeze(-1).float()
        )
        return SimpleNamespace(logits=logits, past_key_values=None)


def _drain(generator):
    tokens = []
    labels = []
    pivots = []
    metas = []
    for ids, _logprobs, meta in generator:
        tokens.extend(int(token) for token in ids[0].tolist())
        labels.extend(meta["source_labels"])
        pivots.extend(float(x) for x in meta["aggregate_pivots"].reshape(-1))
        metas.append(meta)
    return tokens, labels, torch.tensor(pivots), metas


def test_uniform_race_matches_report_exponential_noisy_max():
    logits = torch.tensor([0.0, 1.1, -2.0, 0.5, float("-inf")])
    uniforms = torch.tensor(
        [[0.8, 0.3, 0.01, 0.7, 0.4], [0.2, 0.9, 0.5, 0.1, 0.6]]
    )
    direct = uniform_race_select(logits, uniforms)
    noisy_max = torch.argmax(logits.unsqueeze(0) - torch.log(uniforms), dim=-1)
    assert torch.equal(direct, noisy_max)


def test_sparse_support_race_is_bit_exact_to_dense_masked_race():
    logits = torch.tensor(
        [float("-inf"), 1.1, float("-inf"), 0.5, -2.0, float("-inf")]
    )
    support = torch.tensor([4, 1, 3])
    uniforms = torch.tensor(
        [[0.8, 0.3, 0.01, 0.7, 0.4, 0.2], [0.2, 0.9, 0.5, 0.1, 0.6, 0.3]]
    )
    dense = uniform_race_select(logits, uniforms)
    sparse = uniform_race_select_support(logits, uniforms, support)
    assert torch.equal(sparse, dense)


def test_aggregate_min_uniform_is_uniform_in_distribution():
    generator = torch.Generator().manual_seed(123)
    fields = torch.rand((4, 200_000), generator=generator)
    aggregate = aggregate_min_uniform(fields)
    assert abs(float(aggregate.mean()) - 0.5) < 0.004
    assert abs(float((aggregate <= 0.1).float().mean()) - 0.1) < 0.004
    assert abs(float((aggregate <= 0.9).float().mean()) - 0.9) < 0.004


def test_tree_has_b_trajectories_and_linear_context_bound():
    model = ToyLM(vocab_size=13, shift=2, scale=0.3)
    width, lookahead = 4, 4
    tree, _ = build_max_order_pf_tree_cached(
        ref_model=model,
        root=(1, 2, 3),
        lookahead=lookahead,
        width=width,
        source_factory=PFRSourceFactory(b"linear-tree"),
        process_logits_kwargs={"temperature": 1.0, "top_k": 13},
        max_vocab_size=13,
    )
    assert all(len(level) <= width for level in tree.levels)
    assert sum(len(level) for level in tree.levels) <= 1 + width * lookahead
    assert tree.attempted_draft_tokens == width * lookahead
    for level in tree.levels:
        assert sum(len(tree.active_fields[c]) for c in level) == width


def test_speculation_matches_fixed_b_target_path_for_different_drafters():
    target = ToyLM(shift=0, scale=0.7)
    draft_a = ToyLM(shift=2, scale=0.4)
    draft_b = ToyLM(shift=6, scale=1.4)
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long)
    kwargs = {"temperature": 1.0, "top_k": 9, "top_p": 1.0}
    common = dict(
        input_ids=prompt.clone(),
        width=3,
        max_length=18,
        private_key=b"fixed-b-path",
        process_logits_kwargs=kwargs,
        return_meta=True,
    )

    baseline, _, _, _ = _drain(max_order_pf_generator(target, **common))
    speculative_a, _, _, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_a, lookahead=4, **common
        )
    )
    speculative_b, _, _, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_b, lookahead=4, **common
        )
    )
    assert speculative_a == baseline
    assert speculative_b == baseline


def _literal_algorithm2_block(target, draft, root, *, width, lookahead, key, top_k):
    """Slow equation-level oracle with no tree/cache implementation reuse."""
    factory = PFRSourceFactory(key)
    trajectories = []
    draft_winners = {}
    for field in range(width):
        context = tuple(root)
        trajectory = []
        for _ in range(lookahead):
            ids = torch.tensor([context], dtype=torch.long)
            logits = draft(ids).logits[0, -1]
            processed = process_logits_exact(logits, top_k=top_k)
            label = max_order_context_label(context, width)
            uniforms = factory.build(max_order_field_label(label, field)).uniform_noise(
                (1, logits.numel()), device=logits.device
            )[0]
            # Equation (4), evaluated literally rather than through U/alpha.
            token = int(torch.argmax(processed.float() - torch.log(uniforms)).item())
            draft_winners.setdefault(context, set()).add(token)
            trajectory.append(token)
            context = context + (token,)
        trajectories.append(tuple(trajectory))

    current = tuple(root)
    output = []
    accepted = 0
    covered = True
    for _ in range(lookahead):
        ids = torch.tensor([current], dtype=torch.long)
        logits = target(ids).logits[0, -1]
        processed = process_logits_exact(logits, top_k=top_k)
        label = max_order_context_label(current, width)
        fields = keyed_uniform_fields(
            source_factory=factory,
            context_label=label,
            fields=range(width),
            vocab_size=logits.numel(),
            device=logits.device,
        )
        aggregate = 1.0 - (1.0 - fields.amin(dim=0)).pow(width)
        token = int(torch.argmax(processed.float() - torch.log(aggregate)).item())
        output.append(token)
        if token not in draft_winners.get(current, set()):
            covered = False
            break
        accepted += 1
        current = current + (token,)

    if covered:
        ids = torch.tensor([current], dtype=torch.long)
        logits = target(ids).logits[0, -1]
        processed = process_logits_exact(logits, top_k=top_k)
        label = max_order_context_label(current, width)
        fields = keyed_uniform_fields(
            source_factory=factory,
            context_label=label,
            fields=range(width),
            vocab_size=logits.numel(),
            device=logits.device,
        )
        aggregate = 1.0 - (1.0 - fields.amin(dim=0)).pow(width)
        output.append(
            int(torch.argmax(processed.float() - torch.log(aggregate)).item())
        )
    return output, accepted, trajectories


def test_cached_decoder_matches_literal_algorithm2_block():
    target = ToyLM(vocab_size=17, shift=0, scale=0.7)
    draft = ToyLM(vocab_size=17, shift=2, scale=0.4)
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long)
    key = b"literal-algorithm-2"
    expected, expected_accepted, _ = _literal_algorithm2_block(
        target,
        draft,
        tuple(prompt[0].tolist()),
        width=3,
        lookahead=4,
        key=key,
        top_k=11,
    )
    generator = speculative_max_order_pf_generator(
        target,
        draft,
        prompt,
        lookahead=4,
        width=3,
        max_length=5,
        private_key=key,
        process_logits_kwargs={"temperature": 1.0, "top_k": 11},
        return_meta=True,
    )
    ids, _logprobs, meta = next(iter(generator))
    assert ids[0].tolist() == expected
    assert meta["accepted_count"] == expected_accepted


def test_detector_recovers_aggregate_pivots_not_one_field():
    model = ToyLM(shift=0)
    prompt = torch.tensor([[2, 5]], dtype=torch.long)
    key = b"aggregate-detector"
    tokens, labels, emitted_pivots, _ = _drain(
        max_order_pf_generator(
            model,
            prompt,
            width=4,
            max_length=12,
            private_key=key,
            process_logits_kwargs={"temperature": 1.0, "top_k": 9},
            return_meta=True,
        )
    )
    recovered = recover_max_order_pivots(
        out_ids=torch.tensor([tokens]),
        context_labels=labels,
        width=4,
        private_key=key,
        vocab_size=model.config.vocab_size,
    )
    assert torch.equal(recovered, emitted_pivots)


def test_random_anchor_variant_is_exact_invariant_and_detectable():
    target = ToyLM(shift=0, scale=0.7)
    draft_a = ToyLM(shift=2, scale=0.4)
    draft_b = ToyLM(shift=6, scale=1.4)
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long)
    key = b"random-anchor-variant"
    common = dict(
        input_ids=prompt.clone(),
        width=3,
        max_length=18,
        private_key=key,
        process_logits_kwargs={"temperature": 1.0, "top_k": 9},
        return_meta=True,
        target_coupling="random_anchor",
    )
    baseline, _, baseline_pivots, _ = _drain(
        max_order_pf_generator(target, **common)
    )
    out_a, labels_a, pivots_a, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_a, lookahead=4, **common
        )
    )
    out_b, _, _, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft_b, lookahead=4, **common
        )
    )
    assert out_a == baseline == out_b
    assert torch.equal(pivots_a, baseline_pivots)
    recovered = recover_max_order_pivots(
        out_ids=torch.tensor([out_a]),
        context_labels=labels_a,
        width=3,
        private_key=key,
        vocab_size=target.config.vocab_size,
        target_coupling="random_anchor",
    )
    assert torch.equal(recovered, pivots_a)


def test_counter_backend_exact_target_path_and_detector():
    if not torch.cuda.is_available():
        return
    target = ToyLM(shift=0, scale=0.7).cuda()
    draft = ToyLM(shift=2, scale=0.4).cuda()
    prompt = torch.tensor([[1, 4, 2]], dtype=torch.long, device="cuda")
    key = b"counter-backend"
    common = dict(
        input_ids=prompt.clone(),
        width=3,
        max_length=18,
        private_key=key,
        process_logits_kwargs={"temperature": 1.0, "top_k": 9},
        return_meta=True,
        rng_backend="counter_philox",
    )
    baseline, _, baseline_pivots, _ = _drain(
        max_order_pf_generator(target, **common)
    )
    output, labels, pivots, _ = _drain(
        speculative_max_order_pf_generator(
            target, draft, lookahead=4, **common
        )
    )
    assert output == baseline
    assert torch.equal(pivots, baseline_pivots)
    recovered = recover_max_order_pivots(
        out_ids=torch.tensor([output]),
        context_labels=labels,
        width=3,
        private_key=key,
        vocab_size=target.config.vocab_size,
        device="cuda",
        rng_backend="counter_philox",
    ).cpu()
    assert torch.equal(recovered, pivots)


def test_batched_counter_target_selection_matches_scalar():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    vocab_size = 37
    width = 3
    logits = torch.randn(7, vocab_size, device=device)
    processed, supports = process_logits_exact(
        logits, temperature=0.9, top_k=11, return_support=True
    )
    source_factory = PFRSourceFactory(b"batched-counter-target")
    labels = [f"context-{row}".encode() for row in range(logits.shape[0])]
    seeds = [
        _counter_context_seed(source_factory, label)
        for label in labels
    ]
    batched = counter_target_select_batch_support(
        processed_logits=processed,
        supports=supports,
        context_seeds=seeds,
        width=width,
    )
    scalar = torch.tensor(
        [
            counter_target_select_support(
                processed_logits=processed[row],
                support=supports[row],
                fields=range(width),
                source_factory=source_factory,
                context_label=labels[row],
                exact_pivot=False,
            )[0]
            for row in range(logits.shape[0])
        ],
        device=device,
    )
    assert torch.equal(batched, scalar)


def test_compact_counter_selection_matches_dense_topk():
    if not torch.cuda.is_available():
        return
    logits = torch.randn(41, device="cuda", dtype=torch.float16)
    full, support = process_logits_exact(
        logits, temperature=0.9, top_k=13, return_support=True
    )
    compact, compact_support = _compact_topk_logits(
        logits, top_k=13, temperature=0.9
    )
    assert torch.equal(support, compact_support)
    assert torch.equal(full[support], compact)
    source_factory = PFRSourceFactory(b"compact-counter")
    common = dict(
        support=support,
        fields=range(3),
        source_factory=source_factory,
        context_label=b"same-context",
    )
    full_draft = counter_draft_select_support(
        processed_logits=full, **common
    )
    compact_draft = counter_draft_select_support(
        processed_logits=compact,
        compact_logits=True,
        vocab_size=logits.numel(),
        **common,
    )
    assert torch.equal(full_draft, compact_draft)
    full_target = counter_target_select_support(
        processed_logits=full, exact_pivot=False, **common
    )[0]
    compact_target = counter_target_select_support(
        processed_logits=compact,
        compact_logits=True,
        vocab_size=logits.numel(),
        exact_pivot=False,
        **common,
    )[0]
    assert full_target == compact_target


def test_power_law_score_is_centered_under_uniform_null():
    generator = torch.Generator().manual_seed(7)
    pivots = torch.rand(500_000, generator=generator)
    score = pf_power_law_score(pivots, eps=0.05)
    assert abs(float(score.mean())) < 0.01
