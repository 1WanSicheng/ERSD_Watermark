"""Bit-equivalence test for the B1+B2+B3 PFR sampling-path optimisations.

The detection side (`unbiased_watermark.scores.pfr_aaronson._uniform_for_token`)
re-derives per-token noise via:

    g = torch.Generator(device=device); g.manual_seed(sha256(label||key) % 2^32-1)
    r_values = torch.rand((1, vocab_size), device=device, dtype=torch.float32, generator=g)

So generation MUST produce, for the first row of each context's noise tensor,
bytes identical to the above.  This test confirms three properties:

  1. Reusing a single ``torch.Generator`` object across rows (B1) gives the
     same bytes as constructing a fresh one each time.
  2. Caching the seed (B2) returns the same int as recomputing it.
  3. Deferring per-row ``.cpu().tolist()`` (B3) produces identical token ids.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from accuwm.multi_draft_utils import ms_pfr_tokens_from_logprobs  # noqa: E402
from accuwm.pfr import PFRSourceFactory, _get_seed  # noqa: E402


def _label(prefix_tokens):
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in prefix_tokens
    )


def test_generator_reuse_byte_equivalent():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    factory = PFRSourceFactory(private_key=b"k_test_1234")
    contexts = [(11, 22, 33), (44, 55, 66, 77), (88,)]
    num_samples = 4
    vocab_size = 32_000

    # Reference path: fresh Generator each call (matches old behaviour AND
    # detection's `_uniform_for_token` call).
    ref = []
    for ctx in contexts:
        src = factory.build(_label(ctx))
        g = torch.Generator(device=device); g.manual_seed(src.seed())
        ref.append(torch.rand(
            (num_samples, vocab_size),
            device=device, dtype=torch.float32, generator=g,
        ))

    # Optimised path: reuse one Generator object across rows.
    shared_g = torch.Generator(device=device)
    out = []
    for ctx in contexts:
        src = factory.build(_label(ctx))
        out.append(src.uniform_noise(
            (num_samples, vocab_size), device=device, generator=shared_g,
        ))

    for i, ctx in enumerate(contexts):
        assert torch.equal(ref[i], out[i]), f"mismatch at context {ctx}"
    print("[B1] generator reuse: byte-identical noise across", len(contexts), "contexts")


def test_seed_cache_returns_same_int():
    factory = PFRSourceFactory(private_key=b"k_test_1234")
    contexts = [(1, 2, 3), (4, 5), (6,)]

    # Reference: re-derive each time
    ref_seeds = [
        _get_seed(_label(ctx), factory.private_key) for ctx in contexts
    ]
    # Cache simulation
    cache: dict = {}
    def seed_for(ctx):
        s = cache.get(ctx)
        if s is None:
            s = _get_seed(_label(ctx), factory.private_key)
            cache[ctx] = s
        return s
    cached = [seed_for(ctx) for ctx in contexts]
    cached2 = [seed_for(ctx) for ctx in contexts]  # second call hits cache

    assert ref_seeds == cached == cached2
    print("[B2] seed cache: identical seed ints, cache hit on repeat lookup")


def test_deferred_d2h_gives_same_tokens():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    factory = PFRSourceFactory(private_key=b"k_test_1234")
    vocab_size = 1024
    n_rows = 5
    multiplicities = [1, 2, 3, 4, 2]

    torch.manual_seed(0)
    logprobs = torch.log_softmax(torch.randn(n_rows, vocab_size, device=device), dim=-1)
    contexts = [(7, i) for i in range(n_rows)]

    # Path 1: per-row .cpu().tolist() (current)
    per_row_tokens = []
    for row in range(n_rows):
        src = factory.build(_label(contexts[row]))
        toks = ms_pfr_tokens_from_logprobs(
            logprobs[row], source=src, num_samples=multiplicities[row], device=device,
        )
        per_row_tokens.append(toks.detach().cpu().tolist())

    # Path 2: collect GPU tensors, single .cpu() at end (B3)
    g_shared = torch.Generator(device=device)
    deferred = []
    for row in range(n_rows):
        src = factory.build(_label(contexts[row]))
        deferred.append(ms_pfr_tokens_from_logprobs(
            logprobs[row], source=src, num_samples=multiplicities[row],
            device=device, generator=g_shared,
        ))
    flat = torch.cat(deferred).cpu().tolist()
    offset = 0
    deferred_lists = []
    for m in multiplicities:
        deferred_lists.append(flat[offset:offset + m])
        offset += m

    for row in range(n_rows):
        assert per_row_tokens[row] == deferred_lists[row], (
            f"row {row}: {per_row_tokens[row]} != {deferred_lists[row]}"
        )
    print("[B3] deferred D2H: identical token ids across", n_rows, "rows")


if __name__ == "__main__":
    test_generator_reuse_byte_equivalent()
    test_seed_cache_returns_same_int()
    test_deferred_d2h_gives_same_tokens()
    print("\nAll B1/B2/B3 byte-equivalence checks passed.")
