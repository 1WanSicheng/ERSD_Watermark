"""
Compare AATPS and ANLPPT across DeltaGumbel-watermarking decoders, the
PFR family, and four detection scores that all operate on the per-token
uniform pivot r_t = r_t(w_t) in [0, 1].

Decoders:
  - basic_uwm        : single-target-forward UWM (n=1, AATPS == 1)
  - mc_uwm_speed     : multi-candidate UWM, draft-side reweight
  - mc_uwm_strength  : multi-candidate UWM, target-side reweight
  - pfr              : multi-draft PFR (Poisson first-arrival watermark)
  - pfr_nowatermark  : speculative no-watermark control

Detectors (applied to the per-token uniforms u_t for each decoder):
  - U          : raw uniform                                 h_U(r) = r
  - A          : Aaronson                                    h_A(r) = -log(1-r)
  - Li(Delta)  : Li (2025) optimal score for class P_Delta
  - PL(eps)    : Lattimore (2026) truncated power-law

For DeltaGumbel-watermarked output the per-token uniform comes from
u_t = exp(-exp(-g_t)).  For PFR-watermarked output the per-token uniform is
recomputed from the recorded source labels via _uniform_for_token.  The
control (pfr_nowatermark) produces output unrelated to either watermark; we
score it with the DeltaGumbel pivot, which under H0 should give ANLPPT ~= 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unbiased_watermark as uwm  # noqa: E402
from accuwm.basic_watermark import basic_uwm_generator  # noqa: E402
from accuwm.mc_watermark import mc_uwm_sample_generator  # noqa: E402
from accuwm.pfr import PrefixLabeler, pfr_sample_generator  # noqa: E402
from accuwm.pfr_no_watermark import pfr_no_watermark_generator  # noqa: E402

# B>=1 multi-draft cached variants from MPFR_spec
sys.path.insert(0, str(ROOT / "MPFR_spec"))
from multi_draft_pfr_batched_cached import (  # noqa: E402
    multi_draft_pfr_batched_cached_sample_generator,
)
from mpfr_batched_torchgen_cached import (  # noqa: E402
    finite_multi_draft_pfr_cached_sample_generator,
)

from unbiased_watermark import DeltaGumbel_WatermarkCode  # noqa: E402
from unbiased_watermark.scores import (  # noqa: E402
    DeltaGumbel_A,
    DeltaGumbel_Li,
    DeltaGumbel_PL,
    DeltaGumbel_U,
)
from unbiased_watermark.scores.pfr_aaronson import _uniform_for_token  # noqa: E402


PRIVATE_KEY = b"1234"


def encode_prompt(tokenizer, prompt, device):
    return tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


def load_prompts(n, dataset="gsm8k"):
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["train"].select(range(n))
        return [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row["question"]},
            ]
            for row in ds
        ]
    if dataset == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000).select(range(n))
        return [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": (
                    "Summarize the following article in 3-5 sentences:\n\n"
                    f"{row['article'][:1500]}"
                )},
            ]
            for row in ds
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _seeded_key(seed: int) -> bytes:
    return int(seed).to_bytes(8, "big") + PRIVATE_KEY


# ---------------------------------------------------------------------------
# Decoder dispatch -- each returns
#   (out_ids: (1, T) torch.LongTensor on CPU,
#    block_lens: list[int],
#    private_key: bytes,
#    source_labels: Optional[list[bytes]] -- only set for PFR)
# ---------------------------------------------------------------------------


def _drain(gen, max_length, collect_labels: bool = False):
    output_chunks = []
    block_lens: List[int] = []
    source_labels: List[bytes] = []
    masked_flags: List[bool] = []
    generated = 0
    for step in gen:
        if isinstance(step, tuple) and len(step) == 3:
            step_ids, _logp, meta = step
        elif isinstance(step, tuple):
            step_ids = step[0]
            meta = {}
        else:
            step_ids = step
            meta = {}
        block_len = step_ids.shape[1]
        if generated + block_len > max_length:
            block_len = max_length - generated
        if block_len <= 0:
            break
        output_chunks.append(step_ids[:, :block_len].detach().cpu())
        block_lens.append(int(block_len))
        if collect_labels and "labels" in meta:
            for label in meta["labels"][:block_len]:
                source_labels.append(label.source_label)
                masked_flags.append(bool(getattr(label, "masked", False)))
        generated += block_len
        if generated >= max_length:
            break
    out_ids = (
        torch.cat(output_chunks, dim=1) if output_chunks else torch.empty((1, 0), dtype=torch.long)
    )
    return out_ids, block_lens, source_labels, masked_flags


def run_basic_uwm(target, draft, input_ids, lookahead, max_length, seed):
    """Single-target-forward-per-token UWM.  Force n=1 so AATPS == 1."""
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    pk = _seeded_key(seed)
    gen = basic_uwm_generator(
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=pk,
        model=target,
        input_ids=input_ids,
        n=1,
    )
    out_ids, block_lens, _, _ = _drain(gen, max_length)
    return out_ids, block_lens, pk, "DeltaGumbel", None, None


def run_mc_uwm(target, draft, input_ids, lookahead, max_length, seed, *, strength: bool):
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    pk = _seeded_key(seed)
    gen = mc_uwm_sample_generator(
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=pk,
        reweight_in_mc=strength,
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
    )
    out_ids, block_lens, _, _ = _drain(gen, max_length)
    return out_ids, block_lens, pk, "DeltaGumbel", None, None


def run_pfr(target, draft, input_ids, lookahead, max_length, seed):
    pk = _seeded_key(seed)
    gen = pfr_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        private_key=pk,
        labeler_mode="context_code",
    )
    out_ids, block_lens, source_labels, masked_flags = _drain(gen, max_length, collect_labels=True)
    return out_ids, block_lens, pk, "PFR", source_labels, masked_flags


def run_pfr_nowm(target, draft, input_ids, lookahead, max_length, seed):
    pk = _seeded_key(seed)
    gen = pfr_no_watermark_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
    )
    out_ids, block_lens, _, _ = _drain(gen, max_length)
    return out_ids, block_lens, pk, "DeltaGumbel", None, None


# ---------------------------------------------------------------------------
# B>=1 cached variants from MPFR_spec.  These don't emit per-token labels in
# meta, so for detection we reconstruct labels from the realized prefix at
# each emitted position using the same labeling convention each generator
# uses internally.
# ---------------------------------------------------------------------------


def _prefix_labeler_label(prefix_ids: torch.LongTensor) -> bytes:
    """Reproduce accuwm.pfr.PrefixLabeler.label byte layout."""
    tokens = prefix_ids.detach().cpu().numpy().astype(np.int32, copy=False)
    payload = tokens.tobytes()
    length = prefix_ids.shape[-1].to_bytes(8, byteorder="big", signed=False)
    return length + payload


def _mpfr_direct_label(prefix_ids: torch.LongTensor) -> bytes:
    """Reproduce MPFR_spec.mpfr_batched_torchgen_cached._context_label byte
    layout: ``b"MPFR_DIRECT_CLOCK_V1"`` + int64-be(signed)*N."""
    tokens = prefix_ids.detach().cpu().tolist()
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in tokens
    )


def _rebuild_labels_for_prefix_scheme(
    in_ids: torch.LongTensor,
    out_ids: torch.LongTensor,
    label_fn,
) -> List[bytes]:
    """For each emitted token at position L_in + i, the label used at gen
    time corresponds to the prefix [0, L_in + i).  Build that list."""
    full = torch.cat([in_ids[0], out_ids[0].to(in_ids.device)], dim=0)
    T_prompt = int(in_ids.shape[-1])
    T_out = int(out_ids.shape[-1])
    labels: List[bytes] = []
    for i in range(T_out):
        prefix = full[: T_prompt + i]
        labels.append(label_fn(prefix))
    return labels


def run_ms_pfr_cached(target, draft, input_ids, lookahead, max_length, seed,
                     *, num_drafts: int = 4):
    """B>=1 ms_pfr cached: GPU torch.Generator primitive + batched verify
    + KV cache (MPFR_spec/multi_draft_pfr_batched_cached.py).  Default B=4
    matches the prior MPFR_spec benchmarks."""
    pk = _seeded_key(seed)
    gen = multi_draft_pfr_batched_cached_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        num_drafts=num_drafts,
        private_key=pk,
        return_meta=True,
    )
    out_ids, block_lens, _, _ = _drain(gen, max_length)
    # Reconstruct PrefixLabeler labels for each emitted token; no
    # repeat-context masking is in play (PrefixLabeler is bijective on
    # prefixes), so masked_flags is all False.
    labels = _rebuild_labels_for_prefix_scheme(
        input_ids, out_ids, _prefix_labeler_label
    )
    masked_flags = [False] * len(labels)
    return out_ids, block_lens, pk, "PFR", labels, masked_flags


def run_mpfr_torchgen_cached(target, draft, input_ids, lookahead, max_length, seed,
                             *, num_drafts: int = 4):
    """B>=1 MPFR_spec direct-finite + GPU torchgen + KV cache.  Default B=4."""
    pk = _seeded_key(seed)
    gen = finite_multi_draft_pfr_cached_sample_generator(
        model=target,
        ref_model=draft,
        input_ids=input_ids,
        n=lookahead,
        max_length=max_length,
        num_drafts=num_drafts,
        private_key=pk,
        return_meta=True,
    )
    out_ids, block_lens, _, _ = _drain(gen, max_length)
    labels = _rebuild_labels_for_prefix_scheme(
        input_ids, out_ids, _mpfr_direct_label
    )
    masked_flags = [False] * len(labels)
    return out_ids, block_lens, pk, "PFR", labels, masked_flags


def run_invariant_multi(target, draft, input_ids, lookahead, max_length, seed,
                        *, num_drafts: int = 4):
    """InvariantMultiDraftStrategy: pre-sampled fresh torch.rand randomness,
    no recoverable watermark.  ANLPPT under PFR detector should be ~H0."""
    sys.path.insert(0, str(ROOT / "SpeculativeDecoding"))
    from strategy import InvariantMultiDraftStrategy
    from generator import InvariantGenerator

    # vocab_size must match the smaller of (target, draft) — strategy.py
    # truncates target logits to this and indexes pre-sampled randomness
    # of the same width.  Using the larger model would cause a shape
    # mismatch against the draft model's logits.
    eff_vocab = min(int(target.config.vocab_size), int(draft.config.vocab_size))
    class _StubTok:
        vocab_size = eff_vocab
    strategy = InvariantMultiDraftStrategy(
        target=target, drafter=draft, tokenizer=_StubTok(),
        max_draft_len=lookahead, max_num_drafts=num_drafts,
    )
    generator = InvariantGenerator(strategy)
    eos = int(getattr(target.config, "eos_token_id", 0) or 0)
    out = generator(
        input_ids=input_ids,
        eos_token_id=eos,
        max_new_tokens=max_length,
        temperature=1.0,
    )
    full_seq = out.sequences  # (1, L_in + T_out)
    L_in = int(input_ids.shape[-1])
    out_ids = full_seq[:, L_in:].detach().cpu()
    n_tokens = int(out_ids.shape[-1])
    n_invocations = int(out.num_invocations)
    block_lens = [max(n_tokens // max(n_invocations, 1), 1)] * n_invocations
    pk = _seeded_key(seed)
    return out_ids, block_lens, pk, "DeltaGumbel", None, None  # H0 control  # H0 control under DG pivot


# ---------------------------------------------------------------------------
# Per-token uniforms u_t in [0, 1] for each watermark family
# ---------------------------------------------------------------------------


def _uniforms_from_dg(
    *,
    target,
    out_ids: torch.LongTensor,
    in_ids: torch.LongTensor,
    private_key: bytes,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return per-token uniforms u_t and the skip mask under the DeltaGumbel
    pivot.  detect_pre's `skipped` array marks tokens whose context code
    repeated within the same generation; per Aaronson-style detection these
    must be excluded so the score sums only over independent contexts."""
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    vocab_size = int(target.config.vocab_size)
    _, _, _, watermark_code, skipped = uwm.lm.detect_pre(
        vocab_size, reweight, cc_extractor, cch, private_key,
        out_ids.to(target.device), in_ids=in_ids.to(target.device), p_logits=None,
    )
    gs = torch.gather(watermark_code.g, -1, out_ids.to(watermark_code.g.device).unsqueeze(-1)).squeeze(-1)
    Us = torch.exp(-torch.exp(-gs)).clamp(1e-10, 1 - 1e-10)
    return Us, np.asarray(skipped, dtype=bool)


def _uniforms_from_pfr(
    *,
    out_ids: torch.LongTensor,
    source_labels: List[bytes],
    masked_flags: List[bool],
    private_key: bytes,
    vocab_size: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return per-token uniforms u_t reconstructed from PFR source labels."""
    out_list = out_ids[0].tolist()
    if len(source_labels) != len(out_list):
        raise RuntimeError(
            f"label/token count mismatch: {len(source_labels)} labels, {len(out_list)} tokens"
        )
    Us = [
        _uniform_for_token(label, private_key, int(tok), vocab_size)
        for label, tok in zip(source_labels, out_list)
    ]
    skipped = np.array([masked_flags], dtype=bool)
    return torch.tensor([Us], dtype=torch.float32), skipped


def _scores_from_uniforms(
    Us: torch.Tensor,
    *,
    li_delta: float,
    pl_eps: float,
    skipped: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run U, A, Li, PL detectors on per-token uniforms (1, T) and return ANLPPT.

    `skipped` (1, T) bool mask excludes repeated-context tokens from the
    aggregate Chernoff bound -- per Aaronson detection convention these
    tokens use the same keyed noise as a previous occurrence so they are
    NOT independent samples of the watermark statistic.
    """
    Us = Us.detach().cpu().clamp(1e-10, 1 - 1e-10)
    n_tokens_total = max(int(Us.shape[-1]), 1)
    if skipped is None:
        skipped = np.zeros((1, Us.shape[-1]), dtype=bool)
    else:
        skipped = np.asarray(skipped, dtype=bool).reshape(1, -1)
        assert skipped.shape == (1, Us.shape[-1]), (
            f"skipped shape {skipped.shape} != Us shape (1, {Us.shape[-1]})"
        )
    n_added = int((~skipped).sum())
    if n_added == 0:
        zero = {"ANLPPT_U": 0.0, "ANLPPT_A": 0.0, "ANLPPT_Li": 0.0, "ANLPPT_PL": 0.0}
        return zero

    # Build a fake DeltaGumbel_WatermarkCode with a single-vocab dim so that
    # the existing scorers' gather(g, ids) recovers exactly these uniforms.
    gs = -torch.log(-torch.log(Us))                  # (1, T)
    g_fake = gs.unsqueeze(-1).contiguous()           # (1, T, 1)
    code_fake = DeltaGumbel_WatermarkCode(g_fake)
    ids_fake = torch.zeros(1, Us.shape[-1], dtype=torch.long)

    s_U = DeltaGumbel_U.from_watermarkcode(code_fake, ids_fake, skipped)
    s_A = DeltaGumbel_A.from_watermarkcode(code_fake, ids_fake, skipped)
    s_Li = DeltaGumbel_Li.builder(li_delta).from_watermarkcode(
        code_fake, ids_fake, skipped, None, None
    )
    s_PL = DeltaGumbel_PL.builder(pl_eps).from_watermarkcode(
        code_fake, ids_fake, skipped, None, None
    )
    # Normalize by n_added (independent contributing tokens) so ANLPPT is
    # comparable across decoders with different masking rates.
    return {
        "ANLPPT_U": -float(s_U.get_log_p_value()) / n_added,
        "ANLPPT_A": -float(s_A.get_log_p_value()) / n_added,
        "ANLPPT_Li": -float(s_Li.get_log_p_value()) / n_added,
        "ANLPPT_PL": -float(s_PL.get_log_p_value()) / n_added,
        "n_skipped": int(skipped.sum()),
        "n_added": n_added,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


DECODERS = {
    "basic_uwm":              run_basic_uwm,
    "mc_uwm_speed":           lambda *a, **kw: run_mc_uwm(*a, strength=False, **kw),
    "mc_uwm_strength":        lambda *a, **kw: run_mc_uwm(*a, strength=True, **kw),
    "pfr":                    run_pfr,
    "pfr_nowatermark":        run_pfr_nowm,
    "ms_pfr_cached":          run_ms_pfr_cached,
    "mpfr_torchgen_cached":   run_mpfr_torchgen_cached,
    "invariant_multi":        run_invariant_multi,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "cnn_dailymail"])
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--li-delta", type=float, default=0.5)
    ap.add_argument("--pl-eps", type=float, default=0.1)
    ap.add_argument("--target-model", type=Path,
                    default=ROOT / "model" / "Qwen2.5-7B-Instruct")
    ap.add_argument("--draft-model", type=Path,
                    default=ROOT / "model" / "Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS.keys()),
                    choices=list(DECODERS.keys()))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    print(f"loading target {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(str(args.target_model), local_files_only=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target_model),
        device_map=args.device,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        str(args.draft_model),
        device_map=args.device,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    vocab_size = int(target.config.vocab_size)

    prompts = load_prompts(args.samples, dataset=args.dataset)
    print(f"dataset={args.dataset}  prompts={len(prompts)}  lookahead={args.lookahead}  "
          f"max_new={args.max_new_tokens}  Li.Delta={args.li_delta}  PL.eps={args.pl_eps}")

    rows: Dict[str, List[dict]] = {d: [] for d in args.decoders}
    for d_name in args.decoders:
        decoder = DECODERS[d_name]
        for idx, prompt in enumerate(prompts):
            input_ids = encode_prompt(tokenizer, prompt, target.device)
            torch.manual_seed(idx + 7)
            _sync()
            t0 = time.perf_counter()
            out_ids, block_lens, pk, wm_kind, source_labels, masked_flags = decoder(
                target, draft, input_ids, args.lookahead, args.max_new_tokens, idx + 7,
            )
            _sync()
            elapsed = time.perf_counter() - t0
            n_tokens = int(out_ids.shape[-1])
            n_steps = max(len(block_lens), 1)
            aatps = n_tokens / n_steps

            if wm_kind == "PFR":
                Us, skipped = _uniforms_from_pfr(
                    out_ids=out_ids, source_labels=source_labels,
                    masked_flags=masked_flags or [False] * n_tokens,
                    private_key=pk, vocab_size=vocab_size,
                )
            else:
                Us, skipped = _uniforms_from_dg(
                    target=target, out_ids=out_ids, in_ids=input_ids, private_key=pk,
                )

            scores = _scores_from_uniforms(
                Us, li_delta=args.li_delta, pl_eps=args.pl_eps, skipped=skipped,
            )

            row = {
                "prompt_idx": idx,
                "tokens": n_tokens,
                "blocks": n_steps,
                "AATPS": aatps,
                "elapsed_sec": elapsed,
                "TR": n_tokens / elapsed if elapsed > 0 else 0.0,
                "wm_kind": wm_kind,
                **scores,
            }
            rows[d_name].append(row)
            print(f"  [{d_name}] prompt={idx} tok={n_tokens} AATPS={aatps:.3f} "
                  f"skip={scores.get('n_skipped', 0)}/{n_tokens} "
                  f"U={scores['ANLPPT_U']:.3f}  A={scores['ANLPPT_A']:.3f}  "
                  f"Li={scores['ANLPPT_Li']:.3f}  PL={scores['ANLPPT_PL']:.3f}")

    print("\n=== Mean over {} prompts ===".format(len(prompts)))
    print(f"{'decoder':<18s} {'AATPS':>7s} "
          f"{'ANLPPT_U':>10s} {'ANLPPT_A':>10s} "
          f"{'ANLPPT_Li':>11s} {'ANLPPT_PL':>11s}")
    summary = {}
    for d_name in args.decoders:
        rs = rows[d_name]
        if not rs:
            continue
        agg = {
            "AATPS": float(np.mean([r["AATPS"] for r in rs])),
            "ANLPPT_U": float(np.mean([r["ANLPPT_U"] for r in rs])),
            "ANLPPT_A": float(np.mean([r["ANLPPT_A"] for r in rs])),
            "ANLPPT_Li": float(np.mean([r["ANLPPT_Li"] for r in rs])),
            "ANLPPT_PL": float(np.mean([r["ANLPPT_PL"] for r in rs])),
            "TR": float(np.mean([r["TR"] for r in rs])),
        }
        summary[d_name] = agg
        print(f"{d_name:<18s} {agg['AATPS']:>7.3f} "
              f"{agg['ANLPPT_U']:>10.3f} {agg['ANLPPT_A']:>10.3f} "
              f"{agg['ANLPPT_Li']:>11.3f} {agg['ANLPPT_PL']:>11.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "summary": summary,
            "rows": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
