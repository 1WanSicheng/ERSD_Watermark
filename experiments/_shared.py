"""
Shared helpers for the JSON-driven single-draft and multi-draft experiment
runners.

Layout:
  - Model + tokenizer + dataset loading
  - Process-logits warper construction
  - Decoder registry (dispatch by name)
  - Per-token uniform recovery (DG / PFR)
  - ANLPPT (U / Li / PL) and KL/WS-ratio metric computation
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Late imports below depend on ROOT being on sys.path.
import unbiased_watermark as uwm  # noqa: E402
from accuwm.basic_watermark import basic_uwm_generator  # noqa: E402
from accuwm.mc import mc_sample_generator  # noqa: E402
from accuwm.mc_watermark import mc_uwm_sample_generator  # noqa: E402
from accuwm.pfr import pfr_sample_generator  # noqa: E402
from accuwm.pfr_no_watermark import pfr_no_watermark_generator  # noqa: E402
from unbiased_watermark import DeltaGumbel_WatermarkCode  # noqa: E402
from unbiased_watermark.scores import (  # noqa: E402
    DeltaGumbel_A,
    DeltaGumbel_Li,
    DeltaGumbel_PL,
    DeltaGumbel_U,
)
from unbiased_watermark.scores.pfr_aaronson import _uniform_for_token  # noqa: E402
from unbiased_watermark.scores.pfr_watermark_strength import (  # noqa: E402
    compute_pfr_watermark_strength_from_sequence,
)

# Multi-draft cached variants under MPFR_spec/.
sys.path.insert(0, str(ROOT / "MPFR_spec"))
from multi_draft_pfr_batched_cached import (  # noqa: E402
    multi_draft_pfr_batched_cached_sample_generator,
)
from mpfr_batched_torchgen_cached import (  # noqa: E402
    finite_multi_draft_pfr_cached_sample_generator,
)


DEFAULT_TARGET_MODEL = ROOT / "model" / "Qwen2.5-7B-Instruct"
DEFAULT_DRAFT_MODEL = ROOT / "model" / "Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return p


def resolve_model_id(value, default):
    """Resolve a model identifier: HF Hub ID (e.g. ``huggyllama/llama-7b``)
    pass-through, or a Path for local directories.

    A value is treated as a Hub ID iff it matches ``<org>/<name>`` (single ``/``,
    no ``\\``), and ``ROOT/<value>`` does not exist on disk.
    """
    if value is None:
        return default
    if isinstance(value, str) and "\\" not in value and value.count("/") == 1:
        candidate = ROOT / value
        if not candidate.exists():
            return value
    return resolve_path(value, default)


def private_key_from_str(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, list):
        return bytes(value)
    return b"1234"


def seeded_private_key(seed: int, base_key: bytes) -> bytes:
    return int(seed).to_bytes(8, "big") + base_key


# ---------------------------------------------------------------------------
# Models / dataset
# ---------------------------------------------------------------------------


def load_model(model_path, device: str):
    is_local = isinstance(model_path, Path)
    kwargs = dict(
        pretrained_model_name_or_path=str(model_path),
        device_map=device,
        local_files_only=is_local,
        low_cpu_mem_usage=True,
    )
    if device.startswith("cuda"):
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    return AutoModelForCausalLM.from_pretrained(**kwargs).eval()


def load_models_and_tokenizer(config: dict, device: str | None = None):
    device = device or config.get(
        "device", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    target_model = resolve_model_id(
        config.get("target_model"), DEFAULT_TARGET_MODEL
    )
    draft_model = resolve_model_id(
        config.get("draft_model"), DEFAULT_DRAFT_MODEL
    )
    target = load_model(target_model, device)
    draft = load_model(draft_model, device)
    tokenizer = AutoTokenizer.from_pretrained(
        str(target_model), local_files_only=isinstance(target_model, Path)
    )
    return target, draft, tokenizer, device


def encode_prompt(tokenizer, prompt, device, use_chat_template: bool = True):
    """Encode a chat-style prompt to input_ids.

    use_chat_template: when False, ignore any chat_template the tokenizer
    carries and emit the last user message as plain text.  Required for base
    models (e.g. huggyllama/llama-7b) whose tokenizer inherits a Llama-2-Chat
    template — feeding them ``[INST] <<SYS>> ...`` tokens as prompt drives
    them to generate chat-template artifacts in the output.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
    user_msgs = [m["content"] for m in prompt if m["role"] == "user"]
    text = (
        user_msgs[-1]
        if user_msgs
        else "\n".join(m["content"] for m in prompt)
    )
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def load_prompts(dataset: str, n: int) -> List[List[dict]]:
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
                {
                    "role": "user",
                    "content": (
                        "Summarize the following article in 3-5 sentences:\n\n"
                        f"{row['article'][:1500]}"
                    ),
                },
            ]
            for row in ds
        ]
    if dataset == "eli5":
        ds = load_dataset("sentence-transformers/eli5", split="train")
        ds = ds.shuffle(seed=42).select(range(n))
        return [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": (
                        "Please explain like I'm five:\n\n"
                        f"{row['question']}"
                    ),
                },
            ]
            for row in ds
        ]
    raise ValueError(f"unknown dataset: {dataset}")


def build_process_logits_kwargs(spec: dict | None) -> dict:
    """Build kwargs for accuwm.utils.process_logits from a config block.

    spec is e.g. {"top_k": 50, "top_p": 1.0, "temperature": 1.0}.
    """
    spec = spec or {}
    top_k = int(spec.get("top_k", 0) or 0)
    top_p = float(spec.get("top_p", 1.0) or 1.0)
    temperature = float(spec.get("temperature", 1.0) or 1.0)
    parts = []
    if temperature != 1.0:
        parts.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        parts.append(TopKLogitsWarper(top_k))
    if 0.0 < top_p < 1.0:
        parts.append(TopPLogitsWarper(top_p))
    if not parts:
        return {}
    warper = LogitsProcessorList(parts)

    def warp(input_ids, logits):
        return warper(input_ids, logits)

    return {"logits_warper": warp}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Decoder registry — single-draft (B=1)
# ---------------------------------------------------------------------------


@dataclass
class GenResult:
    out_ids: torch.LongTensor                # (1, T) on CPU
    block_lens: List[int]
    elapsed_sec: float
    private_key: bytes
    wm_kind: str                              # 'DeltaGumbel' | 'PFR' | 'none'
    source_labels: Optional[List[bytes]] = None
    masked_flags: Optional[List[bool]] = None


def _drain(gen, max_length: int, *, collect_labels: bool = False) -> Tuple[
    torch.LongTensor, List[int], List[bytes], List[bool]
]:
    chunks = []
    block_lens: List[int] = []
    labels: List[bytes] = []
    masked: List[bool] = []
    generated = 0
    for step in gen:
        if isinstance(step, tuple) and len(step) == 3:
            ids, _logp, meta = step
        elif isinstance(step, tuple):
            ids, meta = step[0], {}
        else:
            ids, meta = step, {}
        bl = ids.shape[1]
        if generated + bl > max_length:
            bl = max_length - generated
        if bl <= 0:
            break
        chunks.append(ids[:, :bl].detach().cpu())
        block_lens.append(int(bl))
        if collect_labels and "labels" in meta:
            for lab in meta["labels"][:bl]:
                labels.append(lab.source_label)
                masked.append(bool(getattr(lab, "masked", False)))
        generated += bl
        if generated >= max_length:
            break
    out_ids = (
        torch.cat(chunks, dim=1)
        if chunks
        else torch.empty((1, 0), dtype=torch.long)
    )
    return out_ids, block_lens, labels, masked


def _run_mc(target, draft, input_ids, *, lookahead, max_length, seed,
            base_key, plk, **_):
    pk = seeded_private_key(seed, base_key)
    gen = mc_sample_generator(
        model=target, ref_model=draft, input_ids=input_ids,
        n=lookahead, process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    return out, blocks, pk, "none", None, None


def _run_basic_uwm(target, draft, input_ids, *, lookahead, max_length, seed,
                   base_key, plk, **_):
    """Autoregressive UWM with n=1 so AATPS == 1 (lookahead is ignored)."""
    pk = seeded_private_key(seed, base_key)
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    gen = basic_uwm_generator(
        reweight=reweight, cc_extractor=cc_extractor, cch=cch,
        private_key=pk, model=target, input_ids=input_ids, n=1,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    return out, blocks, pk, "DeltaGumbel", None, None


def _run_mc_uwm(target, draft, input_ids, *, lookahead, max_length, seed,
                base_key, plk, strength: bool, **_):
    pk = seeded_private_key(seed, base_key)
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    gen = mc_uwm_sample_generator(
        reweight=reweight, cc_extractor=cc_extractor, cch=cch,
        private_key=pk, reweight_in_mc=strength,
        model=target, ref_model=draft, input_ids=input_ids, n=lookahead,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    return out, blocks, pk, "DeltaGumbel", None, None


def _run_pfr(target, draft, input_ids, *, lookahead, max_length, seed,
             base_key, plk, **_):
    pk = seeded_private_key(seed, base_key)
    gen = pfr_sample_generator(
        model=target, ref_model=draft, input_ids=input_ids,
        n=lookahead, max_length=max_length, private_key=pk,
        labeler_mode="context_code", process_logits_kwargs=plk,
    )
    out, blocks, lbls, msk = _drain(gen, max_length, collect_labels=True)
    return out, blocks, pk, "PFR", lbls, msk


def _run_pfr_nowm(target, draft, input_ids, *, lookahead, max_length, seed,
                  base_key, plk, **_):
    pk = seeded_private_key(seed, base_key)
    gen = pfr_no_watermark_generator(
        model=target, ref_model=draft, input_ids=input_ids,
        n=lookahead, max_length=max_length, process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    return out, blocks, pk, "DeltaGumbel", None, None  # H0 control under DG pivot


SINGLE_DRAFT_DECODERS: Dict[str, Callable] = {
    "mc":               _run_mc,
    "basic_uwm":        _run_basic_uwm,
    "mc_uwm_speed":     lambda *a, **kw: _run_mc_uwm(*a, strength=False, **kw),
    "mc_uwm_strength":  lambda *a, **kw: _run_mc_uwm(*a, strength=True, **kw),
    "pfr":              _run_pfr,
    "pfr_no_watermark": _run_pfr_nowm,
}

# basic_uwm doesn't vary with lookahead.
LOOKAHEAD_INVARIANT = {"basic_uwm"}


# ---------------------------------------------------------------------------
# Decoder registry — multi-draft (B>=1)
# ---------------------------------------------------------------------------


def _prefix_labeler_label(prefix_ids: torch.LongTensor) -> bytes:
    tokens = prefix_ids.detach().cpu().numpy().astype(np.int32, copy=False)
    payload = tokens.tobytes()
    length = prefix_ids.shape[-1].to_bytes(8, byteorder="big", signed=False)
    return length + payload


def _mpfr_direct_label(prefix_ids: torch.LongTensor) -> bytes:
    tokens = prefix_ids.detach().cpu().tolist()
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in tokens
    )


def _rebuild_labels_for_prefix_scheme(in_ids, out_ids, label_fn) -> List[bytes]:
    full = torch.cat([in_ids[0], out_ids[0].to(in_ids.device)], dim=0)
    T_prompt = int(in_ids.shape[-1])
    T_out = int(out_ids.shape[-1])
    labels: List[bytes] = []
    for i in range(T_out):
        prefix = full[: T_prompt + i]
        labels.append(label_fn(prefix))
    return labels


_DEFER_PREFIX_LABELS = "DEFER_PREFIX_LABELS"


def finalize_labels(src_labels, masked_flags, out_ids):
    """Resolve any deferred label-rebuild sentinel from a multi-draft decoder.

    Decoders for the prefix-scheme PFR families return a sentinel tuple in place
    of the per-token label list so the O(T) Python rebuild stays *outside* the
    timing window measured for token_rate. Single-draft / non-PFR decoders return
    plain lists (or None) and are passed through unchanged.
    """
    if (
        isinstance(src_labels, tuple)
        and len(src_labels) >= 3
        and src_labels[0] == _DEFER_PREFIX_LABELS
    ):
        _, in_ids, label_fn = src_labels[:3]
        labels = _rebuild_labels_for_prefix_scheme(in_ids, out_ids, label_fn)
        return labels, [False] * len(labels)
    return src_labels, masked_flags


def _run_ms_pfr_cached(target, draft, input_ids, *, lookahead, max_length,
                      seed, base_key, plk, num_drafts: int, **_):
    pk = seeded_private_key(seed, base_key)
    gen = multi_draft_pfr_batched_cached_sample_generator(
        model=target, ref_model=draft, input_ids=input_ids,
        n=lookahead, max_length=max_length, num_drafts=num_drafts,
        private_key=pk, return_meta=True,
        process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    sentinel = (_DEFER_PREFIX_LABELS, input_ids, _prefix_labeler_label)
    return out, blocks, pk, "PFR", sentinel, None


def _run_mpfr_torchgen_cached(target, draft, input_ids, *, lookahead,
                              max_length, seed, base_key, plk,
                              num_drafts: int, **_):
    pk = seeded_private_key(seed, base_key)
    gen = finite_multi_draft_pfr_cached_sample_generator(
        model=target, ref_model=draft, input_ids=input_ids,
        n=lookahead, max_length=max_length, num_drafts=num_drafts,
        private_key=pk, return_meta=True,
        process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    sentinel = (_DEFER_PREFIX_LABELS, input_ids, _mpfr_direct_label)
    return out, blocks, pk, "PFR", sentinel, None


def _run_invariant_or_strong_multi(target, draft, input_ids, *,
                                   lookahead, max_length, seed, base_key,
                                   num_drafts: int, strategy_kind: str,
                                   **_):
    """Wrapper for SpeculativeDecoding.strategy.{Invariant,Strong}MultiDraftStrategy
    + InvariantGenerator.  Both use unkeyed pre-sampled randomness, so no
    recoverable watermark."""
    sys.path.insert(0, str(ROOT / "SpeculativeDecoding"))
    from strategy import (  # noqa: E402
        InvariantMultiDraftStrategy,
        StrongMultiDraftStrategy,
    )
    from generator import InvariantGenerator  # noqa: E402

    eff_vocab = min(int(target.config.vocab_size), int(draft.config.vocab_size))

    class _StubTok:
        vocab_size = eff_vocab

    if strategy_kind == "invariant":
        strategy = InvariantMultiDraftStrategy(
            target=target, drafter=draft, tokenizer=_StubTok(),
            max_draft_len=lookahead, max_num_drafts=num_drafts,
        )
    elif strategy_kind == "strong":
        strategy = StrongMultiDraftStrategy(
            target=target, drafter=draft, tokenizer=_StubTok(),
            max_draft_len=lookahead, max_num_drafts=num_drafts,
        )
    else:
        raise ValueError(f"unknown strategy_kind {strategy_kind}")

    generator = InvariantGenerator(strategy)
    eos = int(getattr(target.config, "eos_token_id", 0) or 0)
    out_full = generator(
        input_ids=input_ids, eos_token_id=eos,
        max_new_tokens=max_length, temperature=1.0,
    )
    L_in = int(input_ids.shape[-1])
    out = out_full.sequences[:, L_in:].detach().cpu()
    n_tokens = int(out.shape[-1])
    n_inv = int(out_full.num_invocations)
    block_lens = [max(n_tokens // max(n_inv, 1), 1)] * n_inv
    pk = seeded_private_key(seed, base_key)
    # No recoverable watermark; report under DG pivot for H0 control.
    return out, block_lens, pk, "DeltaGumbel", None, None


MULTI_DRAFT_DECODERS: Dict[str, Callable] = {
    "ms_pfr_cached":         _run_ms_pfr_cached,
    "mpfr_torchgen_cached":  _run_mpfr_torchgen_cached,
    "invariant_multi":       lambda *a, **kw: _run_invariant_or_strong_multi(
                                  *a, strategy_kind="invariant", **kw),
    "strong_multi":          lambda *a, **kw: _run_invariant_or_strong_multi(
                                  *a, strategy_kind="strong", **kw),
}


# ---------------------------------------------------------------------------
# Per-token uniforms u_t recovery (DG / PFR)
# ---------------------------------------------------------------------------


def uniforms_from_dg(
    *, target, out_ids, in_ids, private_key,
) -> Tuple[torch.Tensor, np.ndarray]:
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    vocab_size = int(target.config.vocab_size)
    _, _, _, watermark_code, skipped = uwm.lm.detect_pre(
        vocab_size, reweight, cc_extractor, cch, private_key,
        out_ids.to(target.device), in_ids=in_ids.to(target.device),
        p_logits=None,
    )
    gs = torch.gather(
        watermark_code.g, -1,
        out_ids.to(watermark_code.g.device).unsqueeze(-1),
    ).squeeze(-1)
    Us = torch.exp(-torch.exp(-gs)).clamp(1e-10, 1 - 1e-10)
    return Us, np.asarray(skipped, dtype=bool)


def uniforms_from_pfr(
    *, out_ids, source_labels, masked_flags, private_key, vocab_size,
) -> Tuple[torch.Tensor, np.ndarray]:
    out_list = out_ids[0].tolist()
    if len(source_labels) != len(out_list):
        raise RuntimeError(
            f"label/token count mismatch: {len(source_labels)} vs {len(out_list)}"
        )
    Us = [
        _uniform_for_token(label, private_key, int(tok), vocab_size)
        for label, tok in zip(source_labels, out_list)
    ]
    skipped = np.array([masked_flags or [False] * len(Us)], dtype=bool)
    return torch.tensor([Us], dtype=torch.float32), skipped


# ---------------------------------------------------------------------------
# Metric: ANLPPT (U / Li / PL)
# ---------------------------------------------------------------------------


def anlppt_metrics(
    Us: torch.Tensor,
    *,
    li_delta: float,
    pl_eps: float,
    skipped: Optional[np.ndarray] = None,
    variants: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute ANLPPT-{U, A, Li, PL} from per-token uniforms (1, T).

    skipped (1, T) excludes repeated-context tokens from the Chernoff bound.
    """
    Us = Us.detach().cpu().clamp(1e-10, 1 - 1e-10)
    n_tokens_total = max(int(Us.shape[-1]), 1)
    if skipped is None:
        skipped = np.zeros((1, Us.shape[-1]), dtype=bool)
    else:
        skipped = np.asarray(skipped, dtype=bool).reshape(1, -1)
        if skipped.shape != (1, Us.shape[-1]):
            raise ValueError(
                f"skipped shape {skipped.shape} != (1, {Us.shape[-1]})"
            )
    n_added = int((~skipped).sum())
    if n_added == 0:
        return {
            "ANLPPT_U": 0.0, "ANLPPT_A": 0.0,
            "ANLPPT_Li": 0.0, "ANLPPT_PL": 0.0,
            "n_skipped": int(skipped.sum()), "n_added": 0,
        }

    gs = -torch.log(-torch.log(Us))
    g_fake = gs.unsqueeze(-1).contiguous()
    code_fake = DeltaGumbel_WatermarkCode(g_fake)
    ids_fake = torch.zeros(1, Us.shape[-1], dtype=torch.long)

    out: Dict[str, float] = {}
    if variants is None:
        variants = ["U", "A", "Li", "PL"]
    if "U" in variants:
        s = DeltaGumbel_U.from_watermarkcode(code_fake, ids_fake, skipped)
        out["ANLPPT_U"] = -float(s.get_log_p_value()) / n_added
    if "A" in variants:
        s = DeltaGumbel_A.from_watermarkcode(code_fake, ids_fake, skipped)
        out["ANLPPT_A"] = -float(s.get_log_p_value()) / n_added
    if "Li" in variants:
        s = DeltaGumbel_Li.builder(li_delta).from_watermarkcode(
            code_fake, ids_fake, skipped, None, None
        )
        out["ANLPPT_Li"] = -float(s.get_log_p_value()) / n_added
    if "PL" in variants:
        s = DeltaGumbel_PL.builder(pl_eps).from_watermarkcode(
            code_fake, ids_fake, skipped, None, None
        )
        out["ANLPPT_PL"] = -float(s.get_log_p_value()) / n_added
    out["n_skipped"] = int(skipped.sum())
    out["n_added"] = n_added
    return out


# ---------------------------------------------------------------------------
# Metric: KL/WS ratio (Definition 3.1, delta-conditional empirical estimator)
# ---------------------------------------------------------------------------


@torch.no_grad()
def kl_ws_ratio_pfr(
    *, target_model, full_ids, prompt_length, private_key,
    process_logits_kwargs: Optional[dict],
    labeler_mode: str = "context_code",
) -> Dict[str, float]:
    """PFR-detector-recovery empirical estimator (delta-conditional case).

    The per-token cc must match the gen-side labeler convention or the
    recovered PFR winner won't equal the actually-emitted token and the
    WS estimate becomes meaningless.  ``pfr_sample_generator`` uses
    ``mode="context_code"`` by default, so we mirror it here.

    Returns dict with ws_sum, h_sum, ratio, num_scored, masked_ratio.
    """
    from accuwm.pfr import build_default_labeler
    score_labeler = build_default_labeler(mode=labeler_mode)
    score = compute_pfr_watermark_strength_from_sequence(
        full_ids=full_ids,
        prompt_length=prompt_length,
        target_model=target_model,
        labeler=score_labeler,
        private_key=private_key,
        process_logits_kwargs=process_logits_kwargs,
    )
    return {
        "kl_ws_ratio": float(score["ratio"]),
        "kl_ws_mean": float(score["WS_PFR_hat"]),
        "kl_h_mean": float(score["H_P_hat"]),
        "kl_ws_sum": float(score["ws_sum"]),
        "kl_h_sum": float(score["entropy_sum"]),
        "kl_num_scored": int(score["num_scored"]),
        "kl_masked_ratio": float(score["masked_ratio"]),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_rows(rows: List[dict], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if not vals:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out
