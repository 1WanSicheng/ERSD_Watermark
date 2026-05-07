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
import torch.nn.functional as F
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
from accuwm.utils import process_logits  # noqa: E402
from unbiased_watermark.scores.pfr_watermark_strength import (  # noqa: E402
    compute_pfr_watermark_strength_from_sequence,
)

# Multi-draft cached variants under MPFR_spec/.
sys.path.insert(0, str(ROOT / "MPFR_spec"))
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
    is_local = isinstance(target_model, Path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(target_model), local_files_only=is_local
        )
    except Exception:
        # Fast tokenizer conversion can fail on SentencePiece-only repos
        # (e.g. lmsys/vicuna-7b-v1.5 ships only tokenizer.model, no
        # tokenizer.json). Fall back to the slow Python tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            str(target_model), local_files_only=is_local, use_fast=False
        )
    _maybe_inject_chat_template(tokenizer, target_model)
    return target, draft, tokenizer, device


# Vicuna v1.1+ chat format (used by lmsys/vicuna-* and FastChat):
# {system_prompt} USER: {q1} ASSISTANT: {a1}</s>USER: {q2} ASSISTANT: ...
# When the model ships without a chat_template attribute (vicuna repos
# typically don't), apply_chat_template raises and our experiment silently
# falls back to plain user-message text — defeating the whole point of
# evaluating the model in instruction-following mode. Inject the template
# explicitly for known repos.
_VICUNA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + ' ' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'ASSISTANT: ' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)


def _maybe_inject_chat_template(tokenizer, model_id) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    name = str(model_id).lower()
    if "vicuna" in name:
        tokenizer.chat_template = _VICUNA_CHAT_TEMPLATE


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


def load_references(dataset: str, n: int) -> Optional[List[str]]:
    """Return aligned-by-index reference texts for ROUGE-L scoring, or None
    if the dataset has no canonical reference field.

    Used by the ROUGE-L-vs-reference quality audit (Exp 1 in the ablation
    plan).  Indexing matches ``load_prompts(dataset, n)`` so row.prompt_idx
    maps directly into the returned list.
    """
    if dataset == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000).select(range(n))
        return [str(row.get("highlights", "")) for row in ds]
    return None


def _rouge_l_f1(a_tokens: List[int], b_tokens: List[int]) -> float:
    """ROUGE-L F1 over token-id sequences (no external dependency).

    LCS-based recall/precision, F1 = 2PR/(P+R).  Operates on token ids
    directly so it's tokenizer-agnostic for cross-method comparisons; for
    "vs gold reference" we tokenize the reference text with the same
    tokenizer used to generate, so both sides are on the same id space.
    """
    if not a_tokens or not b_tokens:
        return 0.0
    n, m = len(a_tokens), len(b_tokens)
    if n < m:
        a_tokens, b_tokens = b_tokens, a_tokens
        n, m = m, n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] > prev[j] else prev[j]
        prev = cur
    L = prev[m]
    if L == 0:
        return 0.0
    p = L / len(b_tokens)
    r = L / len(a_tokens)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


def rouge_l_against_reference(
    *, tokenizer, output_ids: List[int], reference_text: str,
) -> float:
    """ROUGE-L F1 between generated output token ids and a reference text
    (the reference is tokenized with the same tokenizer)."""
    if not reference_text:
        return float("nan")
    ref_tok_ids = tokenizer(reference_text, add_special_tokens=False)["input_ids"]
    return _rouge_l_f1(list(output_ids), list(ref_tok_ids))


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
    if dataset == "cnn_paper_summarization":
        # Paper-exact prompt format from prior work [hu2024inevitable]
        # Table 1. On a base llama, this template lands
        # the model in summarization mode (recognized as instruction-tuned
        # demonstration) — much lower per-token entropy than the
        # "Summarize the following article in 3-5 sentences:" prompt, which
        # base models treat as plain text continuation.
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000).select(range(n))
        return [
            [
                {
                    "role": "user",
                    "content": (
                        "System:Summarize the following article.\n"
                        f"INPUT:{row['article'][:1000]}\n"
                        "OUTPUT:"
                    ),
                },
            ]
            for row in ds
        ]
    if dataset == "cnn_dailymail_basefmt":
        # Same instruction content as ``cnn_dailymail`` ("in 3-5 sentences")
        # but wrapped in the System:/INPUT:/OUTPUT: section markers from
        # the [hu2024inevitable] protocol so base llama recognises it as an
        # instruction-tuned demonstration (lower entropy, no early-EOS
        # pathology). Used when running base targets with the same
        # instruction wording we use elsewhere; instruction-tuned targets
        # should keep using ``cnn_dailymail`` with their native chat template.
        ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
        ds = ds.filter(lambda x: len(x["article"]) < 3000).select(range(n))
        return [
            [
                {
                    "role": "user",
                    "content": (
                        "System:Summarize the following article in 3-5 sentences.\n"
                        f"INPUT:{row['article'][:1500]}\n"
                        "OUTPUT:"
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

    spec is e.g. ``{"top_k": 50, "top_p": 1.0, "temperature": 1.0,
    "draft_temperature": 1.5}``.

    The returned dict packs:
    - ``logits_warper``: warper using the *target* temperature (consumed by
      single-draft warper-style code paths and by target forwards in PFR).
    - ``draft_logits_warper`` (only if ``draft_temperature`` differs from
      ``temperature``): warper using the *drafter* temperature.  PFR /
      mc_uwm drafter forwards swap this in as their ``logits_warper`` for
      drafter-invariance ablations.
    - scalar fields ``temperature``, ``top_k``, ``top_p``,
      ``draft_temperature`` (when set): consumed by multi-draft scalar
      paths via ``MPFR_spec/mpfr_direct_optimized._temperature_for_draft``
      etc.
    """
    spec = spec or {}
    top_k = int(spec.get("top_k", 0) or 0)
    top_p = float(spec.get("top_p", 1.0) or 1.0)
    temperature = float(spec.get("temperature", 1.0) or 1.0)
    draft_temperature = spec.get("draft_temperature", None)
    if draft_temperature is not None:
        draft_temperature = float(draft_temperature)

    out: dict = {
        "temperature": temperature,
        "top_k": top_k if top_k > 0 else 0,
        "top_p": top_p,
    }
    if draft_temperature is not None:
        out["draft_temperature"] = draft_temperature

    def _build_warper(temp: float):
        parts = []
        if temp != 1.0:
            parts.append(TemperatureLogitsWarper(temp))
        if top_k > 0:
            parts.append(TopKLogitsWarper(top_k))
        if 0.0 < top_p < 1.0:
            parts.append(TopPLogitsWarper(top_p))
        if not parts:
            return None
        warper = LogitsProcessorList(parts)
        def warp(input_ids, logits):
            return warper(input_ids, logits)
        return warp

    target_warp = _build_warper(temperature)
    if target_warp is not None:
        out["logits_warper"] = target_warp

    if draft_temperature is not None and draft_temperature != temperature:
        draft_warp = _build_warper(draft_temperature)
        if draft_warp is not None:
            out["draft_logits_warper"] = draft_warp

    return out


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
        process_logits_kwargs=plk,
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
        process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    return out, blocks, pk, "DeltaGumbel", None, None


def _mc_uwm_pseudo_r_key(seed: int, base_key: bytes) -> bytes:
    """Second key (zeta_R) for the pseudorandom-acceptance step in
    Algorithm 1 of arXiv:2602.01428.  Distinct from the watermark key
    (zeta_D / zeta_T = `private_key`) so the acceptance pseudorandomness is
    decorrelated from the watermark itself."""
    return b"PSEUDO_R::" + int(seed).to_bytes(8, "big") + base_key


def _run_mc_uwm_pseudo_r(target, draft, input_ids, *, lookahead, max_length,
                          seed, base_key, plk, **_):
    """Algorithm 1 of arXiv:2602.01428 instantiated with DeltaGumbel reweight.

    speed-mode topology (target verify uses raw P, not P_zeta) plus:
      - watermarked residual sampled from (P - Q)_+ reweighted under zeta_T
        (= private_key, same as draft), via mc_sample_synthid path with
        DeltaGumbel reweight (NOT SynthID — function name is misleading).
      - pseudorandom acceptance variable u_t = G(zeta_R_t) where zeta_R is
        a separate key from the watermark, replacing torch.rand() in the
        accept/reject decision.

    Theorem 4.1 of the paper claims this achieves max sampling efficiency
    (1 - TV(Q,P)) AND max watermark strength (Ent(P)) simultaneously.
    """
    pk = seeded_private_key(seed, base_key)
    mc_pk = _mc_uwm_pseudo_r_key(seed, base_key)
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    gen = mc_uwm_sample_generator(
        reweight=reweight, cc_extractor=cc_extractor, cch=cch,
        private_key=pk, reweight_in_mc=False,
        mc_synthid=True, mc_private_key=mc_pk, psedo_r=True,
        model=target, ref_model=draft, input_ids=input_ids, n=lookahead,
        process_logits_kwargs=plk,
    )
    out, blocks, _, _ = _drain(gen, max_length)
    # Bundle both keys so the dual-key DG detector can score the residual-
    # token contribution under zeta_R as well as the coupled tokens under
    # zeta_T (= pk).  _DualPK is bytes-compatible for any single-key path.
    return out, blocks, _DualPK(pk, mc_pk), "DeltaGumbelDual", None, None


def _run_pfr(target, draft, input_ids, *, lookahead, max_length, seed,
             base_key, plk, **_):
    """Single-draft PFR with PrevN(3) context_code labeler.

    Aligned with the DG-family decoders (basic_uwm, mc, mc_uwm_*) which all
    derive per-token randomness from PrevN(3) context codes.  This makes
    cross-method TPR comparisons in single_draft directly comparable.
    """
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
    # Algorithm 1 of arXiv:2602.01428 with DeltaGumbel reweight: speed-mode
    # topology + watermarked residual + pseudorandom acceptance via 2nd key.
    # Same DG-UWM detector as the rest of this row (Aaronson Gamma-tail).
    "mc_uwm_pseudo_r":  _run_mc_uwm_pseudo_r,
    "pfr":              _run_pfr,
    "pfr_no_watermark": _run_pfr_nowm,
}

# basic_uwm doesn't vary with lookahead.
LOOKAHEAD_INVARIANT = {"basic_uwm"}


# ---------------------------------------------------------------------------
# Decoder registry — multi-draft (B>=1)
# ---------------------------------------------------------------------------


def _mpfr_direct_label(prefix_ids: torch.LongTensor) -> bytes:
    tokens = prefix_ids.detach().cpu().tolist()
    return b"MPFR_DIRECT_CLOCK_V1" + b"".join(
        int(t).to_bytes(8, "big", signed=True) for t in tokens
    )


def make_context_code_label_fn(n: int = 3, mask_prefix: bytes = b"repeat::"):
    """Build a stateful per-token label function matching the gen-side
    ``RepeatedContextMaskingLabeler(ContextCodeLabeler(PrevN_ContextCodeExtractor(n)))``
    behavior used by ``pfr_sample_generator(labeler_mode="context_code", ...)``.

    Returns a closure that takes a prefix_ids tensor and emits the matching
    label bytes (with masking prefix when the same base label has been seen
    before in this generation).  Each call must use a FRESH closure per
    prompt so the masking state does not leak across prompts.
    """
    seen: set = set()

    def label_fn(prefix_ids: torch.LongTensor) -> bytes:
        # PrevN(n): take last n tokens of the prefix as the base label.
        tail = prefix_ids[..., -n:].detach().cpu().numpy().astype(np.int32, copy=False)
        base_label = tail.tobytes()
        if base_label in seen:
            return mask_prefix + base_label
        seen.add(base_label)
        return base_label

    return label_fn


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


def _run_invariant_multi(target, draft, input_ids, *,
                         lookahead, max_length, seed, base_key,
                         num_drafts: int, **_):
    """Wrapper for accuwm.invariant_multi.InvariantMultiDraftStrategy
    + InvariantGenerator.  Uses unkeyed pre-sampled randomness, so no
    recoverable watermark."""
    from accuwm.invariant_multi import (  # noqa: E402
        InvariantMultiDraftStrategy,
        InvariantGenerator,
    )

    eff_vocab = min(int(target.config.vocab_size), int(draft.config.vocab_size))

    class _StubTok:
        vocab_size = eff_vocab

    strategy = InvariantMultiDraftStrategy(
        target=target, drafter=draft, tokenizer=_StubTok(),
        max_draft_len=lookahead, max_num_drafts=num_drafts,
    )

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
    "mpfr_torchgen_cached":  _run_mpfr_torchgen_cached,
    "invariant_multi":       _run_invariant_multi,
}


# ---------------------------------------------------------------------------
# Per-token uniforms u_t recovery (DG / PFR)
# ---------------------------------------------------------------------------


# Sidecar map from primary key bytes -> residual mc_pk bytes.  Used by the
# dual-key DG detector to look up zeta_R given zeta_T (= the primary
# private_key returned by the decoder).  Avoids subclassing bytes (which
# does not permit instance attributes) and avoids changing the 6-tuple
# decoder return contract.
_DUAL_PK_MAP: Dict[bytes, bytes] = {}


def _DualPK(pk: bytes, mc_pk: bytes) -> bytes:
    """Register (pk -> mc_pk) and return pk unchanged so all single-key
    paths continue to see plain bytes; dual-key detectors look mc_pk up
    via ``dual_pk_lookup(pk)``."""
    pk_b = bytes(pk)
    _DUAL_PK_MAP[pk_b] = bytes(mc_pk)
    return pk_b


def dual_pk_lookup(pk: bytes):
    return _DUAL_PK_MAP.get(bytes(pk))


def _dg_uniforms_for_key(
    *, target, out_ids, in_ids, private_key,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute per-token DG uniforms u_t under one private_key."""
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


def uniforms_from_dg(
    *, target, out_ids, in_ids, private_key,
) -> Tuple[torch.Tensor, np.ndarray]:
    return _dg_uniforms_for_key(
        target=target, out_ids=out_ids, in_ids=in_ids,
        private_key=private_key,
    )


def _r_values_under_key(
    *, out_ids, in_ids, private_key,
) -> np.ndarray:
    """Reconstruct the pseudorandom acceptance variables r_t = G(zeta_R)_t
    used by Algorithm 1 (arXiv:2602.01428).  Mirrors improving_KL's
    ``unbiased_watermark.lm.get_r_values``: PrevN(3) cc, sha256-seeded
    Generator.random() per token."""
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=out_ids.shape[:-1])
    ids = out_ids if in_ids is None else torch.cat(
        [in_ids.to(out_ids.device), out_ids], dim=-1
    )
    cc_s = []
    for i in range(ids.shape[-1] - out_ids.shape[-1], ids.shape[-1]):
        cc_step = cch.step(cc_extractor, ids[..., :i])
        cc = cc_step[0] if isinstance(cc_step, tuple) else cc_step
        cc_s.append(cc)
    cc = np.stack(cc_s, axis=-1)
    pk = bytes(private_key)
    r_values = np.empty(cc.shape, dtype=np.float32)
    for index in np.ndindex(r_values.shape):
        rng = _get_rng_local(cc[index], pk)
        r_values[index] = rng.random()
    return r_values


def _get_rng_local(*bs):
    """Local sha256-seeded numpy Generator (matches accuwm.utils.get_rng on
    the pod and improving_KL's get_rng).  Bytes-coerce non-bytes inputs."""
    import hashlib
    m = hashlib.sha256()
    for b in bs:
        if isinstance(b, str):
            b = b.encode("utf-8")
        elif hasattr(b, "item"):
            b = b.item()
        if not isinstance(b, (bytes, bytearray, memoryview)):
            b = bytes(b)
        m.update(b)
    seed = int.from_bytes(m.digest(), "big") % (2**32 - 1)
    return np.random.default_rng(seed)


def dg_dual_key_components(
    *, target, out_ids, in_ids, private_key, mc_private_key,
) -> Dict[str, np.ndarray]:
    """Return (Us_pk, Us_mc, r_values, skipped_pk, skipped_mc) for the
    dual-key DG detector.  Lets callers post-hoc sweep mixing thresholds
    or implement Ars-tau without re-running detection.  Shapes (1, T)
    each (or (T,) for 1D)."""
    Us_pk, skipped_pk = _dg_uniforms_for_key(
        target=target, out_ids=out_ids, in_ids=in_ids,
        private_key=bytes(private_key),
    )
    Us_mc, skipped_mc = _dg_uniforms_for_key(
        target=target, out_ids=out_ids, in_ids=in_ids,
        private_key=bytes(mc_private_key),
    )
    r_values = _r_values_under_key(
        out_ids=out_ids, in_ids=in_ids, private_key=bytes(mc_private_key),
    )
    return {
        "Us_pk": Us_pk.detach().cpu().numpy().astype(np.float32),
        "Us_mc": Us_mc.detach().cpu().numpy().astype(np.float32),
        "r_values": np.asarray(r_values, dtype=np.float32),
        "skipped_pk": np.asarray(skipped_pk, dtype=bool),
        "skipped_mc": np.asarray(skipped_mc, dtype=bool),
    }


def uniforms_from_dg_dual_key(
    *, target, out_ids, in_ids, private_key, mc_private_key,
    accept_threshold: float = 0.5,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Dual-key Aaronson detector for Algorithm 1 (arXiv:2602.01428).

    Token-level mix using r_values: at gen time a token is accepted iff
    ``r_t = G(zeta_R)_t <= prob_ratio_t``, where r_t is the pseudorandom
    acceptance variable.  Without prob_ratio at detection time we use a
    fixed threshold ``accept_threshold`` (rough estimate of mean accept
    probability).  Tokens with r_t > threshold are predicted to be
    residual-resampled (watermarked under zeta_R = mc_private_key); the
    rest are predicted accepted (watermarked under zeta_T = private_key).

    Returns the SELECTED uniforms per token, shape (1, T), so the caller
    can run the standard Aaronson Gamma-tail test on top.
    """
    Us_pk, skipped_pk = _dg_uniforms_for_key(
        target=target, out_ids=out_ids, in_ids=in_ids,
        private_key=bytes(private_key),
    )
    Us_mc, skipped_mc = _dg_uniforms_for_key(
        target=target, out_ids=out_ids, in_ids=in_ids,
        private_key=bytes(mc_private_key),
    )
    r_values = _r_values_under_key(
        out_ids=out_ids, in_ids=in_ids, private_key=bytes(mc_private_key),
    )
    r_t = torch.from_numpy(r_values).to(Us_pk.device)
    use_mc = r_t > float(accept_threshold)
    Us = torch.where(use_mc, Us_mc.to(Us_pk.device), Us_pk)
    # skipped iff skipped under whichever key was chosen
    skipped = np.where(use_mc.cpu().numpy(), skipped_mc, skipped_pk)
    return Us.clamp(1e-10, 1 - 1e-10), skipped


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
# Metric: log-perplexity (LPPL) under target's processed distribution
# ---------------------------------------------------------------------------


@torch.no_grad()
def lppl_under_target(
    *, target_model, full_ids: torch.LongTensor, prompt_length: int,
    process_logits_kwargs: Optional[dict] = None,
) -> Dict[str, float]:
    """Mean per-token NLL of the realized output under target's *processed*
    distribution (top_k / top_p / temperature applied).

    Cost: 1 target forward over the full prompt+output sequence.  Run OUTSIDE
    the timing window since this would otherwise inflate the measured TR.

    Returns:
        {"LPPL": float, "num_scored_lppl": int}.  LPPL=NaN if there are no
        scored tokens (or every emitted token has -inf log-prob under the
        processed distribution).
    """
    if process_logits_kwargs is None:
        process_logits_kwargs = {}
    n_total = int(full_ids.shape[1])
    n_out = n_total - int(prompt_length)
    if n_out <= 0:
        return {"LPPL": float("nan"), "num_scored_lppl": 0}

    full_ids = full_ids.to(target_model.device)
    out = target_model(full_ids)
    # logits[:, t, :] predicts token at t+1.  We need predictions for output
    # positions [prompt_length, n_total) -> use logits[:, prompt_length-1:n_total-1, :].
    raw = out.logits[:, prompt_length - 1: n_total - 1, :]  # (1, n_out, V)

    # Top_k/top_p warpers operate on (batch, vocab); flatten the time axis and
    # feed dummy input_ids since the warper signature requires them but
    # top_k/top_p/temperature ignore them.
    flat = raw.reshape(-1, raw.shape[-1])  # (n_out, V)
    dummy_ids = torch.zeros(
        (flat.shape[0], 1), dtype=torch.long, device=flat.device
    )
    processed = process_logits(dummy_ids, flat, **process_logits_kwargs)
    logprobs = F.log_softmax(processed.float(), dim=-1)  # (n_out, V)

    emitted = full_ids[0, prompt_length:n_total].to(logprobs.device)
    nll = -logprobs.gather(1, emitted.unsqueeze(-1)).squeeze(-1)  # (n_out,)
    finite = torch.isfinite(nll)
    if not bool(finite.any()):
        return {"LPPL": float("nan"), "num_scored_lppl": 0}
    return {
        "LPPL": float(nll[finite].mean().item()),
        "num_scored_lppl": int(finite.sum().item()),
    }


# ---------------------------------------------------------------------------
# Metric: TPR @ FPR=fpr at first n_tokens (PFR-family decoders only)
# ---------------------------------------------------------------------------


def detector_at_first_n(
    Us: torch.Tensor,
    skipped: Optional[np.ndarray],
    *,
    n_tokens: int = 64,
    fpr: float = 0.01,
    li_delta: float = 0.5,
    pl_eps: float = 0.1,
    variants: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Per-prompt detector decision after the first ``n_tokens`` output tokens.

    Reports TWO test variants per prompt:

    1. **Chernoff bounds** on the existing ANLPPT-{U, Li, PL} statistics.
       Conservative (Chernoff is a p-value upper bound), under-reports TPR.
       Fields: ``log_p_at_{n_tokens}_{v}``, ``det_at_{n_tokens}_{v}``.

    2. **Aaronson Gamma-tail** test (LR-optimal under DG / PFR watermark):
       per-token score s_t = -log(1 - U_t).  Under H0, s_t ~ Exp(1) iid →
       sum ~ Gamma(n, 1).  Exact log p-value via the Erlang series in
       ``unbiased_watermark.scores.pfr_aaronson._log_gamma_tail_p_value``.
       Fields: ``score_aaronson_at_{n_tokens}``, ``log_p_aaronson_at_{n_tokens}``,
       ``det_aaronson_at_{n_tokens}``.

    Both share ``n_added_at_{n_tokens}`` as the effective sample size after
    skipping repeated-context tokens.

    If the realized output has fewer than ``n_tokens`` tokens, the detector
    runs on whatever is available (the row is NOT auto-rejected).
    """
    if variants is None:
        variants = ["U", "Li", "PL"]
    threshold = np.log(float(fpr))
    Us_full = Us.detach().cpu()
    actual_n = int(min(int(n_tokens), Us_full.shape[1]))
    if actual_n <= 0:
        return {f"n_added_at_{n_tokens}": 0}
    Us_n = Us_full[:, :actual_n]
    skipped_n = (
        np.asarray(skipped, dtype=bool).reshape(1, -1)[:, :actual_n]
        if skipped is not None else None
    )
    inner = anlppt_metrics(
        Us_n, li_delta=li_delta, pl_eps=pl_eps,
        skipped=skipped_n, variants=variants,
    )
    n_added = int(inner.get("n_added", 0))
    out: Dict[str, float] = {f"n_added_at_{n_tokens}": n_added}

    # ---- (1) Chernoff bound on ANLPPT-{U, Li, PL} (existing) ----
    if n_added <= 0:
        for v in variants:
            out[f"log_p_at_{n_tokens}_{v}"] = float("nan")
            out[f"det_at_{n_tokens}_{v}"] = 0
    else:
        for v in variants:
            anlppt_key = f"ANLPPT_{v}"
            if anlppt_key not in inner:
                continue
            log_p = -float(inner[anlppt_key]) * n_added
            out[f"log_p_at_{n_tokens}_{v}"] = log_p
            out[f"det_at_{n_tokens}_{v}"] = int(log_p <= threshold)

    # ---- (2) Aaronson exact Gamma-tail (LR-optimal) ----
    from unbiased_watermark.scores.pfr_aaronson import _log_gamma_tail_p_value
    Us_arr = Us_n.numpy().reshape(-1)
    if skipped_n is not None:
        valid_mask = ~skipped_n.reshape(-1)
        Us_arr = Us_arr[valid_mask]
    n_eff = int(Us_arr.size)
    if n_eff <= 0:
        out[f"score_aaronson_at_{n_tokens}"] = float("nan")
        out[f"log_p_aaronson_at_{n_tokens}"] = float("nan")
        out[f"det_aaronson_at_{n_tokens}"] = 0
    else:
        # s_t = -log(1 - U_t); use log1p for numerical stability when U near 0.
        # Clip away from 1 to avoid -log(0) = +inf when U exactly == 1
        # (recovered uniforms are clipped to [1e-10, 1 - 1e-10] upstream).
        u_clip = np.clip(Us_arr, 0.0, 1.0 - 1e-10)
        score_aaronson = float(np.sum(-np.log1p(-u_clip)))
        log_p_aaronson = _log_gamma_tail_p_value(n_eff, score_aaronson)
        out[f"score_aaronson_at_{n_tokens}"] = score_aaronson
        out[f"log_p_aaronson_at_{n_tokens}"] = float(log_p_aaronson)
        out[f"det_aaronson_at_{n_tokens}"] = int(log_p_aaronson <= threshold)
    return out


# ---------------------------------------------------------------------------
# Metric: KL/WS ratio (Definition 3.1, delta-conditional empirical estimator)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _logits_per_output_step(model, full_ids, prompt_length, plk):
    """Run model on full_ids, return per-output-position processed logits."""
    if full_ids.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    seq_len = int(full_ids.shape[1])
    if not 0 < prompt_length <= seq_len:
        raise ValueError("invalid prompt_length")
    if prompt_length == seq_len:
        return torch.empty(
            (1, 0, int(model.config.vocab_size)),
            device=full_ids.device, dtype=torch.float32,
        )
    out = model(full_ids)
    raw = out.logits[:, prompt_length - 1:-1, :]
    if not plk:
        return raw
    proc = []
    for offset in range(raw.shape[1]):
        pos = prompt_length + offset
        proc.append(process_logits(full_ids[:, :pos], raw[:, offset, :], **plk))
    return torch.stack(proc, dim=1)


def _watermark_code_from_contexts(reweight, cc_extractor, cch, private_key,
                                  out_ids, vocab_size, in_ids=None):
    """Per-step watermark codes for output tokens, with skipped flags."""
    ids = out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)
    cc_s, sk_s = [], []
    start = ids.shape[-1] - out_ids.shape[-1]
    for i in range(start, ids.shape[-1]):
        step = cch.step(cc_extractor, ids[..., :i])
        if isinstance(step, tuple) and len(step) >= 2:
            cc, sk = step[0], step[1]
        else:
            cc, sk = step, np.zeros(out_ids.shape[:-1], dtype=bool)
        cc_s.append(cc)
        sk_s.append(sk)
    cc_arr = np.stack(cc_s, axis=-1)
    sk_arr = np.stack(sk_s, axis=-1)
    rng = np.empty(cc_arr.shape, dtype=object)
    pk_b = bytes(private_key)
    for index in np.ndindex(rng.shape):
        rng[index] = _get_rng_local(cc_arr[index], pk_b)
    code = reweight.watermark_code_type.from_random(rng, vocab_size)
    code = code.tensor_shape_map(lambda x: x.to(out_ids.device))
    return code, sk_arr


@torch.no_grad()
def kl_ws_ratio_mc_pseudo_r(
    *, target_model, draft_model, full_ids, prompt_length,
    private_key, mc_private_key,
    process_logits_kwargs: Optional[dict],
) -> Dict[str, float]:
    """Algorithm-1 (arXiv:2602.01428) effective-distribution KL.

    The fixed-zeta one-step effective distribution at each output position
    is::

        effective = qz * indicator(r <= alpha) + (1 - accept_mass) * residual_zeta

    where:
      - qz       = zeta_T-reweight(Q) under DeltaGumbel + PrevN(3)
      - alpha    = min(1, P/Q)
      - r        = pseudorandom acceptance variable G(zeta_R)
      - residual_zeta = zeta_R-reweight((P-Q)+ normalised)
    KL(effective || P) is computed and aggregated like the basic DG case.

    With ``psedo_r=True`` ALL randomness is pinned to zeta_T + zeta_R, so
    the effective distribution should approach P-watermark-strength-bound
    (KL/H ratio close to 1).  Mirrors improving_KL
    compute_mc_uwm_speed_kl_from_sequence(pseudo_r_private_key, residual_private_key).
    """
    seq_len = int(full_ids.shape[1])
    if not 0 < prompt_length < seq_len:
        return {"kl_ws_ratio": float("nan"), "kl_ws_sum": 0.0,
                "kl_h_sum": 0.0, "kl_num_scored": 0,
                "kl_masked_ratio": 0.0}
    out_ids = full_ids[:, prompt_length:]
    in_ids = full_ids[:, :prompt_length]
    plk = process_logits_kwargs or {}

    # Target P logits and draft Q logits at each output step
    p_logits = _logits_per_output_step(target_model, full_ids, prompt_length, plk)
    q_logits = _logits_per_output_step(draft_model, full_ids, prompt_length, plk)
    V = min(int(p_logits.shape[-1]), int(q_logits.shape[-1]))
    p_logits = p_logits[..., :V]
    q_logits = q_logits[..., :V]

    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch_t = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    code_t, skipped = _watermark_code_from_contexts(
        reweight, cc_extractor, cch_t, bytes(private_key),
        out_ids.to(target_model.device), V, in_ids=in_ids.to(target_model.device),
    )
    # qz = zeta_T-reweight applied to Q
    qz_logits = reweight.reweight_logits(code_t, q_logits.float())
    qz_probs = F.softmax(qz_logits.float(), dim=-1)

    p_probs = F.softmax(p_logits.float(), dim=-1)
    q_probs = F.softmax(q_logits.float(), dim=-1)
    eps = 1e-20
    alpha = torch.minimum(torch.ones_like(p_probs),
                          p_probs / q_probs.clamp_min(eps))

    # r values from zeta_R
    r_values = _r_values_under_key(
        out_ids=out_ids.to(target_model.device),
        in_ids=in_ids.to(target_model.device),
        private_key=bytes(mc_private_key),
    )
    r = torch.from_numpy(np.asarray(r_values, dtype=np.float32)).to(p_probs.device)
    indicator = (r.unsqueeze(-1) <= alpha).float()
    accept_mass = (qz_probs * indicator).sum(dim=-1, keepdim=True)

    # residual_zeta = zeta_R reweight of (P-Q)+
    cch_r = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    code_r, _ = _watermark_code_from_contexts(
        reweight, cc_extractor, cch_r, bytes(mc_private_key),
        out_ids.to(target_model.device), V, in_ids=in_ids.to(target_model.device),
    )
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(eps)
    res_logits = reweight.reweight_logits(code_r, residual.clamp_min(eps).log())
    residual_zeta = F.softmax(res_logits.float(), dim=-1)

    effective = qz_probs * indicator + (1.0 - accept_mass) * residual_zeta
    effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(eps)

    # discrete KL(effective || p_probs)
    e = effective.clamp_min(eps)
    p = p_probs.clamp_min(eps)
    per_token_kl = (e * (e.log() - p.log())).sum(dim=-1)
    per_token_h = -torch.where(p > 0, p * p.log(), torch.zeros_like(p)).sum(dim=-1)

    skipped_t = torch.tensor(np.asarray(skipped, dtype=bool),
                             device=per_token_kl.device, dtype=torch.bool)
    if skipped_t.shape != per_token_kl.shape:
        skipped_t = skipped_t.reshape(per_token_kl.shape)
    mask = ~skipped_t
    count = int(mask.sum().item())
    kl_sum = float(per_token_kl[mask].sum().item()) if count else 0.0
    h_sum = float(per_token_h[mask].sum().item()) if count else 0.0
    return {
        "kl_ws_ratio": float(kl_sum / max(h_sum, eps)),
        "kl_ws_mean": kl_sum / max(count, 1),
        "kl_h_mean": h_sum / max(count, 1),
        "kl_ws_sum": kl_sum,
        "kl_h_sum": h_sum,
        "kl_num_scored": count,
        "kl_masked_ratio": float(skipped_t.float().mean().item())
        if skipped_t.numel() else 0.0,
    }


@torch.no_grad()
def kl_ws_ratio_mc_speed(
    *, target_model, draft_model, full_ids, prompt_length, private_key,
    process_logits_kwargs: Optional[dict],
) -> Dict[str, float]:
    """mc_uwm_speed effective-distribution KL (no pseudo_r, no residual_zeta).

    effective = qz * alpha + (1 - accept_mass) * residual_uniform_over_(P-Q)+

    Mirrors improving_KL compute_mc_uwm_speed_kl_from_sequence with
    pseudo_r_private_key=None.
    """
    seq_len = int(full_ids.shape[1])
    if not 0 < prompt_length < seq_len:
        return {"kl_ws_ratio": float("nan"), "kl_ws_sum": 0.0,
                "kl_h_sum": 0.0, "kl_num_scored": 0,
                "kl_masked_ratio": 0.0}
    out_ids = full_ids[:, prompt_length:]
    in_ids = full_ids[:, :prompt_length]
    plk = process_logits_kwargs or {}
    p_logits = _logits_per_output_step(target_model, full_ids, prompt_length, plk)
    q_logits = _logits_per_output_step(draft_model, full_ids, prompt_length, plk)
    V = min(int(p_logits.shape[-1]), int(q_logits.shape[-1]))
    p_logits = p_logits[..., :V]
    q_logits = q_logits[..., :V]

    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch_t = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    code_t, skipped = _watermark_code_from_contexts(
        reweight, cc_extractor, cch_t, bytes(private_key),
        out_ids.to(target_model.device), V, in_ids=in_ids.to(target_model.device),
    )
    qz_logits = reweight.reweight_logits(code_t, q_logits.float())
    qz_probs = F.softmax(qz_logits.float(), dim=-1)

    p_probs = F.softmax(p_logits.float(), dim=-1)
    q_probs = F.softmax(q_logits.float(), dim=-1)
    eps = 1e-20
    alpha = torch.minimum(torch.ones_like(p_probs),
                          p_probs / q_probs.clamp_min(eps))
    accept_mass = (qz_probs * alpha).sum(dim=-1, keepdim=True)
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(eps)
    effective = qz_probs * alpha + (1.0 - accept_mass) * residual
    effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(eps)

    e = effective.clamp_min(eps)
    p = p_probs.clamp_min(eps)
    per_token_kl = (e * (e.log() - p.log())).sum(dim=-1)
    per_token_h = -torch.where(p > 0, p * p.log(), torch.zeros_like(p)).sum(dim=-1)
    skipped_t = torch.tensor(np.asarray(skipped, dtype=bool),
                             device=per_token_kl.device, dtype=torch.bool)
    if skipped_t.shape != per_token_kl.shape:
        skipped_t = skipped_t.reshape(per_token_kl.shape)
    mask = ~skipped_t
    count = int(mask.sum().item())
    kl_sum = float(per_token_kl[mask].sum().item()) if count else 0.0
    h_sum = float(per_token_h[mask].sum().item()) if count else 0.0
    return {
        "kl_ws_ratio": float(kl_sum / max(h_sum, eps)),
        "kl_ws_mean": kl_sum / max(count, 1),
        "kl_h_mean": h_sum / max(count, 1),
        "kl_ws_sum": kl_sum,
        "kl_h_sum": h_sum,
        "kl_num_scored": count,
        "kl_masked_ratio": float(skipped_t.float().mean().item())
        if skipped_t.numel() else 0.0,
    }


@torch.no_grad()
def kl_ws_ratio_dg(
    *, target_model, full_ids, prompt_length, private_key,
    process_logits_kwargs: Optional[dict],
) -> Dict[str, float]:
    """Per-token KL(S(P,zeta) || P) summed over output for DG-family
    watermarks (basic_uwm, mc_uwm_*).

    Computes one target forward over (prompt + output), extracts p_logits
    at each output position, applies process_logits, then runs detect_pre
    with DeltaGumbel reweight + PrevN(3) extractor under ``private_key``
    to recover per-step q_logits.  Returns kl_sum / entropy_sum (==
    ws/H ratio used in improving_KL).

    Mirrors improving_KL.unbiased_watermark.scores.kl_watermark_strength.
    compute_basic_uwm_kl_from_sequence (target-side full-vocab variant).
    """
    if full_ids.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    seq_len = int(full_ids.shape[1])
    if not 0 < prompt_length < seq_len:
        return {"kl_ws_ratio": float("nan"), "kl_ws_sum": 0.0,
                "kl_h_sum": 0.0, "kl_num_scored": 0,
                "kl_masked_ratio": 0.0}
    out_len = seq_len - prompt_length
    out_ids = full_ids[:, prompt_length:]
    in_ids = full_ids[:, :prompt_length]
    # Target forward over the full sequence; p_logits[t] is the target's
    # logits AT position prompt_length+t (predicting the t-th output).
    output = target_model(full_ids)
    raw_logits = output.logits[:, prompt_length - 1:-1, :]  # (1, out_len, V)
    plk = process_logits_kwargs or {}
    if plk:
        proc = []
        for offset in range(raw_logits.shape[1]):
            pos = prompt_length + offset
            proc.append(process_logits(
                full_ids[:, :pos], raw_logits[:, offset, :], **plk,
            ))
        p_logits = torch.stack(proc, dim=1)
    else:
        p_logits = raw_logits
    reweight = uwm.DeltaGumbel_Reweight()
    cc_extractor = uwm.lm.PrevN_ContextCodeExtractor(n=3)
    cch = uwm.lm.ContextCodeHistory(batch_shape=(1,))
    _wm_logits, q_logits, _cc, _code, skipped = uwm.lm.detect_pre(
        int(target_model.config.vocab_size), reweight, cc_extractor, cch,
        bytes(private_key),
        out_ids.to(target_model.device),
        in_ids=in_ids.to(target_model.device),
        p_logits=p_logits,
    )
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    q = log_q.exp()
    per_token_kl = torch.where(q > 0, q * (log_q - log_p),
                               torch.zeros_like(q)).sum(dim=-1)
    p = log_p.exp()
    # Guard 0 * (-inf) = NaN at top-k-masked positions: zero those terms.
    per_token_h = -torch.where(p > 0, p * log_p,
                               torch.zeros_like(p)).sum(dim=-1)
    skipped_t = torch.tensor(np.asarray(skipped, dtype=bool),
                             device=per_token_kl.device, dtype=torch.bool)
    if skipped_t.shape != per_token_kl.shape:
        skipped_t = skipped_t.reshape(per_token_kl.shape)
    mask = ~skipped_t
    count = int(mask.sum().item())
    kl_sum = float(per_token_kl[mask].sum().item()) if count else 0.0
    h_sum = float(per_token_h[mask].sum().item()) if count else 0.0
    ratio = kl_sum / max(h_sum, 1e-20)
    return {
        "kl_ws_ratio": float(ratio),
        "kl_ws_mean": kl_sum / max(count, 1),
        "kl_h_mean": h_sum / max(count, 1),
        "kl_ws_sum": kl_sum,
        "kl_h_sum": h_sum,
        "kl_num_scored": count,
        "kl_masked_ratio": float(skipped_t.float().mean().item())
        if skipped_t.numel() else 0.0,
    }


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
