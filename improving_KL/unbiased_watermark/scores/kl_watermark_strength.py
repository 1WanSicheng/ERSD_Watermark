from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from ..base import AbstractReweight
from ..lm import (
    AbstractContextCodeExtractor,
    ContextCodeHistory,
    detect_pre,
    get_mse_target_logprob,
    get_r_values,
    get_rng,
    step_watermark_synthid,
)


def discrete_kl_from_logits(
    q_logits: FloatTensor,
    p_logits: FloatTensor,
    skipped: np.ndarray | torch.Tensor | None = None,
    eps: float = 1e-20,
) -> dict[str, Any]:
    """Compute tokenwise discrete KL(q || p) from logits on the same support."""
    if q_logits.shape != p_logits.shape:
        raise ValueError(
            f"q_logits and p_logits must have the same shape, got "
            f"{tuple(q_logits.shape)} and {tuple(p_logits.shape)}"
        )
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    q = log_q.exp()
    terms = torch.where(q > 0, q * (log_q - log_p), torch.zeros_like(q))
    per_token_kl = terms.sum(dim=-1)
    p = log_p.exp()
    per_token_entropy = -(p * log_p).sum(dim=-1)
    return summarize_tokenwise_kl(
        per_token_kl,
        skipped=skipped,
        per_token_entropy=per_token_entropy,
        eps=eps,
    )


def discrete_kl_from_probs(
    q_probs: FloatTensor,
    p_probs: FloatTensor,
    skipped: np.ndarray | torch.Tensor | None = None,
    eps: float = 1e-20,
) -> dict[str, Any]:
    """Compute tokenwise discrete KL(q || p) from normalized probabilities."""
    if q_probs.shape != p_probs.shape:
        raise ValueError(
            f"q_probs and p_probs must have the same shape, got "
            f"{tuple(q_probs.shape)} and {tuple(p_probs.shape)}"
        )
    q = q_probs.float().clamp_min(eps)
    p = p_probs.float().clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    per_token_kl = (q * (q.log() - p.log())).sum(dim=-1)
    per_token_entropy = -(p * p.log()).sum(dim=-1)
    return summarize_tokenwise_kl(
        per_token_kl,
        skipped=skipped,
        per_token_entropy=per_token_entropy,
        eps=eps,
    )


def summarize_tokenwise_kl(
    per_token_kl: FloatTensor,
    skipped: np.ndarray | torch.Tensor | None = None,
    per_token_entropy: FloatTensor | None = None,
    eps: float = 1e-20,
) -> dict[str, Any]:
    """Summarize tokenwise KL values with an optional skip mask."""
    if skipped is None:
        skipped_t = torch.zeros_like(per_token_kl, dtype=torch.bool)
    elif isinstance(skipped, torch.Tensor):
        skipped_t = skipped.to(device=per_token_kl.device, dtype=torch.bool)
    else:
        skipped_t = torch.tensor(skipped, device=per_token_kl.device, dtype=torch.bool)
    if skipped_t.shape != per_token_kl.shape:
        raise ValueError(
            f"skipped mask shape {tuple(skipped_t.shape)} does not match "
            f"KL shape {tuple(per_token_kl.shape)}"
        )
    mask = ~skipped_t
    count = int(mask.sum().item())
    kl_sum = float(per_token_kl[mask].sum().item()) if count else 0.0
    entropy_sum = 0.0
    entropy_mean = 0.0
    ratio = 0.0
    if per_token_entropy is not None:
        if per_token_entropy.shape != per_token_kl.shape:
            raise ValueError(
                f"entropy shape {tuple(per_token_entropy.shape)} does not match "
                f"KL shape {tuple(per_token_kl.shape)}"
            )
        entropy_sum = float(per_token_entropy[mask].sum().item()) if count else 0.0
        entropy_mean = entropy_sum / max(count, 1)
        ratio = kl_sum / max(entropy_sum, eps)
    return {
        "per_token_kl": per_token_kl.detach().cpu().numpy(),
        "per_token_entropy": per_token_entropy.detach().cpu().numpy()
        if per_token_entropy is not None
        else None,
        "KL_WS_sum": kl_sum,
        "KL_WS_mean": kl_sum / max(count, 1),
        "KL_WS_entropy_sum": entropy_sum,
        "KL_WS_entropy_mean": entropy_mean,
        "KL_WS_ratio": ratio,
        "KL_WS_count": count,
        "KL_WS_skipped_ratio": float(skipped_t.float().mean().item())
        if skipped_t.numel()
        else 0.0,
    }


@torch.no_grad()
def next_token_logits_from_full_sequence(
    model,
    full_ids: LongTensor,
    prompt_length: int,
    process_logits_kwargs: dict | None = None,
) -> FloatTensor:
    """Return logits for positions prompt_length..len(full_ids)-1."""
    if full_ids.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    if not 0 < prompt_length <= full_ids.shape[1]:
        raise ValueError("prompt_length must be in [1, sequence length]")
    if prompt_length == full_ids.shape[1]:
        return torch.empty(
            (1, 0, model.config.vocab_size),
            device=full_ids.device,
            dtype=torch.float32,
        )
    process_logits_kwargs = process_logits_kwargs or {}
    outputs = model(full_ids)
    logits = outputs.logits[:, prompt_length - 1 : -1, :]
    if process_logits_kwargs:
        from accuwm.utils import process_logits

        processed = []
        for offset in range(logits.shape[1]):
            pos = prompt_length + offset
            processed.append(
                process_logits(
                    full_ids[:, :pos],
                    logits[:, offset, :],
                    **process_logits_kwargs,
                )
            )
        logits = torch.stack(processed, dim=1)
    return logits


def _ids_for_output_contexts(out_ids: LongTensor, in_ids: LongTensor | None) -> LongTensor:
    return out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)


def watermark_code_from_contexts(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    vocab_size: int,
    in_ids: LongTensor | None = None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Rebuild watermark codes for final output contexts without logits."""
    batch_shape = out_ids.shape[:-1]
    if cch.data.shape != batch_shape:
        raise ValueError("ContextCodeHistory shape must match output batch shape")
    ids = _ids_for_output_contexts(out_ids, in_ids)
    cc_s, skipped_s = [], []
    start = ids.shape[-1] - out_ids.shape[-1]
    for i in range(start, ids.shape[-1]):
        cc, skipped = cch.step(cc_extractor, ids[..., :i])
        cc_s.append(cc)
        skipped_s.append(skipped)
    cc = np.stack(cc_s, axis=-1)
    skipped = np.stack(skipped_s, axis=-1)
    rng = np.empty(cc.shape, dtype=object)
    for index in np.ndindex(rng.shape):
        rng[index] = get_rng(cc[index], private_key)
    code = reweight.watermark_code_type.from_random(rng, vocab_size)
    code = code.tensor_shape_map(lambda x: x.to(out_ids.device))
    return code, skipped, cc


def reweight_probs(
    reweight: AbstractReweight,
    code: Any,
    probs: FloatTensor,
    eps: float = 1e-20,
) -> FloatTensor:
    """Apply a UWM reweighting operation to a probability distribution."""
    probs = probs.float()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    try:
        logits = reweight.reweight_logits(code, probs, input_is_probs=True)
    except TypeError:
        logits = reweight.reweight_logits(code, probs.clamp_min(eps).log())
    return F.softmax(logits.float(), dim=-1)


def reweight_logits_to_probs(
    reweight: AbstractReweight,
    code: Any,
    logits: FloatTensor,
    temperature: float = 1.0,
) -> FloatTensor:
    """Apply reweighting to logits and return normalized probabilities."""
    q_logits = reweight.reweight_logits(code, logits.float() / temperature)
    return F.softmax(q_logits.float(), dim=-1)


def compute_basic_uwm_kl_from_sequence(
    p_logits: FloatTensor,
    out_ids: LongTensor,
    in_ids: LongTensor,
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    private_key: bytes,
) -> dict[str, Any]:
    """Compute target-side KL(S(P,zeta) || P) for Basic UWM style methods."""
    cch = ContextCodeHistory(batch_shape=out_ids.shape[:-1])
    wm_logits, q_logits, _cc, _code, skipped = detect_pre(
        vocab_size=p_logits.shape[-1],
        reweight=reweight,
        cc_extractor=cc_extractor,
        cch=cch,
        private_key=private_key,
        out_ids=out_ids,
        in_ids=in_ids,
        p_logits=p_logits,
    )
    del wm_logits
    result = discrete_kl_from_logits(q_logits, p_logits, skipped=skipped)
    result["KL_WS_kind"] = "target_side_full_vocab"
    return result


def mc_speed_effective_probs(
    p_probs: FloatTensor,
    q_probs: FloatTensor,
    qz_probs: FloatTensor,
    eps: float = 1e-20,
) -> FloatTensor:
    """Alpha-averaged one-step effective distribution for mc_uwm_speed."""
    p = p_probs.float() / p_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q_probs.float() / q_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    qz = qz_probs.float() / qz_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = torch.minimum(torch.ones_like(p), p / q.clamp_min(eps))
    accept_mass = (qz * alpha).sum(dim=-1, keepdim=True)
    residual = torch.clamp(p - q, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(eps)
    effective = qz * alpha + (1.0 - accept_mass) * residual
    return effective / effective.sum(dim=-1, keepdim=True).clamp_min(eps)


def mc_pseudo_r_effective_probs(
    p_probs: FloatTensor,
    q_probs: FloatTensor,
    qz_probs: FloatTensor,
    r_values: FloatTensor,
    residual_zeta_probs: FloatTensor | None = None,
    eps: float = 1e-20,
) -> FloatTensor:
    """Fixed-zeta one-step effective distribution for pseudo-r variants."""
    p = p_probs.float() / p_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q_probs.float() / q_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    qz = qz_probs.float() / qz_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    alpha = torch.minimum(torch.ones_like(p), p / q.clamp_min(eps))
    indicator = (r_values.to(p.device).float().unsqueeze(-1) <= alpha).float()
    accept_mass = (qz * indicator).sum(dim=-1, keepdim=True)
    if residual_zeta_probs is None:
        residual = torch.clamp(p - q, min=0.0)
        residual_zeta = residual / residual.sum(dim=-1, keepdim=True).clamp_min(eps)
    else:
        residual_zeta = residual_zeta_probs.float()
        residual_zeta = residual_zeta / residual_zeta.sum(dim=-1, keepdim=True).clamp_min(eps)
    effective = qz * indicator + (1.0 - accept_mass) * residual_zeta
    return effective / effective.sum(dim=-1, keepdim=True).clamp_min(eps)


def compute_mc_uwm_speed_kl_from_sequence(
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    out_ids: LongTensor,
    in_ids: LongTensor,
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    private_key: bytes,
    pseudo_r_private_key: bytes | None = None,
    residual_private_key: bytes | None = None,
    watermark_temperature: float = 1.0,
    baseline_temperature: float = 1.0,
) -> dict[str, Any]:
    """Compute one-step KL for mc_uwm_speed or fixed-zeta pseudo-r variants."""
    vocab_size = min(p_logits.shape[-1], q_logits.shape[-1])
    p_logits = p_logits[..., :vocab_size]
    q_logits = q_logits[..., :vocab_size]
    p_baseline_logits = p_logits.float() / baseline_temperature
    code, skipped, _cc = watermark_code_from_contexts(
        reweight,
        cc_extractor,
        ContextCodeHistory(batch_shape=out_ids.shape[:-1]),
        private_key,
        out_ids,
        vocab_size,
        in_ids=in_ids,
    )
    p_accept_probs = F.softmax(p_logits.float(), dim=-1)
    q_accept_probs = F.softmax(q_logits.float(), dim=-1)
    p_baseline_probs = F.softmax(p_baseline_logits, dim=-1)
    qz_probs = reweight_logits_to_probs(
        reweight, code, q_logits, temperature=watermark_temperature
    )
    if pseudo_r_private_key is None:
        effective = mc_speed_effective_probs(p_accept_probs, q_accept_probs, qz_probs)
        kind = "mc_uwm_speed_alpha_avg_one_step"
    else:
        r_values = get_r_values(
            cc_extractor,
            ContextCodeHistory(batch_shape=out_ids.shape[:-1]),
            pseudo_r_private_key,
            out_ids,
            in_ids=in_ids,
        )
        residual_zeta = None
        if residual_private_key is not None:
            residual = torch.clamp(p_accept_probs - q_accept_probs, min=0.0)
            residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            residual_code, _res_skipped, _ = watermark_code_from_contexts(
                reweight,
                cc_extractor,
                ContextCodeHistory(batch_shape=out_ids.shape[:-1]),
                residual_private_key,
                out_ids,
                vocab_size,
                in_ids=in_ids,
            )
            residual_zeta = reweight_logits_to_probs(
                reweight,
                residual_code,
                residual.clamp_min(1e-20).log(),
                temperature=watermark_temperature,
            )
        effective = mc_pseudo_r_effective_probs(
            p_accept_probs,
            q_accept_probs,
            qz_probs,
            torch.tensor(r_values, device=p_logits.device),
            residual_zeta_probs=residual_zeta,
        )
        kind = "mc_uwm_pseudo_r_fixed_zeta_one_step"
    result = discrete_kl_from_probs(effective, p_baseline_probs, skipped=skipped)
    result["KL_WS_kind"] = kind
    return result


def synthid_topk_kl_for_contexts(
    p_logits: FloatTensor,
    full_ids: LongTensor,
    prompt_length: int,
    reweight,
    cc_extractor: AbstractContextCodeExtractor,
    temperature: float,
    top_k: int,
    apply_top_k: bool = True,
) -> dict[str, Any]:
    """Compute SynthID KL(S(P,zeta) || P) with dense top-k watermark and full P baseline."""
    if p_logits.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    cch = ContextCodeHistory(batch_shape=(1,))
    q_probs = []
    p_probs = []
    skipped_s = []
    for offset in range(p_logits.shape[1]):
        pos = prompt_length + offset
        context = full_ids[:, :pos]
        p_step = p_logits[:, offset, :]
        wm_logits, _q_logits, _cc, _g, skipped, indices = step_watermark_synthid(
            reweight,
            p_step,
            context,
            cc_extractor,
            cch,
            temperature,
            top_k,
            apply_top_k=apply_top_k,
        )
        p_probs.append(F.softmax(p_step.float() / temperature, dim=-1))
        if apply_top_k:
            dense_q = torch.zeros_like(p_step, dtype=torch.float32)
            dense_q.scatter_(1, indices, F.softmax(wm_logits.float(), dim=-1))
        else:
            dense_q = F.softmax(wm_logits.float(), dim=-1)
        q_probs.append(dense_q)
        skipped_s.append(skipped)
    if not q_probs:
        return summarize_tokenwise_kl(torch.empty((1, 0), device=p_logits.device))
    skipped = np.stack(skipped_s, axis=-1)
    result = discrete_kl_from_probs(
        torch.stack(q_probs, dim=1),
        torch.stack(p_probs, dim=1),
        skipped=skipped,
    )
    result["KL_WS_kind"] = "synthid_dense_topk_full_p"
    return result


def synthid_context_distributions(
    p_logits: FloatTensor,
    full_ids: LongTensor,
    prompt_length: int,
    reweight,
    cc_extractor: AbstractContextCodeExtractor,
    temperature: float,
    top_k: int,
) -> tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, np.ndarray]:
    """Return dense raw, top-k, and SynthID top-k distributions for contexts."""
    if p_logits.shape[0] != 1:
        raise ValueError("only batch_size=1 is supported")
    cch = ContextCodeHistory(batch_shape=(1,))
    raw_probs = []
    topk_probs = []
    wm_probs = []
    wm_logprobs = []
    skipped_s = []
    for offset in range(p_logits.shape[1]):
        pos = prompt_length + offset
        context = full_ids[:, :pos]
        p_step = p_logits[:, offset, :]
        wm_logits, q_logits, _cc, _g, skipped, indices = step_watermark_synthid(
            reweight,
            p_step,
            context,
            cc_extractor,
            cch,
            temperature,
            top_k,
            apply_top_k=True,
        )
        del q_logits
        raw_probs.append(F.softmax(p_step.float(), dim=-1))
        topk = torch.zeros_like(p_step, dtype=torch.float32)
        topk_scores = torch.topk(p_step.float() / temperature, k=top_k, dim=-1)
        topk.scatter_(1, topk_scores.indices, F.softmax(topk_scores.values, dim=-1))
        topk_probs.append(topk)
        wm_topk = F.softmax(wm_logits.float(), dim=-1)
        dense_wm = torch.zeros_like(p_step, dtype=torch.float32)
        dense_wm.scatter_(1, indices, wm_topk)
        wm_probs.append(dense_wm)
        dense_wm_log = torch.full_like(p_step, -1e12, dtype=torch.float32)
        dense_wm_log.scatter_(1, indices, F.log_softmax(wm_logits.float(), dim=-1))
        wm_logprobs.append(dense_wm_log)
        skipped_s.append(skipped)
    if not raw_probs:
        empty = torch.empty((1, 0, p_logits.shape[-1]), device=p_logits.device)
        return empty, empty, empty, empty, np.empty((1, 0), dtype=bool)
    return (
        torch.stack(raw_probs, dim=1),
        torch.stack(topk_probs, dim=1),
        torch.stack(wm_probs, dim=1),
        torch.stack(wm_logprobs, dim=1),
        np.stack(skipped_s, axis=-1),
    )


def synthid_mc_effective_probs(
    method: str,
    p_raw_probs: FloatTensor,
    q_raw_probs: FloatTensor,
    p_wm_probs: FloatTensor,
    q_wm_probs: FloatTensor,
    p_wm_logprobs: FloatTensor,
    q_wm_logprobs: FloatTensor,
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    full_ids: LongTensor,
    prompt_length: int,
    residual_reweight,
    cc_extractor: AbstractContextCodeExtractor,
    temperature: float,
    top_k: int,
    eps: float = 1e-20,
) -> FloatTensor:
    """One-step alpha-averaged effective distribution for SynthID MC variants."""
    p = p_raw_probs.float() / p_raw_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q_raw_probs.float() / q_raw_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    if method == "mc_mws":
        proposal = q_wm_probs.float()
        target = p_wm_probs.float()
        proposal_accept = proposal
        target_accept = target
        residual_zeta = None
    else:
        proposal = q_wm_probs.float()
        target = p
        proposal_accept = q
        target_accept = p
        residual_zeta = None
    alpha = torch.minimum(torch.ones_like(target_accept), target_accept / proposal_accept.clamp_min(eps))
    accept_mass = (proposal * alpha).sum(dim=-1, keepdim=True)
    residual = torch.clamp(target_accept - proposal_accept, min=0.0)
    residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(eps)
    if method == "mc_2keys":
        residual_logits = residual.clamp_min(eps).log()
        _raw, _topk, residual_wm, _wm_log, _skipped = synthid_context_distributions(
            residual_logits,
            full_ids,
            prompt_length,
            residual_reweight,
            cc_extractor,
            temperature,
            top_k,
        )
        residual_zeta = residual_wm
    elif method == "mc_mse":
        residual_zeta = residual
    elif method == "mc_mws":
        residual_zeta = residual
    else:
        raise ValueError(f"unknown SynthID MC method: {method}")
    effective = proposal * alpha + (1.0 - accept_mass) * residual_zeta
    return effective / effective.sum(dim=-1, keepdim=True).clamp_min(eps)


def synthid_mc_kl_for_contexts(
    method: str,
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    full_ids: LongTensor,
    prompt_length: int,
    reweight,
    residual_reweight,
    cc_extractor: AbstractContextCodeExtractor,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    """Compute one-step KL for SynthID MC variants against target raw P."""
    vocab_size = min(p_logits.shape[-1], q_logits.shape[-1])
    p_logits = p_logits[..., :vocab_size]
    q_logits = q_logits[..., :vocab_size]
    p_raw, _p_topk, p_wm, p_wm_log, skipped = synthid_context_distributions(
        p_logits,
        full_ids,
        prompt_length,
        reweight,
        cc_extractor,
        temperature,
        top_k,
    )
    q_raw, _q_topk, q_wm, q_wm_log, _q_skipped = synthid_context_distributions(
        q_logits,
        full_ids.to(q_logits.device),
        prompt_length,
        reweight,
        cc_extractor,
        temperature,
        top_k,
    )
    q_raw = q_raw.to(p_logits.device)
    q_wm = q_wm.to(p_logits.device)
    q_wm_log = q_wm_log.to(p_logits.device)
    if method == "mc_mse":
        mse_log = get_mse_target_logprob(q_wm_log, q_logits.float(), p_logits.float())
        effective = F.softmax(mse_log.float(), dim=-1)
    else:
        effective = synthid_mc_effective_probs(
            method,
            p_raw,
            q_raw,
            p_wm,
            q_wm,
            p_wm_log,
            q_wm_log,
            p_logits,
            q_logits,
            full_ids,
            prompt_length,
            residual_reweight,
            cc_extractor,
            temperature,
            top_k,
        )
    result = discrete_kl_from_probs(effective, p_raw, skipped=skipped)
    result["KL_WS_kind"] = f"{method}_synthid_mc_one_step"
    return result
