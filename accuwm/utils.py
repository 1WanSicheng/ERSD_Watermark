import numpy as np
import os
import time
import torch
from torch import FloatTensor, LongTensor
from torch.utils._pytree import tree_map
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

import unbiased_watermark as uwm
from unbiased_watermark import (
    AbstractWatermarkCode,
    AbstractReweight,
    AbstractContextCodeExtractor,
    ContextCodeHistory,
    step_watermark,
)


def process_logits(input_ids, logits, logits_processor=None, logits_warper=None):
    """
    logits_processor: TODO
    logits_warper: TODO
    """
    if logits_processor is not None:
        logits = logits_processor(input_ids, logits)
    if logits_warper is not None:
        logits = logits_warper(input_ids, logits)
    return logits


def cache_is_legacy(past_key_values):
    return isinstance(past_key_values, (list, tuple))


def cache_is_dynamic(past_key_values):
    return hasattr(past_key_values, "key_cache")


def cache_len(past_key_values):
    if past_key_values is None:
        return 0
    if cache_is_legacy(past_key_values):
        return past_key_values[0][0].shape[2]
    if cache_is_dynamic(past_key_values):
        return past_key_values.key_cache[0].shape[2]
    return 0


def create_empty_cache():
    return DynamicCache()


def truncate_cache(past_key_values, new_len):
    if past_key_values is None:
        return None
    if cache_is_legacy(past_key_values):
        return tree_map(lambda x: x[:, :, :new_len, :], past_key_values)
    if cache_is_dynamic(past_key_values):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i][
                :, :, :new_len, :
            ]
            past_key_values.value_cache[i] = past_key_values.value_cache[i][
                :, :, :new_len, :
            ]
        return past_key_values
    return None


def basic_sample(logits: FloatTensor) -> tuple[LongTensor, FloatTensor]:
    """
    logprobs: (batch_size, vocab_size)
    return: (tokens, logprobs)
    tokens: (batch_size, 1)
    logprobs: (batch_size, vocab_size), logsoftmax of logits
    """
    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    new_token = torch.multinomial(probs, num_samples=1)  # shape (batch_size, 1)
    return new_token, logprobs


_mc_debug_iter = 0


def _mc_debug_enabled():
    return os.getenv("MC_DEBUG", "0") == "1"


def _mc_debug_limits():
    max_steps = int(os.getenv("MC_DEBUG_MAX_STEPS", "2"))
    max_iters = int(os.getenv("MC_DEBUG_MAX_ITERS", "2"))
    return max_steps, max_iters


def _mc_debug_log(msg):
    log_path = os.getenv(
        "MC_DEBUG_LOG",
        os.path.join(os.path.dirname(__file__), "..", "logs", "mc_debug.log"),
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _mc_topk(probs, k=5):
    k = min(k, probs.shape[-1])
    vals, idx = torch.topk(probs, k)
    return list(zip(idx.tolist(), vals.tolist()))


@torch.no_grad()
def safe_minus(log_q: FloatTensor, log_p: FloatTensor) -> FloatTensor:
    llr = log_q - log_p
    llr.nan_to_num_(nan=0.0)
    return llr


@torch.no_grad()
def logminusexp(log_a: FloatTensor, log_b: FloatTensor) -> FloatTensor:
    """
    log_a: torch.tensor, must be of full shape
    log_b: torch.tensor or scalar
    return: torch.tensor, log(exp(log_a)-exp(log_b))
    """
    return torch.where(
        log_a <= log_b + np.log(2),
        log_b + torch.log(torch.expm1(torch.clamp(safe_minus(log_a, log_b), min=0.0))),
        log_a + torch.log1p(-torch.exp(log_b - log_a)),
    )


#  from functools import wraps
#  from line_profiler import LineProfiler
#
#  profiler = LineProfiler()
#
#
#  def profile_each_line(func):
#      profiled_func = profiler(func)
#
#      @wraps(func)
#      def wrapper(*args, **kwargs):
#          return profiled_func(*args, **kwargs)
#
#      return wrapper


#  @profile_each_line
#  def mc_sample(logits, ref_logprobs, ref_tokens):
#      """
#      logits: torch.tensor of shape (seq_len,vocab_size)
#      ref_logprobs: torch.tensor of shape (seq_len,vocab_size)
#      ref_token: torch.tensor of shape (seq_len)
#      return: (gen_tokens, logprobs, poverlaps, fully_coupled)
#      gen_tokens: torch.tensor of shape (gen_seq_len)
#      logprobs: torch.tensor of shape (gen_seq_len,vocab_size)
#      poverlaps: torch.tensor of shape (gen_seq_len)
#      fully_coupled: bool
#      """
#      logprobs = F.log_softmax(logits, dim=-1)
#      prob_ratio = torch.exp(
#          torch.clamp(
#              torch.gather(
#                  logprobs - ref_logprobs, dim=-1, index=ref_tokens.unsqueeze(-1)
#              ).squeeze(-1),
#              max=0,
#          )
#      )
#      coupled = torch.rand_like(prob_ratio) <= prob_ratio
#      fully_coupled = bool(coupled.all())
#      if fully_coupled:
#          gen_seq_len = ref_tokens.shape[0]
#          gen_tokens = ref_tokens
#      else:
#          # find the location of first False
#          gen_seq_len = torch.argmin(coupled.int())
#          #  tprobs = torch.clamp(
#          #      torch.exp(logprobs[gen_seq_len]) - torch.exp(ref_logprobs[gen_seq_len]),
#          #      min=0.0,
#          #  )
#          tprobs = F.softmax(
#              logminusexp(logprobs[gen_seq_len], ref_logprobs[gen_seq_len]),
#              dim=-1,
#          )
#          gen_tokens = torch.cat(
#              [
#                  ref_tokens[:gen_seq_len],
#                  torch.multinomial(tprobs, num_samples=1),
#              ]
#          )
#          gen_seq_len = gen_seq_len + 1
#          logprobs = logprobs[:gen_seq_len]
#      poverlaps = torch.exp(
#          torch.min(logprobs[:gen_seq_len], ref_logprobs[:gen_seq_len])
#      ).sum(dim=-1)
#
#      return gen_tokens, logprobs, poverlaps, fully_coupled


#  @profile_each_line
def mc_sample(logits, ref_logprobs, ref_tokens):
    """
    logits: torch.tensor of shape (seq_len,vocab_size)
    ref_logprobs: torch.tensor of shape (seq_len,vocab_size)
    ref_token: torch.tensor of shape (seq_len)
    return: (gen_tokens, logprobs, poverlaps, fully_coupled)
    gen_tokens: torch.tensor of shape (gen_seq_len)
    logprobs: torch.tensor of shape (gen_seq_len,vocab_size)
    poverlaps: torch.tensor of shape (gen_seq_len)
    fully_coupled: bool
    """
    if logits.shape[-1] != ref_logprobs.shape[-1]:
        min_vocab = min(logits.shape[-1], ref_logprobs.shape[-1])
        logits = logits[..., :min_vocab]
        ref_logprobs = ref_logprobs[..., :min_vocab]
        if ref_tokens.max() >= min_vocab:
            ref_tokens = ref_tokens.clamp_max(min_vocab - 1)
    global _mc_debug_iter
    logprobs = F.log_softmax(logits, dim=-1)
    prob_ratio = torch.exp(
        torch.clamp(
            torch.gather(
                logprobs - ref_logprobs, dim=-1, index=ref_tokens.unsqueeze(-1)
            ).squeeze(-1),
            max=0,
        )
    )
    u = torch.rand_like(prob_ratio)
    coupled = u <= prob_ratio
    # coupled: (seq_len)
    coupled = F.pad(coupled, (0, 1), value=False)
    # coupled: (seq_len+1)
    couple_len = torch.argmin(coupled.int()).item()
    # couple_len: scalar, 0<=couple_len<=seq_len
    fully_coupled = couple_len == ref_tokens.shape[0]
    if _mc_debug_enabled():
        max_steps, max_iters = _mc_debug_limits()
        if _mc_debug_iter < max_iters:
            tprobs = torch.exp(logprobs)
            qprobs = torch.exp(ref_logprobs)
            _mc_debug_log(
                "MC iter "
                f"{_mc_debug_iter}: seq_len={ref_tokens.shape[0]} "
                f"fully_coupled={fully_coupled} couple_len={couple_len}"
            )
            for k in range(min(max_steps, prob_ratio.shape[0])):
                overlap = float(torch.min(tprobs[k], qprobs[k]).sum())
                _mc_debug_log(
                    "MC step "
                    f"{k}: prob_ratio={float(prob_ratio[k]):.6f} "
                    f"u={float(u[k]):.6f} coupled={bool(coupled[k])} "
                    f"overlap={overlap:.6f} "
                    f"top_p={_mc_topk(tprobs[k])} top_q={_mc_topk(qprobs[k])}"
                )
        _mc_debug_iter += 1

    if fully_coupled:
        gen_tokens = ref_tokens
    else:
        tprobs = torch.clamp(
            torch.exp(logprobs[couple_len]) - torch.exp(ref_logprobs[couple_len]),
            min=0.0,
        )
        if (not torch.isfinite(tprobs).all()) or tprobs.sum() <= 0:
            # Numerical fallback: sample from target probs to avoid invalid multinomial.
            tprobs = torch.exp(logprobs[couple_len])
        gen_tokens = torch.cat(
            [
                ref_tokens[:couple_len],
                torch.multinomial(
                    tprobs, num_samples=1
                ),  # sum of tprobs do not need to be 1
            ]
        )
        logprobs = logprobs[: couple_len + 1]
    poverlaps = torch.exp(
        torch.min(logprobs[: gen_tokens.shape[0]], ref_logprobs[: gen_tokens.shape[0]])
    ).sum(dim=-1)
    return gen_tokens, logprobs, poverlaps, fully_coupled


def mc_sample_oncpu(logits, ref_logprobs, ref_tokens):
    device = logits.device
    gen_tokens, logprobs, poverlaps, fully_coupled = mc_sample(
        logits.cpu(), ref_logprobs.cpu(), ref_tokens.cpu()
    )
    return (
        gen_tokens.to(device),
        logprobs.to(device),
        poverlaps.to(device),
        fully_coupled,
    )


def fix_gen_n_token_pass_key_values(ref_output_ids, gt_output_ids, ref_past_key_values):
    """
    ref_output_ids: torch.tensor of shape (batch_size, n-ni), batch_size must be 1
    gt_output_ids: torch.tensor of shape (batch_size, m-ni)
    ref_past_key_values: tuple of torch.tensor of shape (batch_size, num_heads, n-1, head_dim)
    return: past_key_values of shape (batch_size, num_heads, nm, head_dim)
    such that ref_output_ids[:, :nm] == gt_output_ids[:, :nm] and nm<n-ni
    """
    if ref_past_key_values is None:
        return None
    min_mn = min(ref_output_ids.shape[1], gt_output_ids.shape[1])
    sub_ref = ref_output_ids[:, :min_mn]
    sub_gt = gt_output_ids[:, :min_mn]
    match_n = min_mn - (sub_ref != sub_gt).cumsum(dim=1).to(torch.bool).sum(dim=1)[0]
    cached_n = cache_len(ref_past_key_values)
    # past_key_values includes all generated draft tokens, so drop the unmatched suffix
    keep_cached_n = cached_n - max(ref_output_ids.shape[1] - match_n, 0)
    return truncate_cache(ref_past_key_values, keep_cached_n)
