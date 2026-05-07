import numpy as np
import os
import time
from typing import Any
import torch
from torch import FloatTensor, LongTensor
from torch.utils._pytree import tree_map
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


AbstractWatermarkCode = Any
AbstractReweight = Any
AbstractContextCodeExtractor = Any
ContextCodeHistory = Any


class _UnbiasedWatermarkProxy:
    def __getattr__(self, name):
        if name in {
            "AbstractWatermarkCode",
            "AbstractReweight",
            "AbstractContextCodeExtractor",
        }:
            return Any
        import unbiased_watermark as _uwm

        return getattr(_uwm, name)


uwm = _UnbiasedWatermarkProxy()


def step_watermark(*args, **kwargs):
    from unbiased_watermark import step_watermark as _step_watermark
    # improving_KL's mc_watermark passes a 7th positional ;
    # the upstream signature is 6-positional. Drop the trailing temperature
    # arg if present (it is unused by the upstream implementation).
    if len(args) == 7:
        args = args[:6]
    kwargs.pop("temperature", None)
    return _step_watermark(*args, **kwargs)



def get_rng(*bs) -> np.random.Generator:
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
    full_hash = m.digest()
    seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
    return np.random.default_rng(seed)


def process_logits(input_ids, logits, logits_processor=None, logits_warper=None,
                   **_unused):
    """
    logits_processor: TODO
    logits_warper: TODO

    ``**_unused`` lets callers thread auxiliary scalar/callable fields
    (temperature, draft_temperature, draft_logits_warper, ...) through
    a single ``process_logits_kwargs`` dict; only ``logits_processor`` /
    ``logits_warper`` are consumed here, the rest are read by other
    consumers (drafter forwards in ``pfr.py`` swap in
    ``draft_logits_warper``; multi-draft ``MPFR_spec/mpfr_*`` reads
    scalar ``draft_temperature``).
    """
    if logits_processor is not None:
        logits = logits_processor(input_ids, logits)
    if logits_warper is not None:
        logits = logits_warper(input_ids, logits)
    return logits


def cache_is_legacy(past_key_values):
    return isinstance(past_key_values, (list, tuple))


def cache_is_dynamic(past_key_values):
    return isinstance(past_key_values, DynamicCache) or hasattr(past_key_values, "key_cache")


def dynamic_cache_to_legacy(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return tuple(zip(past_key_values.key_cache, past_key_values.value_cache))


def dynamic_cache_from_legacy(legacy_cache):
    return DynamicCache.from_legacy_cache(tuple(legacy_cache))


def cache_len(past_key_values):
    if past_key_values is None:
        return 0
    if cache_is_legacy(past_key_values):
        return past_key_values[0][0].shape[2]
    if cache_is_dynamic(past_key_values):
        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values.get_seq_length()
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
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(new_len)
            return past_key_values
        legacy_cache = dynamic_cache_to_legacy(past_key_values)
        legacy_cache = tree_map(lambda x: x[:, :, :new_len, :], legacy_cache)
        return dynamic_cache_from_legacy(legacy_cache)
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


def mc_sample_synthid(
    logits,
    ref_logprobs,
    ref_tokens,
    input_ids,
    cc_extractor,
    mc_private_key,
    reweight,
    temperature,
    psedo_r=False,
    residual_private_key=None,
):
    """
    logits: torch.tensor of shape (n,vocab_size)
    ref_logprobs: torch.tensor of shape (n,vocab_size)
    ref_token: torch.tensor of shape (n)
    input_ids: torch.tensor of shape (seq_len)
    cc_extractor: AbstractContextCodeExtractor
    mc_private_key: bytes
    residual_private_key: bytes, if provided, use this key for residual watermark sampling
    reweight: AbstractReweight
    temperature: float
    psedo_r: bool, if True, use psedo-random number generator
    """
    if logits.shape[-1] != ref_logprobs.shape[-1]:
        min_vocab = min(logits.shape[-1], ref_logprobs.shape[-1])
        logits = logits[..., :min_vocab]
        ref_logprobs = ref_logprobs[..., :min_vocab]
        if ref_tokens.max() >= min_vocab:
            ref_tokens = ref_tokens.clamp_max(min_vocab - 1)
    logprobs = F.log_softmax(logits, dim=-1)
    # During the accept or reject sampling, we didn't consider the temperature, we only care about the temperature influence on watermark signals.
    # Here since we didn't consider the temperture, the result probability is not 'truely' unbiased, but this is not important for this research.
    prob_ratio = torch.exp(
        torch.clamp(
            torch.gather(
                logprobs - ref_logprobs, dim=-1, index=ref_tokens.unsqueeze(-1)
            ).squeeze(-1),
            max=0,
        )
    )   # prob_ratio: (n)
    if psedo_r:
        accepted = torch.zeros_like(prob_ratio)  # shape (n)
        input_context = input_ids.unsqueeze(0)  # shape (1, seq_len)
        for i in range(accepted.shape[0]):
            cc_r = cc_extractor.extract(input_context)  # cc_r is a tuple
            rng_r = get_rng(cc_r[0], mc_private_key)  # cc_r[0] is bytes
            r = rng_r.random()
            accepted[i] = torch.tensor(r) <= prob_ratio[i]
            if accepted[i]:
                input_context = torch.cat([input_context, ref_tokens[i].unsqueeze(0).unsqueeze(0)], dim=1)
            else:
                break
        couple_len = int(torch.sum(accepted).item())
    else:
        coupled = torch.rand_like(prob_ratio) <= prob_ratio
        # coupled: (n)
        coupled = F.pad(coupled, (0, 1), value=False)
        # coupled: (n+1), couple_len = accepted tokens
        couple_len = torch.argmin(coupled.int()).item()
    # couple_len: scalar, 0<=couple_len<=n
    fully_coupled = couple_len == ref_tokens.shape[0]
    if fully_coupled:
        gen_tokens = ref_tokens
    else:
        tprobs = torch.clamp(
            torch.exp(logprobs[couple_len]) - torch.exp(ref_logprobs[couple_len]),
            min=0.0,
        )
        # normalize tprobs, shape (vocab_size)
        tprobs = tprobs / tprobs.sum(dim=-1, keepdim=True)
        t_logits = torch.log(tprobs).unsqueeze(0)
        t_logits_processed = t_logits / temperature  # shape (1, vocab_size)
        # input_ids: (seq_len)
        input_ids = torch.cat([input_ids, ref_tokens[:couple_len]]).unsqueeze(0)
        # input_ids: (1, seq_len)
        # embed watermark based on tprobs, here we do not consider the context code history
        cc = cc_extractor.extract(input_ids)
        rng = np.empty(cc.shape, dtype=object)
        residual_key = residual_private_key if residual_private_key is not None else mc_private_key
        for index in np.ndindex(rng.shape):
            rng[index] = get_rng(cc[index], residual_key)
        watermark_code_type = reweight.watermark_code_type
        watermark_code = reweight.watermark_code_type.from_random(rng, tprobs.size(-1))
        watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(input_ids.device))
        # diff_logits: (1, vocab_size), the input is probs and the output is logits, need to convert to probs!
        diff_logits = reweight.reweight_logits(watermark_code, t_logits_processed)
        diff_probs = torch.exp(diff_logits[0])

        gen_tokens = torch.cat(
            [
                ref_tokens[:couple_len],
                torch.multinomial(
                    diff_probs, num_samples=1
                ),
            ]
        )
        logprobs = logprobs[: couple_len + 1]   # shape (couple_len+1, vocab_size) = (gen_seq_len, vocab_size)

    poverlaps = torch.exp(
        torch.min(logprobs[: gen_tokens.shape[0]], ref_logprobs[: gen_tokens.shape[0]])
    ).sum(dim=-1)
    return gen_tokens, logprobs, poverlaps, fully_coupled


