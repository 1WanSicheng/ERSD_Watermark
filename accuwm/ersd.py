##this is just for ERSD, Poission exponetional race
"""
ERSD (Exponentially Smoothed Random Sampling Decoding) implementation.

This module implements the ERSD algorithm for speculative decoding using exponential races.
The algorithm uses exponential distributions to sample tokens, enabling efficient
speculative decoding with multiple draft tokens (K >= 1).

Algorithm overview (for max_draft_len = K):
1. For k = 0 to K-1:
   - Evaluate draft model Q on x:n||x̃1...x̃k-1 to get Q(i|x:n||x̃1...x̃k-1)
   - Sample ei_k ~ Exp(1) for all i
   - x̃k = arg min_i (ei_k / Q(i|x:n||x̃1...x̃k-1))
2. Evaluate target model P on x:n||x̃1...x̃K to get all distributions
3. For k = 0 to K-1:
   - x1^next_k = arg min_i (ei_k / P(i|x:n||x̃1...x̃k-1))
   - If x̃k = x1^next_k: accept x̃k, continue
   - Else: reject x̃k and all subsequent, return x1^next_k
4. If all K tokens accepted:
   - Sample x2^next ~ P(·|x:n||x̃1...x̃K)
   - Return (x̃1, ..., x̃K, x2^next)
"""

from .utils import *
from .basic import gen_n_token
from transformers import DynamicCache
import torch.nn.functional as F
import numpy as np
import os
import time
import hashlib


def _ersd_debug_enabled():
    return os.getenv("ERSD_DEBUG", "0") == "1"


def _ersd_debug_limits():
    max_steps = int(os.getenv("ERSD_DEBUG_MAX_STEPS", "2"))
    max_iters = int(os.getenv("ERSD_DEBUG_MAX_ITERS", "2"))
    return max_steps, max_iters


def _ersd_debug_log(msg):
    log_path = os.getenv(
        "ERSD_DEBUG_LOG",
        os.path.join(os.path.dirname(__file__), "..", "logs", "ersd_debug.log"),
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _ersd_topk(probs, k=5):
    k = min(k, probs.shape[-1])
    vals, idx = torch.topk(probs, k)
    return list(zip(idx.tolist(), vals.tolist()))


def log(t, eps=1e-20):
    """
    Safe logarithm function that clamps values to avoid log(0).
    
    Args:
        t: Input tensor
        eps: Minimum value to clamp to (default: 1e-20)
    
    Returns:
        log(t) with values clamped to eps minimum
    """
    return torch.log(t.clamp(min=eps))


def exponential_sample(probs, exp_noise=None, dim=-1):
    """
    Sample using exponential distribution: arg min_i (ei / P(i))
    
    This implements the exponential race sampling mechanism where we sample
    a token by finding the minimum ratio of exponential noise to probability.
    
    Args:
        probs: Probability distribution [..., vocab_size]
        exp_noise: Pre-sampled exponential noise [..., vocab_size]. If None, will sample.
        dim: Dimension to sample over (default: -1)
    
    Returns:
        sampled_indices: Indices of sampled tokens
    """
    if exp_noise is None:
        # Generate uniform noise and convert to exponential
        uniform_noise = torch.zeros_like(probs).uniform_(0, 1)
        exp_noise = -log(uniform_noise)

    # Compute ratios: ei / P(i) for each token i.
    # Avoid clamping to keep the sampling distribution exact; treat zero-prob as +inf.
    ratios = torch.where(
        probs > 0,
        exp_noise / probs,
        torch.full_like(probs, float("inf")),
    )

    # Return the index with minimum ratio (winner of the exponential race)
    return ratios.argmin(dim=dim)


def sample_exponential_noise(shape, device):
    """
    Sample exponential noise ei ~ Exp(1) for all tokens.
    
    We use the fact that Exp(1) = -log(U) where U ~ Uniform(0,1).
    
    Args:
        shape: Shape of the noise tensor
        device: Device to create tensor on
    
    Returns:
        exp_noise: Exponential noise tensor
    """
    # Generate uniform random numbers
    uniform_noise = torch.zeros(shape, device=device).uniform_(0, 1)
    # Convert to exponential: Exp(1) = -log(U) where U ~ Uniform(0,1)
    return -log(uniform_noise)


def _seed_from_tokens(input_ids, step, tag="race"):
    tokens = input_ids.detach().to("cpu").numpy().astype(np.int32, copy=False)
    h = hashlib.sha256()
    h.update(tokens.tobytes())
    h.update(str(step).encode("ascii"))
    h.update(tag.encode("ascii"))
    return int.from_bytes(h.digest()[:8], "little") & 0xFFFFFFFF


def sample_seeded_exponential_noise_from_tokens(input_ids, step, vocab_size, device):
    seed = _seed_from_tokens(input_ids, step)
    rng = np.random.default_rng(seed)
    uniform_noise = rng.random(vocab_size, dtype=np.float32)
    uniform_noise = torch.from_numpy(uniform_noise).unsqueeze(0).to(device)
    return -log(uniform_noise)


@torch.no_grad()
def gen_n_token_ersd(
    model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs={},
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, bool]:
    """
    Generate n draft tokens using exponential sampling (for ERSD algorithm).
    
    This function generates draft tokens using exponential races instead of
    standard multinomial sampling. It returns both the tokens and the exponential
    noises used, which are needed for verification.
    
    Args:
        model: Decoder-only model (reference/draft model)
        input_ids: (batch_size, seq_len), need to be on the same device and appropriate dtype
        n: number of tokens to generate
        past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
        max_vocab_size: Optional cap to align draft sampling with target vocabulary
        process_logits_kwargs: Keyword arguments for logits processing
    
    Returns:
        (output_ids, output_logprobs, exp_noises, past_key_values, got_eos)
        output_ids: (batch_size, n)
        output_logprobs: (batch_size, n, vocab_size)
        exp_noises: (n, vocab_size) - exponential noises used for sampling
        past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
        got_eos: bool
    """
    cached_n = cache_len(past_key_values)
    if cached_n > 0:
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids
    
    output_ids = []
    output_logprobs = []
    exp_noises = []
    device = model.device
    got_eos = False
    
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        if max_vocab_size is not None and max_vocab_size < logits.shape[-1]:
            logits = logits[..., :max_vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)  # [batch_size, vocab_size]

        # Sample seeded exponential noise for this step (draft side).
        vocab_size = probs.shape[1]
        t = input_ids.shape[1] + 1
        exp_noise = sample_seeded_exponential_noise_from_tokens(
            input_ids, t, vocab_size, device
        )
        exp_noises.append(exp_noise[0, :])  # Store for batch_size=1 case

        # Sample using exponential distribution: arg min_i (ei / P(i))
        new_token = exponential_sample(probs, exp_noise, dim=-1)
        new_token = new_token.unsqueeze(1)  # [batch_size, 1]
        
        output_logprobs.append(logprobs)
        input_tokens = new_token
        output_ids.append(new_token)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        past_key_values = output.past_key_values
        
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break
    
    output_ids = torch.cat(output_ids, dim=1)  # shape (batch_size, n)
    output_logprobs = torch.stack(
        output_logprobs, dim=1
    )  # shape (batch_size, n, vocab_size)
    exp_noises = torch.stack(exp_noises, dim=0)  # shape (n, vocab_size)
    
    return output_ids, output_logprobs, exp_noises, past_key_values, got_eos


@torch.no_grad()
def gen_ersd(
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_logprobs: FloatTensor,
    exp_noises: FloatTensor,
    past_key_values=None,
    process_logits_kwargs={},
    debug_iter=None,
    coupled: bool = True,
    return_meta: bool = False,
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, bool]:
    """
    Generate tokens using ERSD (Exponentially Smoothed Random Sampling Decoding) algorithm.
    
    This function verifies draft tokens generated by the reference model using the ERSD
    acceptance mechanism. It uses exponential races to determine acceptance/rejection.
    
    Args:
        model: Target model (decoder-only)
        input_ids: (batch_size, seq_len). batch_size must be 1
        ref_output_ids: (batch_size, n) - draft tokens from reference model
        ref_logprobs: (batch_size, n, vocab_size) - log probabilities from reference model
        exp_noises: (n, vocab_size) - exponential noise for each draft position
        past_key_values: KV cache for target model
        process_logits_kwargs: Keyword arguments for logits processing
    
    Returns:
        (output_ids, output_logprobs, poverlaps, past_key_values, got_eos)
        output_ids: (batch_size, gen_len) - accepted tokens (1 to n+1 tokens)
        output_logprobs: (batch_size, gen_len, vocab_size) - log probabilities
        poverlaps: (batch_size, min(gen_len,n)) - probability overlaps (for compatibility)
        past_key_values: Updated KV cache
        got_eos: bool - whether EOS token was generated
    """
    assert input_ids.shape[0] == 1, "ERSD only supports batch_size=1"
    assert ref_output_ids.shape == ref_logprobs.shape[:-1]
    assert exp_noises.shape[0] == ref_output_ids.shape[1], \
        f"exp_noises shape {exp_noises.shape[0]} must match ref_output_ids length {ref_output_ids.shape[1]}"
    
    n = ref_output_ids.shape[1]  # Number of draft tokens
    device = input_ids.device
    
    # Prepare input sequence: concatenate original input with draft tokens
    cached_n = cache_len(past_key_values)
    if cached_n > 0:
        # Get cached length
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    
    # Full sequence for logits processing
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    # _ids: (batch_size, seq_len+n)
    
    # Single forward pass to get all logits for positions we need
    # We need logits for: [P(i|x:n), P(i|x:n||x̃1), ..., P(i|x:n||x̃1...x̃K)]
    output = model(input_tokens, past_key_values=past_key_values)
    logits = output.logits
    # Extract logits for the positions we need (last n+1 positions)
    logits = logits[:, -n - 1:, :]
    # logits: (batch_size, n+1, vocab_size)
    
    # Process logits (apply temperature, top-p, top-k, etc.)
    if process_logits_kwargs != {}:
        for i in range(-1, n):
            logits[:, i + 1, :] = process_logits(
                _ids[:, : _ids.shape[1] - n + i + 1],
                logits[:, i + 1, :],
                **process_logits_kwargs,
            )
    
    # Convert to probabilities
    target_probs = F.softmax(logits, dim=-1)
    # Align vocab sizes across target probs, ref logprobs, and exp noises
    vocab_size = min(
        target_probs.shape[-1],
        ref_logprobs.shape[-1],
        exp_noises.shape[-1],
    )
    if vocab_size < target_probs.shape[-1]:
        target_probs = target_probs[..., :vocab_size]
    if vocab_size < ref_logprobs.shape[-1]:
        ref_logprobs = ref_logprobs[..., :vocab_size]
    if vocab_size < exp_noises.shape[-1]:
        exp_noises = exp_noises[..., :vocab_size]
    # Renormalize on the shared vocabulary so sampling and overlaps use the same support.
    eps = 1e-12
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
    ref_probs = ref_logprobs.exp()
    ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
    # target_probs: (batch_size, n+1, vocab_size)
    # target_probs[:, k, :] = P(i|x:n||x̃1...x̃k-1)
    # target_probs[:, k+1, :] = P(i|x:n||x̃1...x̃k)
    
    # Verify each draft token sequentially using exponential races
    accepted_tokens = []
    original_seq_len = input_ids.shape[1]
    accepted_draft_len = 0
    target_gaps = []
    accept_indicators = []
    do_debug = _ersd_debug_enabled()
    max_debug_steps, max_debug_iters = _ersd_debug_limits()
    debug_active = do_debug and (debug_iter is not None) and (debug_iter < max_debug_iters)
    if debug_active:
        _ersd_debug_log(
            "ERSD iter "
            f"{debug_iter}: input_len={input_ids.shape[1]} cached_n={cached_n} "
            f"draft_len={n} input_tokens_len={input_tokens.shape[1]} "
            f"logits_shape={tuple(logits.shape)}"
        )
    
    for k in range(n):
        # Get target distribution before this draft token
        # target_probs[:, k, :] = P(i|x:n||x̃1...x̃k-1)
        target_probs_before = target_probs[0, k, :]  # [vocab_size]
        # Get exponential noise for this position
        prefix_len = input_ids.shape[1] + k
        t = prefix_len + 1
        prefix_tokens = _ids[:, :prefix_len]
        if coupled:
            exp_noise_k = sample_seeded_exponential_noise_from_tokens(
                prefix_tokens, t, vocab_size, device
            )[0, :]  # [vocab_size]
        else:
            exp_noise_k = sample_exponential_noise((1, vocab_size), device)[0, :]
        
        # Compute x1^next_k = arg min_i (ei_k / P(i|x:n||x̃1...x̃k-1))
        # Use exponential_sample to find the token with minimum ratio
        ratios = torch.where(
            target_probs_before > 0,
            exp_noise_k / target_probs_before,
            torch.full_like(target_probs_before, float("inf")),
        )
        if ratios.numel() >= 2:
            top2 = torch.topk(-ratios, k=2)
            min1 = -top2.values[0]
            min2 = -top2.values[1]
            target_gaps.append(float((min2 - min1).item()))
        else:
            target_gaps.append(float("nan"))
        x1_next_k = exponential_sample(
            target_probs_before.unsqueeze(0),  # [1, vocab_size]
            exp_noise_k.unsqueeze(0),  # [1, vocab_size]
            dim=-1
        )
        x1_next_k = x1_next_k.item()
        
        # Get draft token value
        draft_token_k = ref_output_ids[0, k].item()
        
        # Debug: step-level alignment and distribution snapshot.
        if debug_active and k < max_debug_steps:
            prefix_len = _ids.shape[1] - n + k
            prefix_tail = _ids[0, max(prefix_len - 8, 0) : prefix_len].tolist()
            draft_prob = float(ref_probs[0, k, draft_token_k]) if draft_token_k < vocab_size else 0.0
            target_prob = float(target_probs_before[draft_token_k]) if draft_token_k < vocab_size else 0.0
            noise_draft = float(exp_noise_k[draft_token_k]) if draft_token_k < vocab_size else float("nan")
            noise_x1 = float(exp_noise_k[x1_next_k]) if x1_next_k < vocab_size else float("nan")
            overlap = float(torch.min(target_probs_before, ref_probs[0, k, :]).sum())
            accepted = draft_token_k == x1_next_k
            _ersd_debug_log(
                "ERSD step "
                f"{k}: prefix_len={prefix_len} prefix_tail={prefix_tail} "
                f"draft_token={draft_token_k} x1_next={x1_next_k} "
                f"p_draft={target_prob:.6f} q_draft={draft_prob:.6f} "
                f"noise_draft={noise_draft:.6f} noise_x1={noise_x1:.6f} "
                f"overlap={overlap:.6f} accepted={accepted} "
                f"top_p={_ersd_topk(target_probs_before)} top_q={_ersd_topk(ref_probs[0, k, :])}"
            )
            if os.getenv("ERSD_DEBUG_VERIFY", "0") == "1":
                # Verify logits alignment by recomputing this prefix.
                verify_ids = _ids[:, :prefix_len]
                verify_out = model(verify_ids)
                verify_logits = verify_out.logits[:, -1, :]
                verify_logits = process_logits(
                    verify_ids,
                    verify_logits,
                    **process_logits_kwargs,
                )
                verify_probs = F.softmax(verify_logits, dim=-1)
                verify_probs = verify_probs[..., :vocab_size]
                verify_probs = verify_probs / verify_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
                diff = (verify_probs[0] - target_probs_before).abs()
                _ersd_debug_log(
                    "ERSD verify "
                    f"{k}: max_abs_diff={float(diff.max()):.6e} mean_abs_diff={float(diff.mean()):.6e}"
                )

        # Check if x̃k = x1^next_k (acceptance condition)
        if draft_token_k == x1_next_k:
            # Accept x̃k
            accepted_tokens.append(draft_token_k)
            accepted_draft_len += 1
            accept_indicators.append(1)
            
            # If this is the last draft token and it's accepted, generate additional token
            if k == n - 1:
                # Sample x2^next ~ P(·|x:n||x̃1...x̃K)
                # target_probs[:, n, :] = P(i|x:n||x̃1...x̃K)
                target_probs_after = target_probs[0, n, :]  # [vocab_size]
                x2_next = torch.multinomial(target_probs_after.unsqueeze(0), num_samples=1)
                accepted_tokens.append(x2_next[0, 0].item())
        else:
            # Reject x̃k and all subsequent tokens
            # Return x1^next_k
            accepted_tokens.append(x1_next_k)
            accept_indicators.append(0)
            break
    
    # Build output tensors
    accept_count = len(accepted_tokens)
    output_ids = torch.tensor([accepted_tokens], device=device, dtype=torch.long)
    # output_ids: (1, accept_count)
    
    # Compute output_logprobs for accepted tokens
    output_logprobs = []
    for k in range(accept_count):
        if k < n:
            # For accepted draft tokens, use the target distribution at position k
            output_logprobs.append(target_probs[0, k, :].log().unsqueeze(0))
        else:
            # For the additional token (x2^next), use the distribution at position n
            output_logprobs.append(target_probs[0, n, :].log().unsqueeze(0))
    output_logprobs = torch.stack(output_logprobs, dim=1)
    # output_logprobs: (1, accept_count, vocab_size)
    
    # Compute poverlaps for compatibility with MC sampling
    # poverlaps measures the probability overlap between draft and target for each position
    poverlaps = []
    for k in range(min(accept_count, n)):
        # Get draft probability for the accepted token at position k
        draft_token_k = output_ids[0, k].item()
        if draft_token_k >= vocab_size:
            draft_prob_k = torch.tensor(0.0, device=device)
        else:
            draft_prob_k = ref_probs[0, k, draft_token_k]
        # Get target probability for the accepted token at position k
        if draft_token_k >= vocab_size:
            target_prob_k = torch.tensor(0.0, device=device)
        else:
            target_prob_k = target_probs[0, k, draft_token_k]
        # Overlap is the minimum of the two probabilities for the accepted token
        poverlap_k = torch.min(draft_prob_k, target_prob_k)
        poverlaps.append(poverlap_k)
    poverlaps = torch.stack(poverlaps) if poverlaps else torch.tensor([], device=device)
    poverlaps = poverlaps.unsqueeze(0)  # [1, min(accept_count, n)]
    
    # Check for EOS
    got_eos = False
    if output_ids.shape[1] > 0 and output_ids[0, -1] == model.config.eos_token_id:
        got_eos = True
    
    # Fix past_key_values: cache should include up to original_seq + accepted_tokens - 1
    # (excluding the last token which will be processed in the next iteration)
    past_key_values = output.past_key_values
    cache_len_needed = original_seq_len + accept_count - 1
    past_key_values = truncate_cache(past_key_values, cache_len_needed)
    verify_ops = accepted_draft_len if accepted_draft_len == n else accepted_draft_len + 1
    if return_meta:
        meta = {
            "accepted_draft_len": accepted_draft_len,
            "target_gaps": target_gaps,
            "accept_indicators": accept_indicators,
            "verify_ops": verify_ops,
            "draft_len": n,
        }
        return (
            output_ids,
            output_logprobs,
            poverlaps,
            past_key_values,
            got_eos,
            meta,
        )
    return (
        output_ids,
        output_logprobs,
        poverlaps,
        past_key_values,
        got_eos,
    )


def ersd_sample_generator(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    coupled: bool = True,
    return_meta: bool = False,
    **kwargs,
):
    """
    Generator function for ERSD speculative decoding.
    
    This generator uses the ERSD algorithm where:
    - Each iteration generates n draft tokens using the reference model
    - Draft tokens are verified using exponential races
    - If all n tokens are accepted, generates 1 additional token (total n+1 tokens)
    - If rejected at position k, generates k accepted tokens (total k tokens)
    
    Args:
        model: Target model (large model)
        ref_model: Reference/draft model (small model)
        input_ids: Initial sequence [batch, seq_len]
        n: Number of draft tokens to generate per iteration (max_draft_len)
        past_key_values: KV cache for target model
        ref_past_key_values: KV cache for reference model
        **kwargs: Additional arguments (e.g., process_logits_kwargs)
    
    Yields:
        (output_ids, output_logprobs) tuples for each iteration
        output_ids: (batch_size, gen_len) - generated tokens
        output_logprobs: (batch_size, gen_len, vocab_size) - log probabilities
    """
    model.eval()
    ref_model.eval()
    
    debug_iter = 0
    while True:
        # Step 1: Generate draft tokens using reference model with exponential sampling
        # This function generates draft tokens using exponential races and returns
        # both the tokens and the exponential noises used
        ref_output_ids, ref_logprobs, exp_noises, ref_past_key_values, _got_eos = gen_n_token_ersd(
            ref_model,
            input_ids,
            n,
            past_key_values=ref_past_key_values,
            max_vocab_size=model.config.vocab_size,
            **kwargs,
        )
        # ref_output_ids: (batch_size, n)
        # ref_logprobs: (batch_size, n, vocab_size)
        # exp_noises: (n, vocab_size) - exponential noises used for draft generation
        
        # Step 3: Verify draft tokens using target model
        output_ids, output_logprobs, poverlaps, past_key_values, got_eos, meta = gen_ersd(
            model,
            input_ids,
            ref_output_ids,
            ref_logprobs,
            exp_noises,
            past_key_values=past_key_values,
            **kwargs,
            debug_iter=debug_iter,
            coupled=coupled,
            return_meta=True,
        )
        
        # Step 4: Fix reference model cache to match accepted tokens
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        
        # Yield results
        if return_meta:
            yield output_ids, output_logprobs, meta
        else:
            yield output_ids, output_logprobs
        
        # Update input_ids for next iteration
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        
        # Check for EOS
        if got_eos:
            break
        debug_iter += 1
