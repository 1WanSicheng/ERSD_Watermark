"""
ERSD with Watermark implementation.

This module implements the watermarked ERSD algorithm for speculative decoding using seeded exponential races.
The algorithm uses seeded PRF-generated exponential distributions to sample tokens, enabling both
efficient speculative decoding and watermark detection.

Algorithm overview (for max_draft_len = K):
1. For k = 0 to K-1:
   - Evaluate draft model Q on x:n||x̃1...x̃k-1 to get Q(i|x:n||x̃1...x̃k-1)
   - Generate seeded exponential noise e_{n+1,race}(i) using PRF
   - x̃k = arg min_i (e_{n+1,race}(i) / Q(i|x:n||x̃1...x̃k-1))
2. Evaluate target model P on x:n||x̃1...x̃K to get all distributions
3. For k = 0 to K-1:
   - x1^next_k = arg min_i (e_{n+1,race}(i) / P(i|x:n||x̃1...x̃k-1))
   - If x̃k = x1^next_k: accept x̃k, continue
   - Else: reject x̃k and all subsequent, return x1^next_k
4. If all K tokens accepted:
   - Generate seeded exponential noise e_{n+2,cont}(i) using PRF
   - Sample x2^next ~ P(·|x:n||x̃1...x̃K) using e_{n+2,cont}(i)
   - Return (x̃1, ..., x̃K, x2^next)
"""

from .utils import *
from .basic_watermark import gen_n_token_uwm
from .ersd import (
    log,
    exponential_sample,
)
import torch.nn.functional as F
import os
import time
import unbiased_watermark as uwm
from unbiased_watermark.lm import get_rng


def _ersd_wm_debug_enabled():
    return os.getenv("ERSD_WM_DEBUG", "0") == "1"


def _ersd_wm_debug_limits():
    max_steps = int(os.getenv("ERSD_WM_DEBUG_MAX_STEPS", "2"))
    max_iters = int(os.getenv("ERSD_WM_DEBUG_MAX_ITERS", "2"))
    return max_steps, max_iters


def _ersd_wm_debug_log(msg):
    log_path = os.getenv(
        "ERSD_WM_DEBUG_LOG",
        os.path.join(os.path.dirname(__file__), "..", "logs", "ersd_wm_debug.log"),
    )
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _ersd_wm_topk(probs, k=5):
    k = min(k, probs.shape[-1])
    vals, idx = torch.topk(probs, k)
    return list(zip(idx.tolist(), vals.tolist()))


def sample_seeded_exponential_noise(
    cc: np.ndarray,
    t: int,
    tag: str,
    private_key: bytes,
    vocab_size: int,
    device,
    nonce: bytes = None,
):
    """
    Generate seeded exponential noise e_{t,tag}(i) = -log r_{t,tag}(i),
    where r_{t,tag}(i) = F_{(c_{t-1}, t, tag, ν), k}(i) is a keyed pseudorandom value.
    
    Args:
        cc: Context code (c_{t-1}) as numpy array (batch_shape,)
        t: Time step (position)
        tag: Tag string, either "race" or "cont"
        private_key: Secret key k
        vocab_size: Vocabulary size
        device: Device to create tensor on
        nonce: Optional public nonce ν
    
    Returns:
        exp_noise: Exponential noise tensor [batch_size, vocab_size]
    """
    # Create seed: (c_{t-1}, t, tag, ν), k
    # Convert time step and tag to bytes
    t_bytes = t.to_bytes(8, byteorder='big')
    tag_bytes = tag.encode('utf-8')
    
    # Build seed components
    seed_components = [cc, t_bytes, tag_bytes]
    if nonce is not None:
        seed_components.append(nonce)
    
    # Generate RNG from seed
    rng = get_rng(*seed_components, private_key)
    
    # Generate uniform random numbers r_{t,tag}(i) for all tokens i
    uniform_noise = torch.from_numpy(rng.random(vocab_size).astype(np.float32))
    uniform_noise = uniform_noise.unsqueeze(0).to(device)
    
    # Convert to exponential: e_{t,tag}(i) = -log(r_{t,tag}(i))
    exp_noise = -log(uniform_noise)
    
    return exp_noise


@torch.no_grad()
def gen_n_token_ersd_wm(
    reweight: uwm.AbstractReweight,
    cc_extractor: uwm.AbstractContextCodeExtractor,
    cch: uwm.lm.ContextCodeHistory,
    private_key: bytes,
    model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    max_vocab_size: int | None = None,
    process_logits_kwargs={},
    nonce: bytes = None,
) -> tuple[LongTensor, FloatTensor, FloatTensor, np.ndarray, any, bool]:
    """
    Generate n draft tokens using seeded exponential sampling (for watermarked ERSD algorithm).
    
    This function generates draft tokens using seeded exponential races instead of
    standard multinomial sampling. It returns both the tokens and the exponential
    noises used, which are needed for verification.
    
    Args:
        reweight: Reweight object (for compatibility, not used in ERSD)
        cc_extractor: Context code extractor
        cch: Context code history
        private_key: Secret key for PRF
        model: Decoder-only model (reference/draft model)
        input_ids: (batch_size, seq_len), need to be on the same device and appropriate dtype
        n: number of tokens to generate
        past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
        max_vocab_size: Optional cap to align draft sampling with target vocabulary
        process_logits_kwargs: Keyword arguments for logits processing
        nonce: Optional public nonce
    
    Returns:
        (output_ids, output_logprobs, exp_noises, context_codes, past_key_values, got_eos)
        output_ids: (batch_size, n)
        output_logprobs: (batch_size, n, vocab_size)
        exp_noises: (n, vocab_size) - seeded exponential noises used for sampling
        context_codes: (batch_size, n) - context codes for each position
        past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
        got_eos: bool
    """
    assert input_ids.shape[0] == 1, "ERSD only supports batch_size=1"
    assert cch.data.shape == input_ids.shape[:-1]
    
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
    
    # Compute per-step context codes from the current prefix without mutating cch.
    vocab_size = model.config.vocab_size
    if max_vocab_size is not None:
        vocab_size = min(vocab_size, max_vocab_size)
    context_codes = []
    
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        if vocab_size < logits.shape[-1]:
            logits = logits[..., :vocab_size]
        logprobs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(logprobs)  # [batch_size, vocab_size]
        
        # Use per-step context code and seeded exponential noise.
        cc_step = cc_extractor.extract(input_ids)
        context_codes.append(cc_step[0])
        t = input_ids.shape[1] + 1  # Position of current draft token
        exp_noise_race = sample_seeded_exponential_noise(
            cc_step[0],  # Extract context code for batch 0
            t,
            "race",
            private_key,
            vocab_size,
            device,
            nonce=nonce,
        )
        exp_noise_race = exp_noise_race[0, :]  # [vocab_size]
        exp_noises.append(exp_noise_race.clone())
        
        # Sample using exponential distribution: arg min_i (e_{n+1,race}(i) / P(i))
        new_token = exponential_sample(probs, exp_noise_race.unsqueeze(0), dim=-1)
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
    # Shape to (batch_size, n); avoid stacking which would transpose.
    context_codes = np.array([context_codes], dtype=object)
    
    return output_ids, output_logprobs, exp_noises, context_codes, past_key_values, got_eos


@torch.no_grad()
def gen_ersd_wm(
    reweight: uwm.AbstractReweight,
    cc_extractor: uwm.AbstractContextCodeExtractor,
    cch: uwm.lm.ContextCodeHistory,
    private_key: bytes,
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_logprobs: FloatTensor,
    exp_noises: FloatTensor,
    ref_context_codes: np.ndarray,
    past_key_values=None,
    process_logits_kwargs={},
    nonce: bytes = None,
    debug_iter=None,
) -> tuple[
    LongTensor,
    FloatTensor,
    FloatTensor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    uwm.AbstractWatermarkCode,
    np.ndarray,
    any,
    bool,
]:
    """
    Generate tokens using watermarked ERSD algorithm.
    
    This function verifies draft tokens generated by the reference model using the ERSD
    acceptance mechanism with seeded exponential races.
    
    Args:
        reweight: Reweight object (for compatibility)
        cc_extractor: Context code extractor
        cch: Context code history
        private_key: Secret key for PRF
        model: Target model (decoder-only)
        input_ids: (batch_size, seq_len). batch_size must be 1
        ref_output_ids: (batch_size, n) - draft tokens from reference model
        ref_logprobs: (batch_size, n, vocab_size) - log probabilities from reference model
        exp_noises: (n, vocab_size) - seeded exponential noise for each draft position
        ref_context_codes: (batch_size, n) - context codes from draft generation
        past_key_values: KV cache for target model
        process_logits_kwargs: Keyword arguments for logits processing
        nonce: Optional public nonce
    
    Returns:
        (output_ids, output_logprobs, poverlaps, context_codes, time_steps, tags, watermark_code, skipped, past_key_values, got_eos)
        output_ids: (batch_size, gen_len) - accepted tokens (1 to n+1 tokens)
        output_logprobs: (batch_size, gen_len, vocab_size) - log probabilities
        poverlaps: (batch_size, min(gen_len,n)) - probability overlaps (for compatibility)
        context_codes: (batch_size, gen_len) - context codes for each position
        time_steps: (batch_size, gen_len) - time steps for each position
        tags: (batch_size, gen_len) - tags ("race" or "cont") for each position
        watermark_code: Watermark code for continuation token (if generated)
        skipped: (batch_size, gen_len) - skipped flags
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
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    
    # Full sequence for logits processing
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    # _ids: (batch_size, seq_len+n)
    
    # Single forward pass to get all logits for positions we need
    output = model(input_tokens, past_key_values=past_key_values)
    logits = output.logits
    logits = logits[:, -n - 1:, :]
    # logits: (batch_size, n+1, vocab_size)
    
    # Process logits
    if process_logits_kwargs != {}:
        for i in range(-1, n):
            logits[:, i + 1, :] = process_logits(
                _ids[:, : _ids.shape[1] - n + i + 1],
                logits[:, i + 1, :],
                **process_logits_kwargs,
            )
    
    # Convert to probabilities
    target_probs = F.softmax(logits, dim=-1)
    # Align vocab sizes
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
    
    # Verify each draft token sequentially using exponential races
    accepted_tokens = []
    context_codes = []
    skipped_flags = []
    original_seq_len = input_ids.shape[1]
    do_debug = _ersd_wm_debug_enabled()
    max_debug_steps, max_debug_iters = _ersd_wm_debug_limits()
    debug_active = do_debug and (debug_iter is not None) and (debug_iter < max_debug_iters)
    if debug_active:
        _ersd_wm_debug_log(
            "ERSD_WM iter "
            f"{debug_iter}: input_len={input_ids.shape[1]} cached_n={cached_n} "
            f"draft_len={n} input_tokens_len={input_tokens.shape[1]} "
            f"logits_shape={tuple(logits.shape)}"
        )
    
    for k in range(n):
        target_probs_before = target_probs[0, k, :]  # [vocab_size]
        prefix_len = input_ids.shape[1] + k
        prefix_tokens = _ids[:, :prefix_len]
        cc_step = cc_extractor.extract(prefix_tokens)[0]
        t = prefix_len + 1
        exp_noise_k = sample_seeded_exponential_noise(
            cc_step,
            t,
            "race",
            private_key,
            vocab_size,
            device,
            nonce=nonce,
        )[0, :]  # [vocab_size]
        
        # Compute x1^next_k = arg min_i (e_{n+1,race}(i) / P(i|x:n||x̃1...x̃k-1))
        x1_next_k = exponential_sample(
            target_probs_before.unsqueeze(0),
            exp_noise_k.unsqueeze(0),
            dim=-1
        )
        x1_next_k = x1_next_k.item()
        
        draft_token_k = ref_output_ids[0, k].item()

        # Debug: step-level alignment and context code consistency.
        if debug_active and k < max_debug_steps:
            prefix_len = _ids.shape[1] - n + k
            prefix_tail = _ids[0, max(prefix_len - 8, 0) : prefix_len].tolist()
            draft_prob = float(ref_probs[0, k, draft_token_k]) if draft_token_k < vocab_size else 0.0
            target_prob = float(target_probs_before[draft_token_k]) if draft_token_k < vocab_size else 0.0
            noise_draft = float(exp_noise_k[draft_token_k]) if draft_token_k < vocab_size else float("nan")
            noise_x1 = float(exp_noise_k[x1_next_k]) if x1_next_k < vocab_size else float("nan")
            overlap = float(torch.min(target_probs_before, ref_probs[0, k, :]).sum())
            accepted = draft_token_k == x1_next_k
            cc_expected = cc_step
            cc_draft = ref_context_codes[0, k]
            cc_match = bool(np.array_equal(cc_expected, cc_draft))
            _ersd_wm_debug_log(
                "ERSD_WM step "
                f"{k}: prefix_len={prefix_len} prefix_tail={prefix_tail} "
                f"draft_token={draft_token_k} x1_next={x1_next_k} "
                f"p_draft={target_prob:.6f} q_draft={draft_prob:.6f} "
                f"noise_draft={noise_draft:.6f} noise_x1={noise_x1:.6f} "
                f"overlap={overlap:.6f} accepted={accepted} "
                f"cc_match={cc_match} "
                f"top_p={_ersd_wm_topk(target_probs_before)} top_q={_ersd_wm_topk(ref_probs[0, k, :])}"
            )
            if os.getenv("ERSD_WM_DEBUG_VERIFY", "0") == "1":
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
                _ersd_wm_debug_log(
                    "ERSD_WM verify "
                    f"{k}: max_abs_diff={float(diff.max()):.6e} mean_abs_diff={float(diff.mean()):.6e}"
                )

        # Check if x̃k = x1^next_k (acceptance condition)
        if draft_token_k == x1_next_k:
            accepted_tokens.append(draft_token_k)
            # Use context code from draft generation (all draft tokens share c_n)
            context_codes.append(ref_context_codes[0, k])
            skipped_flags.append(False)
            
            # If this is the last draft token and it's accepted, generate additional token
            if k == n - 1:
                # All n draft tokens were accepted
                # Generate context code c_{n+1} for continuation token
                # c_{n+1} = (x:n||x̃1...x̃K)_{n+1-m+1:n+1}
                # new_context = input_ids || all accepted draft tokens
                # _ids contains input_ids || ref_output_ids (all draft tokens, all accepted)
                new_context = _ids  # input_ids || ref_output_ids (all n draft tokens)
                cc_n1, skipped_n1 = cch.step(cc_extractor, new_context)
                context_codes.append(cc_n1[0])
                
                # Generate seeded exponential noise e_{n+2,cont}(i) for continuation token
                # According to algorithm: if all K draft tokens accepted, continuation token
                # is at position n+K+1 (where n is input_ids length)
                # But the algorithm notation uses e_{n+2,cont} which suggests K=1 case
                # For general K, continuation token is at position L + K + 1
                L = input_ids.shape[1]  # Original input_ids length
                t_cont = L + n + 1  # Position after all n draft tokens (L + n + 1)
                exp_noise_cont = sample_seeded_exponential_noise(
                    cc_n1[0],
                    t_cont,
                    "cont",
                    private_key,
                    vocab_size,
                    device,
                    nonce=nonce,
                )
                exp_noise_cont = exp_noise_cont[0, :]  # [vocab_size]
                
                # Sample x2^next using e_{n+2,cont}(i)
                # Target logits must be based on new_context (input_ids || all draft tokens)
                target_probs_after = target_probs[0, n, :]  # P(·|x:n||x̃1...x̃K)
                x2_next = exponential_sample(
                    target_probs_after.unsqueeze(0),
                    exp_noise_cont.unsqueeze(0),
                    dim=-1
                )
                accepted_tokens.append(x2_next.item())
                skipped_flags.append(skipped_n1[0])
        else:
            # Reject x̃k and all subsequent tokens
            # For rejected token, we need to generate context code based on
            # input_ids || accepted_tokens (excluding the rejected draft token)
            # But since we're rejecting at position k, the context should be based on
            # input_ids || ref_output_ids[:, :k] (accepted draft tokens before rejection)
            if k > 0:
                # There are some accepted draft tokens before rejection
                accepted_draft_seq = torch.cat([input_ids, ref_output_ids[:, :k]], dim=1)
            else:
                # No draft tokens were accepted
                accepted_draft_seq = input_ids
            cc_reject, skipped_reject = cch.step(cc_extractor, accepted_draft_seq)
            accepted_tokens.append(x1_next_k)
            context_codes.append(cc_reject[0])
            skipped_flags.append(skipped_reject[0])
            break
    
    # Build output tensors
    accept_count = len(accepted_tokens)
    output_ids = torch.tensor([accepted_tokens], device=device, dtype=torch.long)
    
    # Compute output_logprobs
    output_logprobs = []
    for k in range(accept_count):
        if k < n:
            output_logprobs.append(target_probs[0, k, :].log().unsqueeze(0))
        else:
            output_logprobs.append(target_probs[0, n, :].log().unsqueeze(0))
    output_logprobs = torch.stack(output_logprobs, dim=1)
    
    # Compute poverlaps
    poverlaps = []
    for k in range(min(accept_count, n)):
        draft_token_k = output_ids[0, k].item()
        if draft_token_k >= vocab_size:
            draft_prob_k = torch.tensor(0.0, device=device)
            target_prob_k = torch.tensor(0.0, device=device)
        else:
            draft_prob_k = ref_probs[0, k, draft_token_k]
            target_prob_k = target_probs[0, k, draft_token_k]
        poverlap_k = torch.min(draft_prob_k, target_prob_k)
        poverlaps.append(poverlap_k)
    poverlaps = torch.stack(poverlaps) if poverlaps else torch.tensor([], device=device)
    poverlaps = poverlaps.unsqueeze(0)
    
    # Build context codes and skipped arrays
    context_codes = np.array(context_codes)  # (accept_count,)
    skipped = np.array(skipped_flags)  # (accept_count,)
    # Build time steps and tags for ERSD Aaronson detection
    time_steps = np.arange(
        original_seq_len + 1, original_seq_len + accept_count + 1, dtype=np.int64
    )
    tags = np.array(["race"] * accept_count, dtype=object)
    if accept_count > n:
        tags[-1] = "cont"
    
    # Generate watermark code for continuation token if it exists
    watermark_code = None
    if accept_count > n:
        # Continuation token was generated, create watermark code for it
        # This is needed for detection
        cc_cont = context_codes[-1]
        L = input_ids.shape[1]  # Original input_ids length
        t_cont = L + n + 1  # Position after all n draft tokens
        t_cont_bytes = t_cont.to_bytes(8, byteorder='big')
        seed_components = [cc_cont, t_cont_bytes, b"cont"]
        if nonce is not None:
            seed_components.append(nonce)
        rng_cont = get_rng(*seed_components, private_key)
        watermark_code = reweight.watermark_code_type.from_random(
            np.array([rng_cont], dtype=object),
            vocab_size
        )
        watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(device))
    
    # Check for EOS
    got_eos = False
    if output_ids.shape[1] > 0 and output_ids[0, -1] == model.config.eos_token_id:
        got_eos = True
    
    # Fix past_key_values
    past_key_values = output.past_key_values
    cache_len_needed = original_seq_len + accept_count - 1
    past_key_values = truncate_cache(past_key_values, cache_len_needed)
    
    return (
        output_ids,
        output_logprobs,
        poverlaps,
        context_codes,
        time_steps,
        tags,
        watermark_code,
        skipped,
        past_key_values,
        got_eos,
    )


def ersd_wm_sample_generator(
    reweight: uwm.AbstractReweight,
    cc_extractor: uwm.AbstractContextCodeExtractor,
    cch: uwm.lm.ContextCodeHistory,
    private_key: bytes,
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    nonce: bytes = None,
    **kwargs,
):
    """
    Generator function for watermarked ERSD speculative decoding.
    
    Args:
        reweight: Reweight object
        cc_extractor: Context code extractor
        cch: Context code history
        private_key: Secret key for PRF
        model: Target model (large model)
        ref_model: Reference/draft model (small model)
        input_ids: Initial sequence [batch, seq_len]
        n: Number of draft tokens to generate per iteration
        past_key_values: KV cache for target model
        ref_past_key_values: KV cache for reference model
        nonce: Optional public nonce
        **kwargs: Additional arguments (e.g., process_logits_kwargs)
    
    Yields:
        (output_ids, output_logprobs) tuples for each iteration
    """
    model.eval()
    ref_model.eval()

    debug_iter = 0
    while True:
        # Step 1: Generate draft tokens using reference model with seeded exponential sampling
        ref_output_ids, ref_logprobs, exp_noises, ref_context_codes, ref_past_key_values, _got_eos = gen_n_token_ersd_wm(
            reweight,
            cc_extractor,
            cch,
            private_key,
            ref_model,
            input_ids,
            n,
            past_key_values=ref_past_key_values,
            max_vocab_size=model.config.vocab_size,
            nonce=nonce,
            **kwargs,
        )
        
        # Step 2: Verify draft tokens using target model
        output_ids, output_logprobs, poverlaps, context_codes, time_steps, tags, watermark_code, skipped, past_key_values, got_eos = gen_ersd_wm(
            reweight,
            cc_extractor,
            cch,
            private_key,
            model,
            input_ids,
            ref_output_ids,
            ref_logprobs,
            exp_noises,
            ref_context_codes,
            past_key_values=past_key_values,
            nonce=nonce,
            **kwargs,
            debug_iter=debug_iter,
        )
        
        # Step 3: Fix reference model cache
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        
        # Yield results with ERSD watermark metadata
        yield output_ids, output_logprobs, {
            "context_codes": context_codes,
            "time_steps": time_steps,
            "tags": tags,
            "skipped": skipped,
        }
        
        # Update input_ids for next iteration
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        
        # Check for EOS
        if got_eos:
            break
        debug_iter += 1


def detect_ersd_wm(
    vocab_size: int,
    cc_extractor: uwm.AbstractContextCodeExtractor,
    cch: uwm.lm.ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
    context_codes: np.ndarray = None,
    time_steps: np.ndarray = None,
    tags: np.ndarray = None,
    nonce: bytes = None,
):
    """
    Detect ERSD watermark using Aaronson test score.
    
    This function computes the Aaronson test score for ERSD watermark:
        TestScore(y_{1:n}) = sum_{t=m+1}^n -log(1 - r_t(y_t))
    
    Args:
        vocab_size: Vocabulary size
        cc_extractor: Context code extractor
        cch: Context code history
        private_key: Secret key for PRF
        out_ids: (batch_size, seq_len) output token IDs
        in_ids: (batch_size, in_seq_len) input token IDs (optional)
        context_codes: (batch_size, seq_len) context codes for each position (optional, will be computed if None)
        time_steps: (batch_size, seq_len) time steps for each position (optional, will be computed if None)
        tags: (batch_size, seq_len) tags ("race" or "cont") for each position (optional, will be computed if None)
        nonce: Optional public nonce
    
    Returns:
        score: ERSD_Aaronson_Score object
    """
    from unbiased_watermark.scores.ersd_aaronson import ERSD_Aaronson_Score
    
    batch_shape = out_ids.shape[:-1]
    seq_len = out_ids.shape[-1]
    
    # If context_codes, time_steps, or tags are not provided, we need to compute them
    # This requires knowing the ERSD generation history, which is typically stored
    # during generation. For now, we assume they are provided.
    if context_codes is None or time_steps is None or tags is None:
        raise ValueError(
            "context_codes, time_steps, and tags must be provided for ERSD watermark detection. "
            "These should be stored during generation."
        )
    
    # Create dummy watermark_code (not used by ERSD_Aaronson_Score)
    dummy_code = None
    
    # Compute skipped flags (for now, assume no skipped tokens)
    # In practice, skipped tokens should be tracked during generation
    skipped = np.zeros((batch_shape + (seq_len,)), dtype=bool)
    
    # Create score
    score = ERSD_Aaronson_Score.from_watermarkcode(
        dummy_code,
        out_ids,
        skipped=skipped,
        context_codes=context_codes,
        time_steps=time_steps,
        tags=tags,
        private_key=private_key,
        vocab_size=vocab_size,
        nonce=nonce,
    )
    
    return score
