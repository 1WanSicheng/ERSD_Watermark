from .utils import *
from unbiased_watermark.synthid import SynthID_Reweight_fast

@torch.no_grad()
def gen_n_token_synthid(
    reweight: SynthID_Reweight_fast,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    model,
    input_ids: LongTensor,
    n: int,
    temperature: float,
    top_k: int,
    past_key_values=None,
    apply_top_k: bool = True,
    process_logits_kwargs={},
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    LongTensor,
    FloatTensor,
    FloatTensor,
    any,
    bool,
]:
    """
    reweight:
    cc_extractor:
    cch: (batch_size, )
    model: Decoder-only model
    input_ids: (batch_size, seq_len), need to be on the same device and appropriate dtype
    n: number of tokens to generate
    temperature: temperature for sampling
    top_k: top_k for sampling
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (context_code, watermark_code, skipped, output_ids, output_logprobs, watermark_logprobs, past_key_values, got_eos)
    context_code: (batch_size, n)
    skipped: (batch_size, n)
    output_ids: (batch_size, n)
    output_logprobs: (batch_size, n, vocab_size)
    watermark_logprobs: (batch_size, n, vocab_size)
    g_values_list: (n, depth)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    """
    assert cch.data.shape == input_ids.shape[:-1]
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids
    output_ids = []
    output_logprobs = []
    watermarked_logprobs = []
    ccs = []
    g_values_list = []
    skippeds = []
    device = model.device
    got_eos = False
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        logprobs = F.log_softmax(logits, dim=-1)
        wm_logits, _q_logts, cc, g_values_all, skipped, indices_mapping = step_watermark_synthid(
            reweight, logits, input_ids, cc_extractor, cch, temperature, top_k, apply_top_k
        )   # wm_logits: (batch_size, top_k), g_values_all: (batch_size, top_k, depth)
        new_token, wm_logprob = basic_sample(wm_logits)  # new_token: (batch_size, 1)
        # extract the g_values for the new token
        g_values = g_values_all[0, new_token[0][0], :]  # shape (depth)
        assert indices_mapping is not None
        # re-mapping to dense indices with indices_mapping
        new_token = torch.vmap(torch.take, in_dims=0, out_dims=0)(
            indices_mapping, new_token
        )
        # new_token: (batch_size, 1)

        output_logprobs.append(logprobs)
        full_wm_logprob = torch.full_like(logprobs, -1e4)  # shape (batch_size, vocab_size)
        # Scatter the top_k logprobs into the correct positions
        full_wm_logprob.scatter_(1, indices_mapping, wm_logprob)
        watermarked_logprobs.append(full_wm_logprob)
        ccs.append(cc)
        g_values_list.append(g_values.cpu().numpy())
        skippeds.append(skipped)
        input_tokens = new_token
        output_ids.append(new_token)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        past_key_values = output.past_key_values
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break
    output_ids = torch.cat(output_ids, dim=1)  # shape (batch_size, n)
    output_logprobs = torch.stack(output_logprobs, dim=1)  # shape (batch_size, n, vocab_size)
    watermarked_logprobs = torch.stack(watermarked_logprobs, dim=1) # shape (batch_size, n, vocab_size)
    cc = np.stack(ccs, axis=-1)
    skipped = np.stack(skippeds, axis=-1)
    g_values_list = np.array(g_values_list)
    return (
        cc,
        g_values_list,  # shape (n, depth)
        skipped,
        output_ids,
        output_logprobs,
        watermarked_logprobs,
        past_key_values,
        got_eos,
    )


def basic_synthid_generator(
    reweight: SynthID_Reweight_fast,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    model,
    input_ids: LongTensor,
    past_key_values=None,
    temperature=1.0,
    top_k=100,
    n=1,
    apply_top_k=True,
    **kwargs
):
    model.eval()
    while True:
        (
            cc,
            g_values_list,
            skipped,
            output_ids,
            output_logprobs,
            watermark_logprobs,
            past_key_values,
            got_eos,
        ) = gen_n_token_synthid(
            reweight,
            cc_extractor,
            cch,
            model,
            input_ids,
            n,
            temperature,
            top_k,
            apply_top_k=apply_top_k,
            past_key_values=past_key_values,
            **kwargs,
        )
        yield output_ids, g_values_list
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break
