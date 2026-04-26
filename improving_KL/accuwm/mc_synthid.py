from .utils import *
from .basic_synthid import gen_n_token_synthid
from unbiased_watermark.synthid import SynthID_Reweight_fast



@torch.no_grad()
#  @profile_each_line
def gen_mc_last_synthid(
    method: str,
    reweight: SynthID_Reweight_fast,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    ref_context_code: np.ndarray,
    ref_skipped: np.ndarray,
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_og_logprobs: FloatTensor,
    ref_wm_logprobs: FloatTensor,
    past_key_values=None,
    mc_private_key: int = None,
    psedo_r: bool = True,
    temperature: float = 1.0,
    top_k: int = 100,
    seed: int = 0,
    process_logits_kwargs={},
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, any, any, bool, list]:
    """
    reweight:
    cc_extractor:
    cch: (batch_size, )
    ref_context_code: (batch_size, n)
    ref_skipped: (batch_size, n)
    model: Decoder-only model
    input_ids: (batch_size, seq_len). batch_size must be 1
    ref_output_ids: (batch_size, n)
    ref_logprobs: (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    temperature: float
    top_k: int
    seed: int
    return: (context_code, skipped, output_ids, output_logprobs, poverlaps, past_key_values, got_eos, prob_ratio_list)
    context_code: (batch_size, gen_len)
    skipped: (batch_size, gen_len)
    output_ids: (batch_size, gen_len)
    output_logprobs: (batch_size, gen_len, vocab_size)
    poverlaps: (batch_size, min(gen_len,n))
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    prob_ratio_list: list
    """
    assert input_ids.shape[0] == 1
    if method in ['mc_mse', 'mc_2keys']:
        ref_logprobs = ref_og_logprobs
    elif method in ['mc_mws', 'mc_comb1', 'mc_comb2']:
        ref_logprobs = ref_wm_logprobs
    else:
        raise ValueError(f"Invalid method: {method}")
    assert ref_output_ids.shape == ref_logprobs.shape[:-1]
    #  get ground truth logprobs
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)    # _ids = [input_ids, ref_output_ids]
    # _ids: (batch_size, seq_len+n)
    output = model(input_tokens, past_key_values=past_key_values)
    #  logits = output.logits.clone()
    logits = output.logits
    logits = logits[:, -ref_output_ids.shape[1] - 1 :, :]
    if process_logits_kwargs != {}:
        for i in range(-1, ref_output_ids.shape[1]):
            logits[:, i + 1, :] = process_logits(
                _ids[:, : _ids.shape[1] - ref_output_ids.shape[1] + i + 1],
                logits[:, i + 1, :],
                **process_logits_kwargs,
            )
    # logits: (batch_size, n+1, vocab_size), get the logits of the target model
    if method in ['mc_mse', 'mc_2keys']:
        target_logits = logits[:, :-1, :]
    elif method == 'mc_mws':   # for method maintain watermark strength, we need to use the watermarked logits of the target model
        target_logits = []
        for i in range(ref_output_ids.shape[1]):
            temp_input_ids = torch.cat([input_ids, ref_output_ids[:, :i]], dim=1)
            temp_full_wm_logprob = step_watermark_synthid_no_update(
                reweight, logits[:, i, :], temp_input_ids, cc_extractor, ref_skipped[:, i], temperature, top_k
            )
            # here do not update the context code history
            target_logits.append(temp_full_wm_logprob)
        target_logits = torch.stack(target_logits, dim=1)
    elif method in ['mc_comb1', 'mc_comb2']:
        target_logits = []
        for i in range(ref_output_ids.shape[1]):
            temp_input_ids = torch.cat([input_ids, ref_output_ids[:, :i]], dim=1)
            mws_target_logprob = step_watermark_synthid_no_update(
                reweight, logits[:, i, :], temp_input_ids, cc_extractor, ref_skipped[:, i], temperature, top_k
            )   # logprobs of the target model with watermark
            mse_target_logprob = get_mse_target_logprob(ref_wm_logprobs[:, i, :], ref_og_logprobs[:, i, :], logits[:, i, :])
            if method == 'mc_comb1':
                target_logits.append(get_combined_logits(mse_target_logprob, mws_target_logprob, gamma=0.3))
            elif method == 'mc_comb2':
                target_logits.append(get_combined_logits(mse_target_logprob, mws_target_logprob, gamma=0.7))
        target_logits = torch.stack(target_logits, dim=1)
    else:
        raise ValueError(f"Invalid method: {method}")
    # target_logits: (batch_size, n, vocab_size)
    assert target_logits.shape == ref_logprobs.shape, f"target_logits.shape: {target_logits.shape}, ref_logprobs.shape: {ref_logprobs.shape}"
    if method == 'mc_2keys':
        assert mc_private_key is not None, "mc_private_key must be provided when use mc_2keys"
    gen_tokens, watermarked_logprobs, poverlaps, fully_coupled, prob_ratio_list = mc_sample_synthid_fast(
        method,
        target_logits[0, :, :],  # shape (n, vocab_size)
        ref_logprobs[0],        # shape (n, vocab_size)
        ref_output_ids[0],
        input_ids[0],
        cc_extractor,
        mc_private_key,
        temperature,
        top_k,
        seed,
        psedo_r=psedo_r,
        residual_private_key=reweight.private_key,
        residual_sampling_table_seed=getattr(reweight, "sampling_table_seed", seed),
        residual_watermarking_depth=len(reweight.keys),
    )
    # gen_tokens: (min(gen_len,n))
    # watermarked_logprobs: (min(gen_len,n), vocab_size)
    # poverlaps: (min(gen_len,n))
    # prob_ratio_list: (min(gen_len,n))
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:       # if all tokens from draft are accepted, then generate an additional token from the watermarked target model
        (
            last_watermarked_logits,
            _last_q_logits,
            last_cc,
            last_g_values_all,
            last_skipped,
            last_indices_mapping,
        ) = step_watermark_synthid(
            reweight, logits[:, -1, :], _ids, cc_extractor, cch, temperature, top_k
        )   # here we use synthid's fast watermark
        # last_watermarked_logits: (batch_size, vocab_size)
        # last_cc: (batch_size, )
        # last_skipped: (batch_size, )
        cc = np.concatenate([ref_context_code, last_cc[:, None]], axis=1)
        skipped = np.concatenate([ref_skipped, last_skipped[:, None]], axis=1)
        new_token, last_watermarked_logprobs = basic_sample(last_watermarked_logits)
        # last_watermarked_logprobs: (batch_size, top_k)
        # re-mapping to dense indices with indices_mapping
        assert last_indices_mapping is not None
        new_token = torch.vmap(torch.take, in_dims=0, out_dims=0)(
            last_indices_mapping, new_token
        )
        # new_token: (batch_size=1, 1)
        output_ids = torch.cat([gen_tokens.unsqueeze(0), new_token], dim=-1)
        # output_ids: (1, n+1)
        output_logprobs = F.log_softmax(logits, dim=-1)
        # output_logprobs: (1, n+1, vocab_size)
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
    else:
        gen_len = gen_tokens.shape[0]
        output_ids = gen_tokens.unsqueeze(0)
        # output_ids: (1, gen_len)
        output_logprobs = F.log_softmax(logits[:, :gen_len, :], dim=-1)
        # output_logprobs: (1, gen_len, vocab_size)
        cch.rollback(ref_output_ids.shape[1] - gen_len) # rollback the rejected(unused) context code history
        cc = ref_context_code[:, :gen_len]
        skipped = ref_skipped[:, :gen_len]
    poverlaps = poverlaps.unsqueeze(0)
    # poverlaps: (1, gen_len)

    # fix past_key_values
    past_key_values = None

    return (
        cc,
        skipped,
        output_ids,
        output_logprobs,
        poverlaps,
        past_key_values,
        got_eos,
        prob_ratio_list,
    )


#  @profile_each_line
def mc_synthid_sample_generator(
    method: str,
    reweight: SynthID_Reweight_fast,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    mc_private_key: int = None,
    psedo_r: bool = True,
    temperature: float = 1.0,
    top_k: int = 100,
    seed: int = 0,
    **kwargs,
):
    model.eval()
    ref_model.eval()
    while True:
        (
            ref_context_code,
            ref_g_values,
            ref_skipped,
            ref_output_ids,
            ref_og_logprobs,
            ref_wm_logprobs,
            ref_past_key_values,
            _got_eos,
        ) = gen_n_token_synthid(
            reweight,
            cc_extractor,
            cch,
            ref_model,
            input_ids,
            n,
            temperature,
            top_k,
            past_key_values=ref_past_key_values,
            apply_top_k=True,
            **kwargs,
        )

        (
            cc,
            skipped,
            output_ids,
            output_logprobs,
            poverlaps,
            past_key_values,
            got_eos,
            prob_ratio_list,
        ) = gen_mc_last_synthid(
            method,
            reweight,
            cc_extractor,
            cch,
            ref_context_code,
            ref_skipped,
            model,
            input_ids,
            ref_output_ids,
            ref_og_logprobs,
            ref_wm_logprobs,
            past_key_values=past_key_values,
            mc_private_key=mc_private_key,
            psedo_r=psedo_r,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            **kwargs,
        )


        ref_past_key_values = None
        yield output_ids, prob_ratio_list
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break
