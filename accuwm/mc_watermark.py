from .utils import *
from .basic_watermark import gen_n_token_uwm
from .mc import gen_mc


@torch.no_grad()
#  @profile_each_line
def gen_mc_last_uwm(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    ref_context_code: np.ndarray,
    ref_watermark_code: uwm.AbstractWatermarkCode,
    ref_skipped: np.ndarray,
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_logprobs: FloatTensor,
    only_last: bool,
    past_key_values=None,
    process_logits_kwargs={},
    mc_synthid: bool = False,
    mc_private_key: bytes = None,
    psedo_r: bool = False,
    temperature: float = 1.0,
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, bool]:
    """
    reweight:
    cc_extractor:
    cch: (batch_size, )
    private_key:
    ref_context_code: (batch_size, n)
    ref_watermark_code: (batch_size, n)
    ref_skipped: (batch_size, n)
    model: Decoder-only model
    input_ids: (batch_size, seq_len). batch_size must be 1
    ref_output_ids: (batch_size, n)
    ref_logprobs: (batch_size, n, vocab_size)
    only_last: bool.  If True, only reweight the last token. If False, reweight all tokens.
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (context_code, watermark_code, skipped, output_ids, output_logprobs, watermark_logprobs, poverlaps, past_key_values, got_eos)
    context_code: (batch_size, gen_len)
    watermark_code: (batch_size, gen_len)
    skipped: (batch_size, gen_len)
    output_ids: (batch_size, gen_len)
    output_logprobs: (batch_size, gen_len, vocab_size)
    watermark_logprobs: (batch_size, gen_len, vocab_size)
    poverlaps: (batch_size, min(gen_len,n))
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    """
    assert input_ids.shape[0] == 1
    assert ref_output_ids.shape == ref_logprobs.shape[:-1]
    #  get ground truth logprobs
    if past_key_values is not None:
        cached_n = cache_len(past_key_values)
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
        past_key_values = create_empty_cache()
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)    # _ids = [input_ids, ref_output_ids]
    # _ids: (batch_size, seq_len+n)
    output = model(input_tokens, past_key_values=past_key_values, use_cache=True)
    #  logits = output.logits.clone()
    logits = output.logits
    logits = logits[:, -ref_output_ids.shape[1] - 1 :, :]
    if process_logits_kwargs != {}:
        #  for i in range(input_tokens.shape[1]):
        #      logits[:, i, :] = process_logits(
        #          _ids[:, : _ids.shape[1] - input_tokens.shape[1] + i + 1],
        #          logits[:, i, :],
        #          **process_logits_kwargs,
        #      )
        for i in range(-1, ref_output_ids.shape[1]):
            logits[:, i + 1, :] = process_logits(
                _ids[:, : _ids.shape[1] - ref_output_ids.shape[1] + i + 1],
                logits[:, i + 1, :],
                **process_logits_kwargs,
            )
    if ref_logprobs.shape[-1] != logits.shape[-1]:
        min_vocab = min(ref_logprobs.shape[-1], logits.shape[-1])
        if hasattr(ref_watermark_code, "g"):
            min_vocab = min(min_vocab, ref_watermark_code.g.shape[-1])
        elif hasattr(ref_watermark_code, "shuffle"):
            min_vocab = min(min_vocab, ref_watermark_code.shuffle.shape[-1])
        elif hasattr(ref_watermark_code, "binary_matrix"):
            min_vocab = min(min_vocab, ref_watermark_code.binary_matrix.shape[-2])
        logits = logits[..., :min_vocab]
        ref_logprobs = ref_logprobs[..., :min_vocab]
        if hasattr(ref_watermark_code, "g") and ref_watermark_code.g.shape[-1] != min_vocab:
            ref_watermark_code = ref_watermark_code.__class__(
                ref_watermark_code.g[..., :min_vocab]
            )
        elif hasattr(ref_watermark_code, "shuffle") and ref_watermark_code.shuffle.shape[-1] != min_vocab:
            ref_watermark_code = ref_watermark_code.__class__(
                ref_watermark_code.shuffle[..., :min_vocab]
            )
        elif hasattr(ref_watermark_code, "binary_matrix") and ref_watermark_code.binary_matrix.shape[-2] != min_vocab:
            ref_watermark_code = ref_watermark_code.__class__(
                ref_watermark_code.binary_matrix[..., :min_vocab, :]
            )
    # logits: (batch_size, n+1, vocab_size), get the logits of the target model
    if not only_last:
        q_logits = reweight.reweight_logits(ref_watermark_code, logits[:, :-1, :])
        # if the context is repeated and detected in draft, then it's also repeated in the target, so we skip the watermark
        pytorch_ref_skipped = torch.tensor(
            ref_skipped, dtype=torch.bool, device=q_logits.device
        )
        target_logits = torch.where(
            pytorch_ref_skipped.unsqueeze(-1),
            logits[:, :-1, :],
            q_logits,
        )
    else:
        target_logits = logits[:, :-1, :]
    # target_logprob: (batch_size, n, vocab_size)
    if not mc_synthid:
        gen_tokens, watermarked_logprobs, poverlaps, fully_coupled = mc_sample(
            target_logits[0, :, :],  # shape (n, vocab_size)
            ref_logprobs[0],
            ref_output_ids[0],
        )
    else:
        assert mc_private_key is not None, "mc_private_key must be provided when mc_synthid is True"
        assert only_last, "we are implementing the fast watermark, so only_last must be True"
        reweight_mc = reweight.__class__()
        gen_tokens, watermarked_logprobs, poverlaps, fully_coupled = mc_sample_synthid(
            target_logits[0, :, :],  # shape (n, vocab_size)
            ref_logprobs[0],
            ref_output_ids[0],
            input_ids[0],
            cc_extractor,
            mc_private_key,
            reweight_mc,
            temperature,
            psedo_r=psedo_r,
            residual_private_key=private_key,
        )
    # gen_tokens: (min(gen_len,n))
    # watermarked_logprobs: (min(gen_len,n), vocab_size)
    # poverlaps: (min(gen_len,n))
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:       # if all tokens from draft are accepted, then generate an additional token from the watermarked target model
        (
            last_watermarked_logits,
            _last_q_logits,
            last_cc,
            last_watermark_code,
            last_skipped,
        ) = step_watermark(
            reweight, logits[:, -1, :], _ids, cc_extractor, cch, private_key, temperature
        )
        # last_watermarked_logits: (batch_size, vocab_size)
        # last_cc: (batch_size, )
        # last_watermark_code: (batch_size, vocab_size, d) for synthid and (batch_size, vocab_size) for gumbel
        # last_skipped: (batch_size, )
        cc = np.concatenate([ref_context_code, last_cc[:, None]], axis=1)
        # for gumbel, it concat[(batch_size, n, vocab_size), (batch_size, 1, vocab_size), dim=-2]
        # for synthid, it concat[(batch_size, n, vocab_size, d), (batch_size, 1, vocab_size, d), dim=-3]
        # print(ref_watermark_code.g.shape, last_watermark_code.g.shape)
        # here we need to use unsqueeze(1) in order to consider synthid's watermark code
        watermark_code = ref_watermark_code.concat(
            [
                ref_watermark_code,
                last_watermark_code.tensor_shape_map(lambda x: x.unsqueeze(1)),
            ],
            dim=-1,
        )
        skipped = np.concatenate([ref_skipped, last_skipped[:, None]], axis=1)
        new_token, last_watermarked_logprobs = basic_sample(last_watermarked_logits)
        # last_watermarked_logprobs: (batch_size, vocab_size)
        if last_watermarked_logprobs.shape[-1] != watermarked_logprobs.shape[-1]:
            last_watermarked_logprobs = last_watermarked_logprobs[
                ..., : watermarked_logprobs.shape[-1]
            ]
        watermarked_logprobs = torch.cat(
            [
                watermarked_logprobs.unsqueeze(0),
                last_watermarked_logprobs.unsqueeze(1),
            ],
            dim=1,
        )
        # watermarked_logprobs: (batch_size, n+1, vocab_size)
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
        cch.rollback(ref_output_ids.shape[1] - gen_len)
        cc = ref_context_code[:, :gen_len]
        watermark_code = ref_watermark_code.tensor_shape_map(lambda x: x[:, :gen_len])
        skipped = ref_skipped[:, :gen_len]
    poverlaps = poverlaps.unsqueeze(0)
    # poverlaps: (1, gen_len)

    # fix past_key_values
    past_key_values = truncate_cache(
        output.past_key_values, input_ids.shape[1] + output_ids.shape[1] - 1
    )

    return (
        cc,
        watermark_code,
        skipped,
        output_ids,
        output_logprobs,
        watermarked_logprobs,
        poverlaps,
        past_key_values,
        got_eos,
    )


#  @profile_each_line
def mc_uwm_sample_generator(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    reweight_in_mc: bool,
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    mc_synthid: bool = False,
    mc_private_key: bytes = None,
    psedo_r: bool = False,
    temperature: float = 1.0,
    **kwargs,
):
    model.eval()
    ref_model.eval()
    while True:
        (
            ref_context_code,
            ref_watermark_code,
            ref_skipped,
            ref_output_ids,
            ref_logprobs,
            ref_watermark_logprobs,
            ref_past_key_values,
            _got_eos,
        ) = gen_n_token_uwm(
            reweight,
            cc_extractor,
            cch,
            private_key,
            ref_model,
            input_ids,
            n,
            temperature,
            past_key_values=ref_past_key_values,
            **kwargs,
        )
        if reweight_in_mc:
            (
                cc,
                watermark_code,
                skipped,
                output_ids,
                output_logprobs,
                watermark_logprobs,
                poverlaps,
                past_key_values,
                got_eos,
            ) = gen_mc_last_uwm(
                reweight,
                cc_extractor,
                cch,
                private_key,
                ref_context_code,
                ref_watermark_code,
                ref_skipped,
                model,
                input_ids,
                ref_output_ids,
                ref_watermark_logprobs,  # not ref_logprobs
                only_last=False,
                past_key_values=past_key_values,
                temperature=temperature,
                **kwargs,
            )
        else:   # fast watermark, we have 2 versions
            (
                cc,
                watermark_code,
                skipped,
                output_ids,
                output_logprobs,
                watermark_logprobs,
                poverlaps,
                past_key_values,
                got_eos,
            ) = gen_mc_last_uwm(
                reweight,
                cc_extractor,
                cch,
                private_key,
                ref_context_code,
                ref_watermark_code,
                ref_skipped,
                model,
                input_ids,
                ref_output_ids,
                ref_logprobs,  # not ref_watermark_logprobs
                only_last=True,
                past_key_values=past_key_values,
                mc_synthid=mc_synthid,  # if use synthid's fast watermark.
                mc_private_key=mc_private_key,
                psedo_r=psedo_r,
                temperature=temperature,
                **kwargs,
            )
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        yield output_ids, output_logprobs
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break
