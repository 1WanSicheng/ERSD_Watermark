from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import LongTensor, FloatTensor, BoolTensor
import torch.nn.functional as F

from . import AbstractWatermarkCode, AbstractReweight
from .scores import AbstractScore
from .synthid import SynthID_Reweight_fast

class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: LongTensor) -> np.ndarray:
        """
        Should return a context code `c` which will be used to initialize a torch.Generator.
        :param context: (..., seq_len)
        :return: (..., ), np.ndarray, dtype=obj or any
        """
        pass


def peel_ndarray(a, last_n_dim=1):
    ra = np.empty(a.shape[:-last_n_dim], dtype=object)
    for index in np.ndindex(ra.shape):
        ra[index] = a[index].copy()
    return ra


@dataclass(frozen=True)
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    """Extracts the last n tokens in the context"""

    n: int

    def extract(self, context: LongTensor) -> np.ndarray:
        c = context[..., -self.n :].detach().cpu().numpy()
        c = peel_ndarray(c, last_n_dim=1)
        c = np.vectorize(lambda x: x.tobytes())(c)
        return c, context[..., -self.n :]


class ContextCodeHistory:
    def __init__(self, data: np.ndarray = None, batch_shape=()):
        """
        data: (..., ), np.ndarray, dtype=obj, each element is a list
        """
        if data is None:
            data = np.empty(batch_shape, dtype=object)
            data.fill([])   # shape array([[], []]) if batch_shape = (2,)
        self.data = data    

    def get_flattened(self) -> set:
        """
        :return: set of context code
        """
        return set(
            cc
            for cch_list in np.nditer(self.data, flags=["refs_ok"])
            for cc in cch_list.item()
        )

    def step(
        self, cc_extractor: AbstractContextCodeExtractor, context: LongTensor, update_cch: bool = True, raw_context_output: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        :param context: (..., seq_len)
        :param update_cch: bool, whether to update the context code history
        :return: context_code, skipped
        context_code: (..., ), np.ndarray, dtype=obj
        skipped: (..., ), torch.bool
        raw_context: (..., seq_len), torch.LongTensor
        """
        fcch = self.get_flattened()     # inital an empty list
        cc, raw_context = cc_extractor.extract(context)  # extract the latest n tokens
        skipped = np.zeros(context.shape[:-1], dtype=bool)  # inital a zero array
        for cc_item, skipped_item in np.nditer(
            [cc, skipped], op_flags=[["readwrite"], ["writeonly"]]
        ):
            cc_item = cc_item.item()
            if cc_item not in fcch:
                fcch.add(cc_item)
            else:
                skipped_item[()] = True
        if update_cch:
            self.add_context_code(cc)   # add the latest n tokens to the context code history
        
        if raw_context_output:
            return cc, skipped, raw_context
        else:
            return cc, skipped

    def add_context_code(self, context_code: np.ndarray):
        """
        :param context_code: (..., ), np.ndarray, dtype=obj
        """
        assert context_code.shape == self.data.shape
        for index in np.ndindex(self.data.shape):
            self.data[index].append(context_code[index])

    def rollback(self, n: int):
        assert n >= 0
        if n == 0:
            return
        for index in np.ndindex(self.data.shape):
            self.data[index] = self.data[index][:-n]


def get_rng(*bs: bytes) -> np.random.Generator:
    import hashlib

    m = hashlib.sha256()
    for b in bs:
        m.update(b)
    full_hash = m.digest()
    seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
    return np.random.default_rng(seed)


def step_watermark(
    reweight: AbstractReweight,
    p_logits: FloatTensor,
    input_ids: LongTensor,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    temperature: float,
) -> tuple[FloatTensor, np.ndarray, AbstractWatermarkCode, np.ndarray]:
    """
    :param p_logits: (..., vocab_size)
    :param input_ids: (..., seq_len)
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory, will be updated
    :param temperature: float
    :return: log_q, context_code, watermark_code, skipped
    log_q: (..., vocab_size)
    context_code: (..., ), np.ndarray, dtype=obj
    watermark_code: AbstractWatermarkCode, shape: (..., )
    skipped: (..., )
    """
    p_logits = p_logits / temperature   # temperature is used to scale the logits
    cc, skipped = cch.step(cc_extractor, input_ids)  # step the watermark code history
    rng = np.empty(cc.shape, dtype=object)
    for index in np.ndindex(rng.shape):  # get the random number generator based on each context code for each batch
        rng[index] = get_rng(cc[index], private_key)
    watermark_code_type = reweight.watermark_code_type
    watermark_code = reweight.watermark_code_type.from_random(rng, p_logits.size(-1))
    watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(input_ids.device))
    q_logits = reweight.reweight_logits(watermark_code, p_logits)
    #  for each batch, if skipped then log_p otherwise log_q
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=input_ids.device)
    wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), p_logits, q_logits)
    return wm_logits, q_logits, cc, watermark_code, skipped


def detect_pre(
    vocab_size: int,
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
    p_logits: FloatTensor = None,
) -> tuple[FloatTensor, np.ndarray, AbstractWatermarkCode, np.ndarray]:
    """
    :param reweight: AbstractReweight
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory
    :param out_ids: (..., out_seq_len)
    :param p_logits: (..., out_seq_len, vocab_size)
    :param in_ids: (..., in_seq_len)
    :return: log_q, context_code, watermark_code, skipped
    log_q: (..., out_seq_len, vocab_size)
    context_code: (..., out_seq_len), np.ndarray, dtype=obj
    watermark_code: AbstractWatermarkCode, shape: (..., out_seq_len)
    skipped: (..., out_seq_len)
    """
    batch_shape = out_ids.shape[:-1]
    assert cch.data.shape == batch_shape
    if in_ids is not None:
        assert in_ids.shape[:-1] == batch_shape
    if p_logits is not None:
        assert p_logits.shape[:-2] == batch_shape
        vocab_size = p_logits.shape[-1]

    ids = out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)
    cc_s, skipped_s = [], []
    for i in range(ids.shape[-1] - out_ids.shape[-1], ids.shape[-1]):
        cc, skipped = cch.step(cc_extractor, ids[..., :i])
        cc_s.append(cc)
        skipped_s.append(skipped)
    cc = np.stack(cc_s, axis=-1)    # shape (..., out_seq_len)
    skipped = np.stack(skipped_s, axis=-1)
    rng = np.empty(cc.shape, dtype=object)
    for index in np.ndindex(rng.shape):
        rng[index] = get_rng(cc[index], private_key)
    watermark_code_type = reweight.watermark_code_type
    watermark_code = reweight.watermark_code_type.from_random(rng, vocab_size)   # shape (..., out_seq_len, vocab_size, 30)
    watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(out_ids.device))
    if p_logits is not None:
        q_logits = reweight.reweight_logits(watermark_code, p_logits)
    else:
        q_logits = None
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=out_ids.device)
    if p_logits is not None and p_logits.shape[-2] == out_ids.shape[-1]:
        wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), p_logits, q_logits)
    else:
        wm_logits = None
    return wm_logits, q_logits, cc, watermark_code, skipped


def detect(
    vocab_size: int,
    score_type: type[AbstractScore],
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
    p_logits: FloatTensor = None,
) -> AbstractScore:
    wm_logits, q_logits, cc, watermark_code, skipped = detect_pre(
        vocab_size, reweight, cc_extractor, cch, private_key, out_ids, in_ids, p_logits
    )
    score = score_type.from_watermarkcode(
        watermark_code,
        out_ids,
        skipped=skipped,
        p_logits=p_logits,
        q_logits=q_logits,
    )
    return score


def get_r_values(
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes | int,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
) -> np.ndarray:
    """
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory
    :param private_key: bytes
    :param out_ids: LongTensor
    :param in_ids: LongTensor
    :return: np.ndarray, the acceptance threshold for each token, shape (..., out_seq_len)
    """
    batch_shape = out_ids.shape[:-1]
    assert cch.data.shape == batch_shape
    if in_ids is not None:
        assert in_ids.shape[:-1] == batch_shape
    ids = out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)
    cc_s = []
    for i in range(ids.shape[-1] - out_ids.shape[-1], ids.shape[-1]):
        cc, _ = cch.step(cc_extractor, ids[..., :i])
        cc_s.append(cc)
    cc = np.stack(cc_s, axis=-1)     # shape (..., out_seq_len)
    rng = np.empty(cc.shape, dtype=object)
    private_key = bytes(private_key) if isinstance(private_key, int) else private_key
    for index in np.ndindex(rng.shape):
        rng[index] = get_rng(cc[index], private_key)
    r_values = np.empty(cc.shape, dtype=np.float32)
    for index in np.ndindex(r_values.shape):
        r_values[index] = rng[index].random()
    return r_values  # shape (..., out_seq_len)


def step_watermark_synthid(
    reweight: SynthID_Reweight_fast,
    p_logits: FloatTensor,
    input_ids: LongTensor,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    temperature: float,
    top_k: int,
    apply_top_k: bool = True,
) -> tuple[FloatTensor, FloatTensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param p_logits: (..., vocab_size)
    :param input_ids: (..., seq_len)
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory, will be updated
    :param temperature: float
    :param top_k: int
    :param apply_top_k: bool, whether to apply top_k
    :return: log_q, context_code, g_values, skipped, top_k_indices
    log_q: (..., vocab_size)
    context_code: (..., ), np.ndarray, dtype=obj
    g_values: (..., top_k, depth)
    skipped: (..., )
    top_k_indices: (..., top_k)
    """
    p_logits_processed = p_logits / temperature
    top_k_result = torch.topk(p_logits_processed, k=top_k, dim=-1)
    if apply_top_k:
        scores_top_k = top_k_result.values   # shape (batch_size, top_k)
        top_k_indices = top_k_result.indices  # shape (batch_size, top_k)
    else:
        scores_top_k = p_logits_processed
        top_k_indices = torch.stack([
            torch.arange(p_logits.size(-1), device=p_logits.device)
            for _ in range(p_logits.size(0))
        ])
    cc, skipped, raw_context = cch.step(cc_extractor, input_ids, raw_context_output=True)  
    # step the watermark code history, cc: (batch_size,) , skipped: (batch_size,), raw_context: (batch_size, n_grams)
    # compute the ngram keys based on the raw_context, and top_k_indices (keep the original token_ids)
    ngram_keys = reweight._compute_keys(raw_context, top_k_indices) # shape (batch_size, top_k, depth)
    g_values_all = reweight.sample_g_values(ngram_keys)  # shape (batch_size, top_k, depth)
    q_logits = reweight.reweight_logits(g_values_all, scores_top_k)
    #  for each batch, if skipped then scores_top_k otherwise log_q
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=input_ids.device)
    wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), scores_top_k, q_logits)
    return wm_logits, q_logits, cc, g_values_all, skipped, top_k_indices


def compute_context_repetition_mask(
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
) -> np.ndarray:
    """
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory
    :param out_ids: (..., out_seq_len)
    :param in_ids: (..., in_seq_len)
    :return: skipped, shape (..., out_seq_len)
    """
    ids = out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)
    skipped_s = []
    for i in range(ids.shape[-1] - out_ids.shape[-1], ids.shape[-1]):
        _, skipped = cch.step(cc_extractor, ids[..., :i])
        skipped_s.append(skipped)
    skipped = np.stack(skipped_s, axis=-1)

    return skipped


def step_watermark_synthid_no_update(
    reweight: SynthID_Reweight_fast,
    p_logits: FloatTensor,
    input_ids: LongTensor,
    cc_extractor: AbstractContextCodeExtractor,
    skipped: np.ndarray,
    temperature: float,
    top_k: int,
    apply_top_k: bool = True,
) -> tuple[FloatTensor, FloatTensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param p_logits: (..., vocab_size)
    :param input_ids: (..., seq_len)
    :param cc_extractor: AbstractContextCodeExtractor
    :param skipped: (..., )
    :param temperature: float
    :param top_k: int
    :param apply_top_k: bool, whether to apply top_k
    :return: full_wm_logprob, shape (..., vocab_size)
    """
    p_logits_processed = p_logits / temperature
    top_k_result = torch.topk(p_logits_processed, k=top_k, dim=-1)
    if apply_top_k:
        scores_top_k = top_k_result.values   # shape (batch_size, top_k)
        top_k_indices = top_k_result.indices  # shape (batch_size, top_k)
    else:
        scores_top_k = p_logits_processed
        top_k_indices = torch.stack([
            torch.arange(p_logits.size(-1), device=p_logits.device)
            for _ in range(p_logits.size(0))
        ])
    _, raw_context = cc_extractor.extract(input_ids)    # not update the context code history
    # compute the ngram keys based on the raw_context, and top_k_indices (keep the original token_ids)
    ngram_keys = reweight._compute_keys(raw_context, top_k_indices) # shape (batch_size, top_k, depth)
    g_values_all = reweight.sample_g_values(ngram_keys)  # shape (batch_size, top_k, depth)
    q_logits = reweight.reweight_logits(g_values_all, scores_top_k)
    #  for each batch, if skipped then scores_top_k otherwise log_q
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=input_ids.device)
    wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), scores_top_k, q_logits)
    # re-mapping to dense indices with indices_mapping and set the logits of the rest to near-zero probability
    assert top_k_indices is not None
    full_wm_logprob = torch.full_like(p_logits, -1e4)
    full_wm_logprob.scatter_(1, top_k_indices, wm_logits)

    return full_wm_logprob


def get_combined_logits(
    mse_target_logprob: FloatTensor,
    mws_target_logprob: FloatTensor,
    gamma: float,
) -> FloatTensor:
    """
    :param mse_target_logprob: (..., vocab_size)
    :param mws_target_logprob: (..., vocab_size)
    :param gamma: float
    :return: (..., vocab_size)
    """
    return (1 - gamma) * mse_target_logprob + gamma * mws_target_logprob


def get_mse_target_logprob(
    ref_wm_logprob: FloatTensor,
    ref_logprobs: FloatTensor,
    target_logprobs: FloatTensor,
) -> FloatTensor:
    """
    Implements the formula for q' in the provided image, returning log-probabilities.
    All inputs are log-probabilities of shape (..., vocab_size).
    ref_wm_logprob: (..., vocab_size)
    ref_logprobs: (..., vocab_size)
    target_logprobs: (..., vocab_size)
    :return: (..., vocab_size)
    """
    # Convert logits to log-probs
    ref_wm_logprob = F.log_softmax(ref_wm_logprob, dim=-1)
    ref_logprobs = F.log_softmax(ref_logprobs, dim=-1)
    target_logprobs = F.log_softmax(target_logprobs, dim=-1)
    # Convert log-probs to probs
    p_wm = torch.exp(ref_wm_logprob)  # (..., vocab_size)
    p = torch.exp(ref_logprobs)       # (..., vocab_size)
    q = torch.exp(target_logprobs)    # (..., vocab_size)
    log_prob_ratio = target_logprobs - ref_logprobs
    prob_ratio = torch.exp(log_prob_ratio)

    # Acceptance ratio: min(1, q/p)
    accept_ratio = torch.minimum(torch.ones_like(q), prob_ratio)

    # First term: p_wm * min(1, q/p)
    first_term = p_wm * accept_ratio

    # Normalization sum over vocab
    norm_sum = (p_wm * accept_ratio).sum(dim=-1, keepdim=True)

    # (q - p)_+ = max(q - p, 0)
    q_minus_p_pos = torch.clamp(q - p, min=0.0)
    q_minus_p_pos = q_minus_p_pos / q_minus_p_pos.sum(dim=-1, keepdim=True)

    # Second term: (1 - norm_sum) * (q - p)_+
    second_term = (1 - norm_sum) * q_minus_p_pos

    # Final result: sum of both terms
    result = first_term + second_term

    log_probs = torch.log(result)
    log_probs = torch.where(
        torch.isfinite(log_probs), log_probs, torch.tensor(-1e4, device=log_probs.device)
    )
    return log_probs
