"""
Aaronson-style watermark scores for the shared-PFR implementation.

The side-information API scores tokens from stored source labels.  The
sequence API implements the detector in pfr_algorithm.md: it receives only the
prompt+output token sequence, recomputes Lambda(w_{1:t-1}) at each output
position, rebuilds PFRSource(label, key), and applies the Gamma(M, 1) exact
null tail probability.
"""

import numpy as np
import torch
import math
from torch import LongTensor

from .additivescore import TokenwiseAdditiveScore
from .. import DeltaGumbel_WatermarkCode
from ..lm import get_rng


def _log_gamma_tail_p_value(shape: int, score: float) -> float:
    if shape <= 0:
        return 0.0
    if score <= 0:
        return 0.0
    terms = np.array(
        [k * np.log(score) - math.lgamma(k + 1) for k in range(shape)],
        dtype=np.float64,
    )
    max_term = float(terms.max())
    return float(-score + max_term + np.log(np.exp(terms - max_term).sum()))


def _log_e_minus1_over_lambda(lam: float) -> float:
    if lam == 0:
        return 0.0
    if lam > 50:
        return lam - np.log(lam)
    return float(np.log(np.expm1(lam)) - np.log(lam))


def _u_chernoff_derivative(lam: float) -> float:
    if lam == 0:
        return 0.5
    if lam > 50:
        return 1.0 - 1.0 / lam
    return float(np.exp(lam) / np.expm1(lam) - 1.0 / lam)


def _solve_u_chernoff_lambda(mean_u: float) -> float:
    if not 0.5 < mean_u < 1.0:
        return 0.0
    lo = 0.0
    hi = 2.0
    while _u_chernoff_derivative(hi) < mean_u:
        hi *= 2.0
        if hi > 1e6:
            return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _u_chernoff_derivative(mid) < mean_u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class PFR_Aaronson_Score(TokenwiseAdditiveScore):
    """Aaronson-style ``-log(1 - U_t)`` score.

    Two equivalent code paths are exposed:

    1. Aligned path — pass a ``DeltaGumbel_WatermarkCode`` as ``code``. The
       per-token uniform is recovered from ``g`` via ``U_t = exp(-exp(-g_t))``
       and the score is computed in a single GPU op. This is the path used by
       ``unbiased_watermark.lm.detect`` so PFR detection can run through the
       same pipeline as ``DeltaGumbel_U``.
    2. Legacy path — pass ``code=None`` together with ``source_labels``,
       ``private_key`` and ``vocab_size``. The uniforms are reconstructed per
       token from ``(label, key)`` to match PFR generation that tracks labels
       directly. Both paths now consume the same ``rng.random((V,),
       dtype=float32)`` so they are bit-identical.
    """

    watermark_code_type = DeltaGumbel_WatermarkCode

    def get_per_token_log_MGF(self, t: float) -> float:
        if t >= 1:
            return np.inf
        return -np.log(1 - t)

    def get_per_token_mu(self) -> float:
        return 1.0

    @classmethod
    def score_from_watermarkcode(
        cls,
        code,
        ids: LongTensor,
        p_logits=None,
        q_logits=None,
        source_labels: np.ndarray = None,  # (batch, seq_len), dtype=object
        time_steps: np.ndarray = None,  # kept for compatibility, unused
        private_key: bytes = None,
        vocab_size: int = None,
    ):
        if code is not None and isinstance(code, DeltaGumbel_WatermarkCode):
            return cls._score_from_deltagumbel_code(code, ids)
        return cls._score_from_source_labels(ids, source_labels, private_key, vocab_size)

    @staticmethod
    def _score_from_deltagumbel_code(
        code: DeltaGumbel_WatermarkCode, ids: LongTensor
    ) -> torch.Tensor:
        assert code.g.shape[:-1] == ids.shape
        gs = torch.gather(code.g, -1, ids.unsqueeze(-1)).squeeze(-1)
        # U_t = exp(-exp(-g_t)), the same uniform that fed get_gumbel_variables.
        Us = torch.exp(-torch.exp(-gs)).clamp(min=1e-10, max=1 - 1e-10)
        return -torch.log(1.0 - Us)

    @staticmethod
    def _score_from_source_labels(
        ids: LongTensor,
        source_labels: np.ndarray,
        private_key: bytes,
        vocab_size: int,
    ) -> torch.Tensor:
        assert source_labels is not None, "source_labels required for legacy PFR_Aaronson scoring"
        assert private_key is not None, "private_key required for legacy PFR_Aaronson scoring"
        assert vocab_size is not None, "vocab_size required for legacy PFR_Aaronson scoring"

        batch_size, seq_len = ids.shape
        device = ids.device
        scores = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)

        for b in range(batch_size):
            for t in range(seq_len):
                source_label = source_labels[b, t]
                token_id = ids[b, t].item()
                r_t_y_t = _uniform_for_token(
                    source_label, private_key, token_id, vocab_size, device=device,
                )
                scores[b, t] = -np.log(1 - r_t_y_t)
        return scores

    @classmethod
    def from_watermarkcode(
        cls,
        code,
        ids: LongTensor,
        skipped: np.ndarray,
        p_logits=None,
        q_logits=None,
        source_labels: np.ndarray = None,
        time_steps: np.ndarray = None,
        private_key: bytes = None,
        vocab_size: int = None,
    ):
        scores = cls.score_from_watermarkcode(
            code,
            ids,
            p_logits,
            q_logits,
            source_labels=source_labels,
            time_steps=time_steps,
            private_key=private_key,
            vocab_size=vocab_size,
        )
        return cls(scores=scores.detach().cpu().numpy(), skipped=skipped)


class PFR_Aaronson_Gamma_Score(PFR_Aaronson_Score):
    def get_log_p_value(self) -> float:
        return _log_gamma_tail_p_value(self.get_num_added(), self.get_score())


class PFR_Aaronson_U_Score(PFR_Aaronson_Score):
    def get_log_p_value(self) -> float:
        num_added = self.get_num_added()
        if num_added == 0:
            return 0.0
        scores = self.scores * (~self.skipped)
        u_vals = 1.0 - np.exp(-scores)
        mean_u = u_vals.sum() / num_added
        if not np.isfinite(mean_u):
            return np.nan
        if mean_u <= 0.5:
            return 0.0
        lam = _solve_u_chernoff_lambda(float(mean_u))
        log_term = _log_e_minus1_over_lambda(lam)
        return num_added * (log_term - lam * mean_u)


def _seed_from_label_key(label: bytes, private_key: bytes) -> int:
    """SHA-256 over (label, private_key) reduced to a 32-bit seed.  Must match
    accuwm.pfr._get_seed exactly so generation and detection share an RNG seed."""
    import hashlib
    m = hashlib.sha256()
    m.update(label)
    m.update(private_key)
    return int.from_bytes(m.digest(), "big") % (2**32 - 1)


def _uniform_for_token(
    label: bytes,
    private_key: bytes,
    token_id: int,
    vocab_size: int,
    *,
    device=None,
) -> float:
    """Recover the per-token uniform u_t used by PFR generation for the chosen
    token.  Generation calls torch.rand((1, vocab_size), device=cuda, dtype=float32)
    with a torch.Generator seeded by SHA256(label || private_key); detection
    must mirror this exactly (same RNG family, same device, same shape, same
    dtype) or the recovered uniforms are uncorrelated with the watermark."""
    seed = _seed_from_label_key(label, private_key)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    r_values = torch.rand((1, vocab_size), device=device, dtype=torch.float32, generator=g)
    return float(torch.clamp(r_values[0, int(token_id)], 1e-10, 1 - 1e-10).item())


def compute_pfr_aaronson_from_sequence(
    full_ids: LongTensor,
    prompt_length: int,
    labeler,
    private_key: bytes,
    vocab_size: int,
) -> dict:
    """
    Compute the detector from pfr_algorithm.md.

    Args:
        full_ids: Prompt and output tokens, shape (1, N).
        prompt_length: Number of prompt tokens n0. Tokens at positions
            prompt_length..N-1 are scored.
        labeler: Stateful Lambda implementation. It is called on each prefix
            c_t = w_{1:t-1}.
        private_key: Secret key k.
        vocab_size: Vocabulary size for the discrete counting-measure case.

    Returns:
        dict with Aaronson score sum, token count, exact Gamma tail log p-value,
        and ANLPPT.
    """
    assert full_ids.shape[0] == 1, "only batch_size=1 is supported"
    assert 0 <= prompt_length <= full_ids.shape[1]

    scores = []
    masked = []
    for pos in range(prompt_length, full_ids.shape[1]):
        context = full_ids[:, :pos]
        token_id = int(full_ids[0, pos].item())
        label_info = labeler.label_info(context)
        r_t = _uniform_for_token(
            label_info.source_label,
            private_key,
            token_id,
            vocab_size,
        )
        scores.append(-np.log(1.0 - r_t))
        masked.append(label_info.masked)

    score_sum = float(np.sum(scores)) if scores else 0.0
    m = len(scores)
    log_p_value = _log_gamma_tail_p_value(m, score_sum)
    anlppt = float(-log_p_value / m) if m else 0.0
    return {
        "score_sum": score_sum,
        "num_scored": m,
        "log_p_value": float(log_p_value),
        "ANLPPT_Aaronson": anlppt,
        "score_mean": float(score_sum / m) if m else 0.0,
        "masked_ratio": float(np.mean(masked)) if masked else 0.0,
    }
