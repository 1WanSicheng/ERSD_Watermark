"""
ERSD Aaronson Score for watermark detection.

This implements the Aaronson watermarking test score for ERSD:
    TestScore(y_{1:n}) = sum_{t=m+1}^n -log(1 - r_t(y_t))

where r_t(y) = F_{(c_{t-1}, t, tag, ν), k}(y) is a keyed pseudorandom function.

For ERSD watermark:
- Draft tokens and verification use tag="race"
- Continuation token uses tag="cont"
"""

import numpy as np
import torch
from torch import FloatTensor, LongTensor

from . import TokenwiseAdditiveScore
from ..lm import get_rng


class ERSD_Aaronson_Score(TokenwiseAdditiveScore):
    """
    Aaronson watermarking test score for ERSD.
    
    This score computes: -log(1 - r_t(y_t)) for each token,
    where r_t(y_t) is regenerated from the PRF using context code, time step, and tag.
    """
    
    @classmethod
    def score_from_watermarkcode(
        cls,
        code,  # Not used, but required by interface
        ids: LongTensor,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
        # ERSD-specific parameters
        context_codes: np.ndarray = None,  # (seq_len,) context codes for each position
        time_steps: np.ndarray = None,  # (seq_len,) time steps for each position
        tags: np.ndarray = None,  # (seq_len,) tags ("race" or "cont") for each position
        private_key: bytes = None,
        vocab_size: int = None,
        nonce: bytes = None,
    ) -> FloatTensor:
        """
        Compute Aaronson test score for ERSD watermark.
        
        Args:
            code: Not used (for interface compatibility)
            ids: (batch_size, seq_len) token IDs
            context_codes: (batch_size, seq_len) context codes (numpy array, dtype=object)
            time_steps: (batch_size, seq_len) time steps (numpy array)
            tags: (batch_size, seq_len) tags ("race" or "cont") (numpy array of strings)
            private_key: Secret key for PRF
            vocab_size: Vocabulary size
            nonce: Optional public nonce
        
        Returns:
            scores: (batch_size, seq_len) test scores for each token
        """
        assert context_codes is not None, "context_codes required for ERSD_Aaronson_Score"
        assert time_steps is not None, "time_steps required for ERSD_Aaronson_Score"
        assert tags is not None, "tags required for ERSD_Aaronson_Score"
        assert private_key is not None, "private_key required for ERSD_Aaronson_Score"
        assert vocab_size is not None, "vocab_size required for ERSD_Aaronson_Score"
        
        batch_size, seq_len = ids.shape
        device = ids.device
        scores = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
        
        # For each position, regenerate r_t(y_t) and compute -log(1 - r_t(y_t))
        for b in range(batch_size):
            for t in range(seq_len):
                cc = context_codes[b, t]
                time_step = int(time_steps[b, t])
                tag = tags[b, t] if isinstance(tags[b, t], str) else tags[b, t].item()
                token_id = ids[b, t].item()
                
                # Regenerate r_t(y_t) using PRF
                # Seed: (c_{t-1}, t, tag, ν), k
                t_bytes = time_step.to_bytes(8, byteorder='big')
                tag_bytes = tag.encode('utf-8')
                seed_components = [cc, t_bytes, tag_bytes]
                if nonce is not None:
                    seed_components.append(nonce)
                
                rng = get_rng(*seed_components, private_key)
                
                # Generate r_t(y_t) for the specific token
                # We need to generate random values for all tokens to ensure
                # we get the same value for token_id as during generation
                # This is because PRF generates a sequence of random values
                # and we need the value at position token_id
                r_values = np.array([rng.random() for _ in range(vocab_size)])
                r_t_y_t = r_values[token_id]
                
                # Compute score: -log(1 - r_t(y_t))
                # Clamp r_t_y_t to avoid numerical issues
                r_t_y_t = np.clip(r_t_y_t, 1e-10, 1 - 1e-10)
                score = -np.log(1 - r_t_y_t)
                scores[b, t] = score
        
        return scores
    
    @classmethod
    def from_watermarkcode(
        cls,
        code,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
        # ERSD-specific parameters
        context_codes: np.ndarray = None,
        time_steps: np.ndarray = None,
        tags: np.ndarray = None,
        private_key: bytes = None,
        vocab_size: int = None,
        nonce: bytes = None,
    ) -> "ERSD_Aaronson_Score":
        """
        Create ERSD_Aaronson_Score from watermark code and ERSD-specific parameters.
        """
        scores = cls.score_from_watermarkcode(
            code,
            ids,
            p_logits,
            q_logits,
            context_codes=context_codes,
            time_steps=time_steps,
            tags=tags,
            private_key=private_key,
            vocab_size=vocab_size,
            nonce=nonce,
        )
        return cls(
            scores=scores.detach().cpu().numpy(),
            skipped=skipped,
        )
    
    def get_per_token_log_MGF(self, t: float) -> float:
        """
        Compute log MGF for a single token.
        
        For -log(1 - U) where U ~ Uniform(0,1):
        E[exp(t * -log(1 - U))] = E[(1 - U)^{-t}] = 1/(1-t) for t < 1
        So log MGF = -log(1-t) for t < 1
        """
        if t >= 1:
            return np.inf
        return -np.log(1 - t)
    
    def get_per_token_mu(self) -> float:
        """
        Expected value of -log(1 - U) where U ~ Uniform(0,1).
        
        E[-log(1 - U)] = 1
        """
        return 1.0

    def get_log_p_value(self) -> float:
        """
        Closed-form Chernoff bound for Aaronson score.

        For per-token log MGF: -log(1 - t), the minimizing t satisfies:
          t* = 1 - 1/s  for s > 1 (s = mean per-token score)
        If s <= 1, the minimum is at t = 0 and log p-value = 0.
        """
        num_added = self.get_num_added()
        if num_added == 0:
            return 0.0
        s = self.get_mean_per_token_score()
        if not np.isfinite(s):
            return np.nan
        if s <= 1.0:
            return 0.0
        # log p per token: log(s) - s + 1
        log_p_per_token = np.log(s) - s + 1.0
        return log_p_per_token * num_added


def _log_gamma_tail_p_value(num_added: int, score_sum: float) -> float:
    if num_added == 0:
        return 0.0
    if not np.isfinite(score_sum):
        return np.nan
    l = torch.tensor(float(num_added), dtype=torch.float64)
    s = torch.tensor(float(score_sum), dtype=torch.float64)
    p = torch.special.gammaincc(l, s)
    if p <= 0:
        return -np.inf
    return float(torch.log(p).item())


def _log_e_minus1_over_lambda(lam: float) -> float:
    if lam <= 0:
        return -np.inf
    if lam > 50:
        # log((e^lam - 1)/lam) = lam + log(1 - e^{-lam}) - log(lam)
        return lam + np.log1p(-np.exp(-lam)) - np.log(lam)
    return np.log(np.expm1(lam)) - np.log(lam)


def _solve_u_chernoff_lambda(mean_u: float) -> float:
    # mean_u in (0.5, 1.0]; solve g(lam) = 0
    def g(lam: float) -> float:
        # g(lam) = e^lam/(e^lam-1) - 1/lam - mean_u
        if lam <= 0:
            return 0.5 - mean_u
        if lam > 50:
            # e^lam/(e^lam-1) -> 1
            return 1.0 - (1.0 / lam) - mean_u
        e = np.exp(lam)
        return (e / (e - 1.0)) - (1.0 / lam) - mean_u

    low = 1e-6
    high = 1.0
    while g(high) < 0 and high < 100.0:
        high *= 2.0
    # Bisection
    for _ in range(60):
        mid = 0.5 * (low + high)
        if g(mid) < 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


class ERSD_Aaronson_Gamma_Score(ERSD_Aaronson_Score):
    """
    Exact Gamma tail p-value for Aaronson TestScore:
    S_A ~ Gamma(L, scale=1) under null, p = P(Gamma >= S_A).
    """

    def get_log_p_value(self) -> float:
        score_sum = self.get_score()
        return _log_gamma_tail_p_value(self.get_num_added(), score_sum)


class ERSD_Aaronson_U_Score(ERSD_Aaronson_Score):
    """
    Chernoff bound p-value using U_t = r_t(y_t) with U-score framework.
    """

    def get_log_p_value(self) -> float:
        num_added = self.get_num_added()
        if num_added == 0:
            return 0.0
        # Recover U_t from score = -log(1 - U_t)
        scores = self.scores * (~self.skipped)
        u_vals = 1.0 - np.exp(-scores)
        s_u = u_vals.sum()
        mean_u = s_u / num_added
        if not np.isfinite(mean_u):
            return np.nan
        if mean_u <= 0.5:
            return 0.0
        lam = _solve_u_chernoff_lambda(float(mean_u))
        log_term = _log_e_minus1_over_lambda(lam)
        log_p = num_added * log_term - lam * s_u
        return float(log_p)
