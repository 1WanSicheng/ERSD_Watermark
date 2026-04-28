"""
Lattimore (2026) truncated power-law detector for the Gumbel watermark.

Per-token score:
    h_{PL,eps}(r) = min( eps^{-1/2}, (1-r)^{-1/2} ) - (2 - sqrt(eps)).

The centering 2 - sqrt(eps) is chosen so that, under the null
(r ~ Unif[0,1]), E[h_{PL,eps}(r)] = 0.  Verified by direct integration:
    int_0^{1-eps} (1-r)^{-1/2} dr = 2 - 2 sqrt(eps),
    int_{1-eps}^1 eps^{-1/2} dr   = sqrt(eps),
total = 2 - sqrt(eps), minus the centering yields zero.

Unlike Aaronson's score (h_A(r) = -log(1-r), which diverges as r -> 1), the
PL score is bounded above by eps^{-1/2} - (2 - sqrt(eps)).  This robustness
is the motivation in Lattimore.

Reference: Lattimore, 'Refined detector for Gumbel watermark', 2026.
"""
import numpy as np
import torch
from torch import FloatTensor, LongTensor

from .. import DeltaGumbel_WatermarkCode, AbstractWatermarkCode
from . import TokenwiseAdditiveScore


class DeltaGumbel_PL(TokenwiseAdditiveScore):
    watermark_code_type = DeltaGumbel_WatermarkCode

    @classmethod
    def from_watermarkcode(*args, **kwargs):
        raise ValueError(
            "Don't use from_watermarkcode for DeltaGumbel_PL directly. "
            "Use DeltaGumbel_PL.builder(eps) instead."
        )

    @classmethod
    def builder(cls, eps: float):
        return DeltaGumbel_PL_Builder(eps)

    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor,
        q_logits: FloatTensor,
        eps: float,
    ) -> FloatTensor:
        assert isinstance(code, cls.watermark_code_type)
        assert code.g.shape[:-1] == ids.shape

        gs = torch.gather(code.g, -1, ids.unsqueeze(-1)).squeeze(-1)
        Us = torch.exp(-torch.exp(-gs))
        cap = float(eps) ** -0.5
        # min(cap, (1-U)^{-1/2}); equivalently clamp (1-U) below by eps.
        clamped = torch.clamp(1.0 - Us, min=float(eps))
        score = clamped.rsqrt() - (2.0 - float(eps) ** 0.5)
        return score

    def get_per_token_log_MGF(self, t: float) -> float:
        import scipy.integrate

        if isinstance(t, np.ndarray):
            assert t.size == 1
            t = t.item()

        eps = self.eps
        cap = eps ** -0.5
        center = 2.0 - eps ** 0.5

        # Integrand on [0, 1-eps): exp(t * ((1-r)^{-1/2} - center))
        def integrand_low(r, d):
            return np.exp(d * ((1.0 - r) ** -0.5 - center))

        low_part, _ = scipy.integrate.quad(
            integrand_low, 0.0, 1.0 - eps, args=(t,)
        )
        # On [1-eps, 1]: integrand is constant exp(t * (cap - center)),
        # mass = eps.
        high_part = eps * np.exp(t * (cap - center))
        return float(np.log(low_part + high_part))

    def get_per_token_mu(self) -> float:
        # By construction E_{H0}[h_{PL,eps}(U)] = 0 for U ~ Unif[0,1].
        return 0.0


class DeltaGumbel_PL_Builder:
    def __init__(self, eps: float):
        if not (0.0 < float(eps) < 1.0):
            raise ValueError(f"eps must be in (0, 1); got {eps}")
        self.eps = float(eps)

    def from_watermarkcode(
        self,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> DeltaGumbel_PL:
        assert ids.shape == skipped.shape
        s = DeltaGumbel_PL(
            scores=DeltaGumbel_PL.score_from_watermarkcode(
                code, ids, p_logits, q_logits, self.eps
            )
            .detach()
            .cpu()
            .numpy(),
            skipped=skipped,
        )
        s.eps = self.eps
        return s

    def __repr__(self):
        return f"DeltaGumbel_PL(eps={self.eps})"
