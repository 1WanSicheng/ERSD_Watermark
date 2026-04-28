"""
Li et al. (2025) optimal Gumbel score.

For the distribution class
    P_Delta = { P : max_v P(v) <= 1 - Delta },
the optimal per-token score is

    h^*_{gum,Delta}(r) = log( k_Delta r^(Delta/(1-Delta))
                              + 1{q_Delta > 0} r^((1-q_Delta)/q_Delta) )

where
    k_Delta = floor( 1 / (1 - Delta) ),
    q_Delta = 1 - k_Delta * (1 - Delta).

It is the log-likelihood ratio against the least-favorable distribution
    P*_Delta = (1-Delta, ..., 1-Delta, q_Delta, 0, ..., 0)
            (k_Delta copies of (1-Delta); the q_Delta term is dropped when q=0).

This functional form is identical to DeltaGumbel_X; this module exposes it
under a clearly-named entry point with the Li et al. reference, leaving
DeltaGumbel_X untouched for backward compatibility.

Reference: Li et al., 'Statistical analysis of LLM watermarking', 2025.
"""
import numpy as np
from torch import FloatTensor, LongTensor

from .. import AbstractWatermarkCode
from .deltagumbel_X import DeltaGumbel_X, DeltaGumbel_X_Builder


class DeltaGumbel_Li(DeltaGumbel_X):
    """Optimal Gumbel score of Li et al. (2025) for the class P_Delta."""

    @classmethod
    def from_watermarkcode(*args, **kwargs):
        raise ValueError(
            "Don't use from_watermarkcode for DeltaGumbel_Li directly. "
            "Use DeltaGumbel_Li.builder(Delta) instead."
        )

    @classmethod
    def builder(cls, Delta: float):
        return DeltaGumbel_Li_Builder(Delta)


class DeltaGumbel_Li_Builder(DeltaGumbel_X_Builder):
    def from_watermarkcode(
        self,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> DeltaGumbel_Li:
        assert ids.shape == skipped.shape
        s = DeltaGumbel_Li(
            scores=DeltaGumbel_Li.score_from_watermarkcode(
                code, ids, p_logits, q_logits, self.floor, self.a1, self.a2
            )
            .detach()
            .cpu()
            .numpy(),
            skipped=skipped,
        )
        s.Delta = self.Delta
        s.tildeDelta = self.tildeDelta
        s.floor = self.floor
        s.a1 = self.a1
        s.a2 = self.a2
        return s

    def __repr__(self):
        return f"DeltaGumbel_Li(Delta={self.Delta})"
