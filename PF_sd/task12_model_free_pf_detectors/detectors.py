"""Model-free PF watermark detectors from Section 3 of the PF draft.

All detectors consume only recovered keyed pivots.  They never access target
or draft model logits.  The null distribution is therefore generated from
i.i.d. Uniform(0, 1) pivots and can be calibrated once for each text length.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np


Array = np.ndarray


def _pivots(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64)
    return np.clip(values, np.finfo(np.float64).tiny, 1.0)


def original_score(pivots: Array) -> Array:
    """Equation (8): per-token contributions to ``sum -log(V_t)``."""
    return -np.log(_pivots(pivots))


def li_score(pivots: Array, rho: float) -> Array:
    """Equation (9): binary-family log-likelihood-ratio contributions."""
    if not 0.0 < float(rho) < 1.0:
        raise ValueError("rho must lie in (0, 1)")
    v = _pivots(pivots)
    density = 1.0 - float(rho) * v
    density += np.where(v <= float(rho), 1.0 - v / float(rho), 0.0)
    return np.log(np.clip(density, np.finfo(np.float64).tiny, None))


def power_law_score(pivots: Array, eps: float) -> Array:
    """Equation (10): centered truncated power-law contributions."""
    if not 0.0 < float(eps) < 1.0:
        raise ValueError("eps must lie in (0, 1)")
    v = _pivots(pivots)
    cap = float(eps) ** -0.5
    return np.minimum(v ** -0.5, cap) - (2.0 - math.sqrt(float(eps)))


@dataclass(frozen=True)
class DetectorSpec:
    """A named, fixed detector whose sum is large under the watermark."""

    name: str
    family: str
    parameter: Optional[float] = None

    def contributions(self, pivots: Array) -> Array:
        if self.family == "original":
            return original_score(pivots)
        if self.family == "li":
            if self.parameter is None:
                raise ValueError("Li detector requires rho")
            return li_score(pivots, self.parameter)
        if self.family == "power_law":
            if self.parameter is None:
                raise ValueError("power-law detector requires eps")
            return power_law_score(pivots, self.parameter)
        raise ValueError(f"unknown detector family: {self.family}")

    def statistic(self, pivots: Array) -> float:
        return float(self.contributions(pivots).sum(dtype=np.float64))


class NullCalibrator:
    """Monte-Carlo null calibration for a fixed detector and text length.

    The finite-sample p-value uses ``(1 + exceedances) / (1 + n_null)``.
    This correction remains valid even at the edge of the simulated tail.
    """

    def __init__(
        self,
        detector: DetectorSpec,
        length: int,
        *,
        n_null: int = 200_000,
        seed: int = 20260806,
        chunk_size: int = 10_000,
    ) -> None:
        if int(length) <= 0:
            raise ValueError("length must be positive")
        if int(n_null) <= 0:
            raise ValueError("n_null must be positive")
        self.detector = detector
        self.length = int(length)
        self.n_null = int(n_null)
        rng = np.random.default_rng(int(seed))
        chunks = []
        remaining = self.n_null
        while remaining:
            count = min(int(chunk_size), remaining)
            uniforms = rng.random((count, self.length), dtype=np.float64)
            scores = detector.contributions(uniforms).sum(axis=1, dtype=np.float64)
            chunks.append(scores)
            remaining -= count
        self.null_scores = np.sort(np.concatenate(chunks))

    def p_value(self, statistic: float) -> float:
        left = int(np.searchsorted(self.null_scores, float(statistic), side="left"))
        exceedances = self.n_null - left
        return (1.0 + exceedances) / (1.0 + self.n_null)

    def threshold(self, false_positive_rate: float) -> float:
        alpha = float(false_positive_rate)
        if not 0.0 < alpha < 1.0:
            raise ValueError("false_positive_rate must lie in (0, 1)")
        # Smallest threshold whose simulated upper-tail mass is at most alpha.
        index = int(math.ceil((1.0 - alpha) * self.n_null)) - 1
        return float(self.null_scores[min(max(index, 0), self.n_null - 1)])

    def evaluate(self, pivots: Array) -> dict[str, float]:
        pivots = _pivots(pivots).reshape(-1)
        if len(pivots) != self.length:
            raise ValueError(f"expected {self.length} pivots, got {len(pivots)}")
        statistic = self.detector.statistic(pivots)
        p_value = self.p_value(statistic)
        return {
            "score": statistic,
            "score_per_token": statistic / self.length,
            "p_value": p_value,
            "ANLPPT": -math.log(p_value) / self.length,
        }
