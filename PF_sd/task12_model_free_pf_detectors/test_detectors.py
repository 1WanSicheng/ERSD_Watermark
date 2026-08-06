import math

import numpy as np

from .detectors import (
    DetectorSpec,
    NullCalibrator,
    li_score,
    original_score,
    power_law_score,
)


def test_pdf_equations() -> None:
    pivots = np.array([0.01, 0.2, 0.9])
    assert np.allclose(original_score(pivots), -np.log(pivots))

    rho = 0.2
    expected_density = 1.0 - rho * pivots
    expected_density += np.where(pivots <= rho, 1.0 - pivots / rho, 0.0)
    assert np.allclose(li_score(pivots, rho), np.log(expected_density))

    eps = 0.05
    expected_pl = np.minimum(pivots ** -0.5, eps ** -0.5) - (
        2.0 - math.sqrt(eps)
    )
    assert np.allclose(power_law_score(pivots, eps), expected_pl)


def test_power_law_is_centered_under_null() -> None:
    rng = np.random.default_rng(7)
    score = power_law_score(rng.random(1_000_000), 0.05)
    assert abs(float(score.mean())) < 0.01


def test_monte_carlo_threshold_has_requested_null_tail() -> None:
    detector = DetectorSpec("PL", "power_law", 0.05)
    calibration = NullCalibrator(detector, 64, n_null=50_000, seed=11)
    threshold = calibration.threshold(0.01)
    realized = float(np.mean(calibration.null_scores >= threshold))
    assert 0.008 <= realized <= 0.012


def test_small_pivots_score_as_watermarked() -> None:
    pivots = np.full(64, 0.02)
    for detector in (
        DetectorSpec("Original", "original"),
        DetectorSpec("Li", "li", 0.2),
        DetectorSpec("PL", "power_law", 0.05),
    ):
        calibration = NullCalibrator(detector, 64, n_null=20_000, seed=13)
        assert calibration.evaluate(pivots)["p_value"] < 0.001
