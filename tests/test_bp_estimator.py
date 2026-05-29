"""Unit tests for blood pressure estimation."""

import numpy as np

from services.bp_estimator import estimate_bp


def test_estimate_bp_returns_none_for_short_signal() -> None:
    """Too few samples should not produce a BP estimate."""
    sbp, dbp = estimate_bp([0.1, 0.2], hr=70.0)
    assert sbp is None
    assert dbp is None


def test_estimate_bp_returns_none_for_none_signal() -> None:
    sbp, dbp = estimate_bp(None, hr=70.0)
    assert sbp is None
    assert dbp is None


def test_estimate_bp_produces_values_for_synthetic_waveform() -> None:
    """A simple oscillating signal with clear peaks should yield clamped BP."""
    t = np.linspace(0, 4 * np.pi, 120)
    signal = 2.0 * np.sin(t) + 0.5 * np.sin(2 * t)
    sbp, dbp = estimate_bp(signal, hr=70.0)
    assert sbp is not None
    assert dbp is not None
    assert 90 <= sbp <= 180
    assert 55 <= dbp <= 110
    assert sbp > dbp
