"""Cardiac load (rate pressure product) calculations."""

from typing import Optional


def cardiac_load(hr: Optional[float], sbp: Optional[float]) -> Optional[float]:
    """
    Compute rate pressure product (RPP) = HR × SBP / 1000.

    Normal rest: 7–10 | Moderate exercise: 15–25 | High: >25

    Args:
        hr: Heart rate in beats per minute.
        sbp: Systolic blood pressure in mmHg.

    Returns:
        RPP value rounded to one decimal, or None if inputs are missing.
    """
    if hr and sbp:
        return round((hr * sbp) / 1000, 1)
    return None
