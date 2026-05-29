"""Blood pressure estimation from BVP waveform."""

from typing import Optional, Sequence, Union

import numpy as np
from scipy.signal import find_peaks

from config.settings import (
    BP_DBP_MAX,
    BP_DBP_MIN,
    BP_HR_FACTOR_COEFF,
    BP_MIN_SIGNAL_LENGTH,
    BP_PEAK_DISTANCE,
    BP_PEAK_PROMINENCE,
    BP_REF_AMP,
    BP_REF_HR,
    BP_REF_MAP,
    BP_REF_PP,
    BP_SBP_MAX,
    BP_SBP_MIN,
)

Signal = Union[Sequence[float], np.ndarray]


def estimate_bp(
    bvp_signal: Optional[Signal],
    hr: float,
) -> tuple[Optional[float], Optional[float]]:
    """
    Estimate systolic and diastolic BP from a BVP waveform and heart rate.

    BVP is assumed normalized by the library (z-scored, clipped to ~[-2, max]).
    Peak-to-trough amplitude is used as a proxy for pulse pressure.

    Reference anchors (population means at rest):
      - Pulse pressure (SBP-DBP) ~40 mmHg
      - Mean arterial pressure ~93 mmHg
      - HR ~70 bpm

    Args:
        bvp_signal: Blood volume pulse time series, or None if unavailable.
        hr: Heart rate in beats per minute.

    Returns:
        Tuple of (systolic_mmhg, diastolic_mmhg), or (None, None) if estimation fails.
    """
    if bvp_signal is None or len(bvp_signal) < BP_MIN_SIGNAL_LENGTH:
        return None, None

    sig = np.array(bvp_signal, dtype=float)

    peaks, _ = find_peaks(sig, distance=BP_PEAK_DISTANCE, prominence=BP_PEAK_PROMINENCE)
    troughs, _ = find_peaks(-sig, distance=BP_PEAK_DISTANCE, prominence=BP_PEAK_PROMINENCE)

    if len(peaks) < 2 or len(troughs) < 2:
        return None, None

    amplitudes: list[float] = []
    for peak_idx in peaks:
        prior_troughs = troughs[troughs < peak_idx]
        if len(prior_troughs):
            amplitudes.append(float(sig[peak_idx] - sig[prior_troughs[-1]]))

    if not amplitudes:
        return None, None

    mean_amp = float(np.mean(amplitudes))

    hr_factor = 1 + BP_HR_FACTOR_COEFF * (hr - BP_REF_HR)
    amp_factor = mean_amp / BP_REF_AMP

    pp = BP_REF_PP * amp_factor * hr_factor
    map_ = BP_REF_MAP * hr_factor

    sbp = map_ + pp * (2 / 3)
    dbp = map_ - pp * (1 / 3)

    sbp_clamped = round(max(BP_SBP_MIN, min(BP_SBP_MAX, sbp)), 1)
    dbp_clamped = round(max(BP_DBP_MIN, min(BP_DBP_MAX, dbp)), 1)
    return sbp_clamped, dbp_clamped
