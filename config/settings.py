"""Application-wide configuration and constants."""

from pathlib import Path

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CSV_FILE: Path = PROJECT_ROOT / "data" / "vitals_log.csv"

# CSV schema
CSV_HEADERS: list[str] = [
    "timestamp",
    "heart_rate_bpm",
    "breathing_rate_bpm",
    "sbp_mmhg",
    "dbp_mmhg",
    "hrv_rmssd_ms",
    "cardiac_load_rpp",
    "sqi",
]

# Signal quality and timing
SQI_THRESHOLD: float = 0.3
MEASUREMENT_INTERVAL_SEC: float = 5.0
HR_WINDOW_START: int = -12

# Camera
CAMERA_INDEX: int = 0
WINDOW_TITLE: str = "Vitals Monitor"
FACE_BOX_COLOR_BGR: tuple[int, int, int] = (0, 255, 0)
FACE_BOX_THICKNESS: int = 2

# Blood pressure estimation anchors (population means at rest)
# BP_REF_AMP: expected peak-to-trough as a fraction of the signal's full range.
# A clean BVP pulse typically spans ~70% of the normalised range.
BP_REF_AMP: float = 0.7
BP_REF_HR: float = 70.0
BP_REF_PP: float = 40.0  # pulse pressure (SBP - DBP)
BP_REF_MAP: float = 93.0  # mean arterial pressure
BP_HR_FACTOR_COEFF: float = 0.004

# Peak detection on BVP signal
# BP_PEAK_PROMINENCE is now a fraction of the signal's peak-to-peak range (10%)
BP_PEAK_DISTANCE: int = 10
BP_PEAK_PROMINENCE: float = 0.10
BP_MIN_SIGNAL_LENGTH: int = 10

# BP output clamps (mmHg)
BP_SBP_MIN: float = 90.0
BP_SBP_MAX: float = 180.0
BP_DBP_MIN: float = 55.0
BP_DBP_MAX: float = 110.0

# Console output
PRINT_SEPARATOR: str = "─" * 38

# BVP signal processing (bvp_processor.py)
# open-rppg outputs BVP at ~30 fps; adjust BVP_SAMPLE_RATE if your camera differs
BVP_SAMPLE_RATE: float = 30.0

# Cardiac band: 0.7 Hz = 42 bpm low end, 3.5 Hz = 210 bpm high end
BVP_BANDPASS_LOW: float = 0.7
BVP_BANDPASS_HIGH: float = 3.5

# Respiration band: 0.1 Hz = 6 breaths/min, 0.5 Hz = 30 breaths/min
BVP_RESPIRATION_LOW: float = 0.1
BVP_RESPIRATION_HIGH: float = 0.5

# Minimum beats required to compute a valid RMSSD
BVP_MIN_HRV_BEATS: int = 4

# Minimum seconds of signal needed to extract a breathing rate
BVP_MIN_RESPIRATION_SECONDS: float = 20.0