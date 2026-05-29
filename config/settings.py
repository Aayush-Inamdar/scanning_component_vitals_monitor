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
HR_WINDOW_START: int = -30

# Camera
CAMERA_INDEX: int = 0
WINDOW_TITLE: str = "Vitals Monitor"
FACE_BOX_COLOR_BGR: tuple[int, int, int] = (0, 255, 0)
FACE_BOX_THICKNESS: int = 2

# Blood pressure estimation anchors (population means at rest)
BP_REF_AMP: float = 2.0
BP_REF_HR: float = 70.0
BP_REF_PP: float = 40.0  # pulse pressure (SBP - DBP)
BP_REF_MAP: float = 93.0  # mean arterial pressure
BP_HR_FACTOR_COEFF: float = 0.004

# Peak detection on BVP signal
BP_PEAK_DISTANCE: int = 10
BP_PEAK_PROMINENCE: float = 0.3
BP_MIN_SIGNAL_LENGTH: int = 10

# BP output clamps (mmHg)
BP_SBP_MIN: float = 90.0
BP_SBP_MAX: float = 180.0
BP_DBP_MIN: float = 55.0
BP_DBP_MAX: float = 110.0

# Console output
PRINT_SEPARATOR: str = "─" * 38
