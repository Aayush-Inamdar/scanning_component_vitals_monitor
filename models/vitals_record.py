"""Data model for a single vitals measurement."""

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class VitalsRecord:
    """One row of vitals data suitable for CSV export and console display."""

    timestamp: str
    heart_rate_bpm: float
    breathing_rate_bpm: Optional[float]
    sbp_mmhg: Optional[float]
    dbp_mmhg: Optional[float]
    hrv_rmssd_ms: Optional[float]
    cardiac_load_rpp: Optional[float]
    sqi: float

    def to_csv_row(self) -> dict[str, Any]:
        """Return a dict matching the CSV column schema."""
        return asdict(self)
