"""Unit tests for the VitalsRecord model."""

from models.vitals_record import VitalsRecord


def test_to_csv_row_matches_fields() -> None:
    record = VitalsRecord(
        timestamp="2026-05-29 12:00:00",
        heart_rate_bpm=72.0,
        breathing_rate_bpm=14.0,
        sbp_mmhg=120.0,
        dbp_mmhg=80.0,
        hrv_rmssd_ms=45.0,
        cardiac_load_rpp=8.6,
        sqi=0.85,
    )
    row = record.to_csv_row()
    assert row["heart_rate_bpm"] == 72.0
    assert row["sqi"] == 0.85
    assert len(row) == 8
