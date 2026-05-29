"""Unit tests for cardiac load calculation."""

from services.cardiac import cardiac_load


def test_cardiac_load_formula() -> None:
    assert cardiac_load(80.0, 120.0) == 9.6


def test_cardiac_load_returns_none_when_missing_inputs() -> None:
    assert cardiac_load(None, 120.0) is None
    assert cardiac_load(70.0, None) is None
    assert cardiac_load(0, 120.0) is None
