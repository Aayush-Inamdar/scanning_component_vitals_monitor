"""CSV log file initialization and append operations."""

import csv
from pathlib import Path

from config.settings import CSV_FILE, CSV_HEADERS
from models.vitals_record import VitalsRecord


def init_csv(path: Path = CSV_FILE) -> None:
    """
    Create the vitals CSV with headers if it does not exist yet.

    Args:
        path: Destination CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=CSV_HEADERS).writeheader()
        print(f"Created log file: {path}")
    else:
        print(f"Appending to existing log: {path}")


def log_to_csv(record: VitalsRecord, path: Path = CSV_FILE) -> None:
    """
    Append one vitals record to the CSV log.

    Args:
        record: Vitals measurement to persist.
        path: Destination CSV file path.
    """
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=CSV_HEADERS).writerow(record.to_csv_row())
