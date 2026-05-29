"""Camera capture loop and vitals measurement orchestration."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import rppg

from config.settings import (
    CAMERA_INDEX,
    CSV_FILE,
    FACE_BOX_COLOR_BGR,
    FACE_BOX_THICKNESS,
    HR_WINDOW_START,
    MEASUREMENT_INTERVAL_SEC,
    PRINT_SEPARATOR,
    SQI_THRESHOLD,
    WINDOW_TITLE,
)
from models.vitals_record import VitalsRecord
from services.bp_estimator import estimate_bp
from services.cardiac import cardiac_load
from services.logger import init_csv, log_to_csv


class VitalsMonitor:
    """
    Runs the rPPG camera preview loop, samples vitals on an interval,
    and logs results to CSV.
    """

    def __init__(
        self,
        csv_path: Path = CSV_FILE,
        camera_index: int = CAMERA_INDEX,
        model: Optional[Any] = None,
    ) -> None:
        """
        Args:
            csv_path: Path to the vitals CSV log file.
            camera_index: OpenCV camera device index.
            model: Optional pre-constructed rppg.Model instance (for testing).
        """
        self._csv_path = csv_path
        self._camera_index = camera_index
        init_csv(self._csv_path)
        self._model = model if model is not None else rppg.Model()

    def run(self) -> None:
        """Start camera capture and process frames until the user quits."""
        print("Model loaded. Starting camera...")

        with self._model.video_capture(self._camera_index):
            print("Camera open. Reading vitals every 5 seconds...\n")
            last_measurement = 0.0

            for frame, box in self._model.preview:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if time.time() - last_measurement > MEASUREMENT_INTERVAL_SEC:
                    self._process_measurement()
                    last_measurement = time.time()

                if box is not None:
                    y1, y2 = box[0]
                    x1, x2 = box[1]
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        FACE_BOX_COLOR_BGR,
                        FACE_BOX_THICKNESS,
                    )

                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._model.alive = False
                    break

    def _process_measurement(self) -> None:
        """Sample HR/HRV/BVP, estimate BP, log and print vitals if SQI is sufficient."""
        result = self._model.hr(start=HR_WINDOW_START)

        if not result or not result.get("hr") or result.get("SQI", 0) <= SQI_THRESHOLD:
            print("Waiting for signal... (stay still, face the light)")
            return

        hr = result["hr"]
        hrv = result.get("hrv") or {}
        rmssd = hrv.get("rmssd")

        br_hz = hrv.get("breathingrate")
        breathing = round(br_hz * 60, 1) if br_hz else None

        bvp_signal, _ = self._model.bvp()
        sbp, dbp = estimate_bp(bvp_signal, hr)
        load = cardiac_load(hr, sbp)

        rmssd_disp = round(rmssd, 1) if rmssd else None

        record = VitalsRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            heart_rate_bpm=round(hr, 1),
            breathing_rate_bpm=breathing,
            sbp_mmhg=sbp,
            dbp_mmhg=dbp,
            hrv_rmssd_ms=rmssd_disp,
            cardiac_load_rpp=load,
            sqi=round(result.get("SQI", 0), 3),
        )
        log_to_csv(record, self._csv_path)
        self._print_vitals(hr, breathing, sbp, dbp, rmssd_disp, load)

    def _print_vitals(
        self,
        hr: float,
        breathing: Optional[float],
        sbp: Optional[float],
        dbp: Optional[float],
        rmssd_disp: Optional[float],
        load: Optional[float],
    ) -> None:
        """Print a formatted vitals summary to stdout."""
        print(PRINT_SEPARATOR)
        print(f"  Heart Rate     : {hr:.1f} BPM")
        print(f"  Breathing Rate : {breathing} breaths/min")
        print(f"  BP (estimated) : {sbp}/{dbp} mmHg")
        print(f"  HRV (RMSSD)    : {rmssd_disp} ms")
        print(f"  Cardiac Load   : {load}  (RPP ×10³)")
        print(f"  Logged → {self._csv_path}")
        print()
