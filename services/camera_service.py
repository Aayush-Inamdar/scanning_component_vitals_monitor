"""Camera capture loop and vitals measurement orchestration.

Capture pipeline:
  1. OpenCV reads raw frames from the camera
  2. FaceMeshProcessor (MediaPipe) detects 468 landmarks and returns a
     skin-only masked crop (forehead + cheeks, eyes/lips/hair excluded)
  3. The masked crop is fed to open-rppg via model.update_face() —
     bypassing open-rppg's own dumb bounding-box face detector entirely
  4. open-rppg runs its neural network on the clean crop → BVP → HR/SQI
  5. Every MEASUREMENT_INTERVAL_SEC seconds, vitals are computed and logged
"""

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
from services.face_mesh_processor import FaceMeshProcessor
from services.logger import init_csv, log_to_csv


class VitalsMonitor:
    """
    Runs the rPPG camera loop with MediaPipe-based skin ROI preprocessing,
    samples vitals on an interval, and logs results to CSV.
    """

    def __init__(
        self,
        csv_path: Path = CSV_FILE,
        camera_index: int = CAMERA_INDEX,
        model: Optional[Any] = None,
    ) -> None:
        self._csv_path = csv_path
        self._camera_index = camera_index
        init_csv(self._csv_path)
        self._model = model if model is not None else rppg.Model("ME-flow.rlap")

    def run(self) -> None:
        """Start camera capture and process frames until the user quits."""
        print("Model loaded. Starting camera...")

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._camera_index}")

        print("Camera open. Reading vitals every 5 seconds...\n")
        last_measurement = 0.0

        with self._model, FaceMeshProcessor() as mesh:
            try:
                while self._model.alive or True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    ts = time.time()

                    # --- MediaPipe: get skin-only masked crop ----------
                    masked_frame, roi_mask, face_found = mesh.process(frame_bgr)

                    if face_found:
                        # Feed the clean masked frame to open-rppg.
                        # update_face expects RGB.
                        face_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                        self._model.update_face(face_rgb, ts=ts, hasface=True)
                    else:
                        # Tell open-rppg no face this frame so it doesn't
                        # stall waiting for input.
                        self._model.update_face(None, ts=ts, hasface=False)

                    # --- Periodic vitals measurement -------------------
                    if ts - last_measurement > MEASUREMENT_INTERVAL_SEC:
                        self._process_measurement()
                        last_measurement = ts

                    # --- Display: draw ROI contour on original frame ---
                    display = mesh.draw_landmarks(frame_bgr, roi_mask)
                    status = "Face found" if face_found else "No face"
                    cv2.putText(
                        display,
                        status,
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        FACE_BOX_COLOR_BGR if face_found else (0, 0, 200),
                        2,
                    )
                    cv2.imshow(WINDOW_TITLE, display)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            finally:
                self._model.alive = False
                cap.release()

    def _process_measurement(self) -> None:
        """Sample HR/HRV/BVP, estimate BP, log and print vitals if SQI is sufficient."""
        result = self._model.hr(start=HR_WINDOW_START)

        if not result or not result.get("hr"):
            print("Waiting for signal... (stay still, face the light)")
            return

        sqi_val = round(float(result.get("SQI") or 0), 3)

        if sqi_val <= SQI_THRESHOLD:
            sqi_label = "good" if sqi_val >= 0.5 else "low" if sqi_val >= 0.3 else "poor"
            print(f"Signal too weak — SQI={sqi_val} ({sqi_label}). Stay still, face the light.")
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
            sqi=sqi_val,
        )
        log_to_csv(record, self._csv_path)
        self._print_vitals(hr, breathing, sbp, dbp, rmssd_disp, load, sqi_val)

    def _print_vitals(
        self,
        hr: float,
        breathing: Optional[float],
        sbp: Optional[float],
        dbp: Optional[float],
        rmssd_disp: Optional[float],
        load: Optional[float],
        sqi: float,
    ) -> None:
        """Print a formatted vitals summary to stdout."""
        sqi_label = "good" if sqi >= 0.5 else "low" if sqi >= 0.3 else "poor"
        print(PRINT_SEPARATOR)
        print(f"  SQI            : {sqi} ({sqi_label})")
        print(f"  Heart Rate     : {hr:.1f} BPM")
        print(f"  Breathing Rate : {breathing} breaths/min")
        print(f"  BP (estimated) : {sbp}/{dbp} mmHg")
        print(f"  HRV (RMSSD)    : {rmssd_disp} ms")
        print(f"  Cardiac Load   : {load}  (RPP ×10³)")
        print(f"  Logged → {self._csv_path}")
        print()