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
import numpy as np

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
        self._model = model if model is not None else rppg.Model()

    def run(self) -> None:
        """Start camera capture and process frames until the user quits."""
        print("Model loaded. Starting camera...")

        import os
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._camera_index, backend)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera index {self._camera_index}")

        # Force uncompressed pixels and high resolution
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Apply your chosen -3 manual exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
        cap.set(cv2.CAP_PROP_EXPOSURE, -3)        
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)          

        print("Camera open. Waiting for stable vitals...\n")
        
        expected_fps = 30.0
        time_step = 1.0 / expected_fps
        base_ts = time.time()
        frame_idx = 0
        last_measurement = base_ts 
        stable_bbox = None

        with self._model, FaceMeshProcessor() as mesh:
            try:
                while self._model.alive or True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    current_ts = time.time()
                    
                    smoothed_frame = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

                    masked_frame, roi_mask, face_found = mesh.process(
                        smoothed_frame,
                        timestamp_ms=int(time.monotonic() * 1000),
                    )

                    if face_found:
                        x, y, w, h = cv2.boundingRect(roi_mask)
                        
                        if w > 0 and h > 0:
                            size = max(w, h)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            
                            # Stable EMA Box
                            if stable_bbox is None:
                                stable_bbox = [center_x, center_y, size]
                            else:
                                alpha = 0.15 
                                stable_bbox[0] = stable_bbox[0] * (1 - alpha) + center_x * alpha
                                stable_bbox[1] = stable_bbox[1] * (1 - alpha) + center_y * alpha
                                stable_bbox[2] = stable_bbox[2] * (1 - alpha) + size * alpha
                                
                            s_cx = int(stable_bbox[0])
                            s_cy = int(stable_bbox[1])
                            s_size = int(stable_bbox[2])
                            
                            half_size = s_size // 2 
                            
                            y1 = max(0, s_cy - half_size)
                            y2 = min(frame_bgr.shape[0], s_cy + half_size)
                            x1 = max(0, s_cx - half_size)
                            x2 = min(frame_bgr.shape[1], s_cx + half_size)
                            
                            # Tight crop to remove black pixels
                            cropped_face = smoothed_frame[y1:y2, x1:x2]
                            
                            if cropped_face.size > 0:
                                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                                self._model.update_face(face_rgb, ts=current_ts, hasface=True)
                            else:
                                self._model.update_face(None, ts=current_ts, hasface=False)
                        else:
                            self._model.update_face(None, ts=current_ts, hasface=False)
                    else:
                        self._model.update_face(None, ts=current_ts, hasface=False)

                    if current_ts - last_measurement > MEASUREMENT_INTERVAL_SEC:
                        self._process_measurement()
                        last_measurement = current_ts

                    # Display
                    display = mesh.draw_landmarks(frame_bgr, roi_mask)
                    status = "Face found" if face_found else "No face"
                    cv2.putText(
                        display, status, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        FACE_BOX_COLOR_BGR if face_found else (0, 0, 200), 2
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
