"""Browser-camera vitals monitoring server.

Pipeline per frame:
  Browser JPEG → WebSocket → FaceMeshProcessor → open-rppg → vitals JSON

Run from project root:
    python server.py
"""

import sys
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Project root is the directory this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rppg import Model
from config.settings import CSV_FILE
from models.vitals_record import VitalsRecord
from services.bp_estimator import estimate_bp
from services.cardiac import cardiac_load
from services.face_mesh_processor import FaceMeshProcessor
from services.logger import init_csv, log_to_csv

SCAN_DURATION = 30.0
STATIC_DIR    = PROJECT_ROOT / "webapp" / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
INDEX_FILE = STATIC_DIR / "index.html"

# Initialise CSV log on startup (creates file + headers if missing)
init_csv(CSV_FILE)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    print("[ws] client connected")

    model = Model("ME-flow.rlap")
    frame_count = 0

    # Scan state
    scan_started            = False
    scan_start_time: float | None = None
    scan_complete           = False

    # Rolling live vitals cache (updated every 30 frames during scan)
    live_vitals: dict       = {}
    last_vitals_at: float | None = None

    with model, FaceMeshProcessor() as mesh:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(), timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue

                arr       = np.frombuffer(data, np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                now = time.monotonic()
                frame_count += 1

                # --- MediaPipe ROI masking ----------------------------
                masked_frame, roi_mask, face_found = mesh.process(frame_bgr)

                if face_found:
                    face_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
                    model.update_face(face_rgb, ts=now, hasface=True)
                else:
                    model.update_face(None, ts=now, hasface=False)

                # --- Base payload ------------------------------------
                payload: dict = {
                    "frame":     frame_count,
                    "face":      face_found,
                    "phase":     "preparing",
                    "progress":  0,
                    "remaining": int(SCAN_DURATION),
                    "ready":     False,
                }

                # --- BVP stream (always when signal available) --------
                if model.has_signal:
                    try:
                        bvp, _ = model.bvp()
                        if len(bvp) > 0:
                            payload["ready"] = True
                            payload["bvp"]   = float(bvp[-1])
                    except Exception as e:
                        print(f"[warn] bvp: {e}")

                # --- Start scan once we have signal + face ------------
                if not scan_started and not scan_complete:
                    if model.has_signal and face_found:
                        scan_started    = True
                        scan_start_time = now
                        print("[ws] scan started")

                # --- Scan in progress ---------------------------------
                if scan_started and not scan_complete:
                    elapsed  = now - scan_start_time
                    progress = min(100, round((elapsed / SCAN_DURATION) * 100))
                    remaining = max(0, int(SCAN_DURATION - elapsed))

                    payload["phase"]     = "scanning"
                    payload["progress"]  = progress
                    payload["remaining"] = remaining

                    # Update live HR / BP / SQI every 30 frames
                    if frame_count % 30 == 0 and model.has_signal:
                        try:
                            result = model.hr(start=-30)
                            if result and result.get("hr"):
                                hr  = result["hr"]
                                sqi = round(float(result.get("SQI") or 0), 3)
                                bvp_signal, _ = model.bvp()
                                sbp, dbp = estimate_bp(bvp_signal, hr)

                                live_vitals = {
                                    "hr":  round(hr, 1),
                                    "sqi": sqi,
                                    "sbp": sbp,
                                    "dbp": dbp,
                                }
                                last_vitals_at = now
                        except Exception as e:
                            print(f"[warn] hr during scan: {e}")

                    # Include live vitals if fresh (< 5 s old)
                    if last_vitals_at and (now - last_vitals_at) < 5.0:
                        payload.update(live_vitals)

                    # --- Scan complete --------------------------------
                    if elapsed >= SCAN_DURATION:
                        scan_complete = True
                        print("[ws] scan complete — computing final vitals")
                        try:
                            result = model.hr(start=-30)
                            if result and result.get("hr"):
                                hr  = result["hr"]
                                sqi = round(float(result.get("SQI") or 0), 3)

                                hrv       = result.get("hrv") or {}
                                rmssd     = hrv.get("rmssd")
                                br_hz     = hrv.get("breathingrate")
                                breathing = round(br_hz * 60, 1) if br_hz else None

                                bvp_signal, _ = model.bvp()
                                sbp, dbp = estimate_bp(bvp_signal, hr)
                                load     = cardiac_load(hr, sbp)

                                # --- Log to CSV ----------------------
                                record = VitalsRecord(
                                    timestamp          = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    heart_rate_bpm     = round(hr, 1),
                                    breathing_rate_bpm = breathing,
                                    sbp_mmhg           = sbp,
                                    dbp_mmhg           = dbp,
                                    hrv_rmssd_ms       = round(rmssd, 1) if rmssd else None,
                                    cardiac_load_rpp   = load,
                                    sqi                = sqi,
                                )
                                log_to_csv(record, CSV_FILE)
                                print(f"[ws] vitals logged → {CSV_FILE}")

                                payload["phase"]     = "complete"
                                payload["progress"]  = 100
                                payload["remaining"] = 0
                                payload["hr"]        = round(hr, 1)
                                payload["sqi"]       = sqi
                                payload["sbp"]       = sbp
                                payload["dbp"]       = dbp
                                payload["rmssd"]     = round(rmssd, 1) if rmssd else None
                                payload["breathing"] = breathing
                                payload["load"]      = load

                        except Exception as e:
                            print(f"[warn] final vitals: {e}")

                elif scan_complete:
                    payload["phase"]     = "complete"
                    payload["progress"]  = 100
                    payload["remaining"] = 0

                await websocket.send_text(json.dumps(payload))

        except WebSocketDisconnect:
            print("[ws] client disconnected")
        except Exception as e:
            print(f"[ws] error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)