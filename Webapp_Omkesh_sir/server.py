import sys
import os
from pathlib import Path
from services.bp_estimator import estimate_bp
from services.cardiac import cardiac_load

# Absolute paths
THIS_DIR = Path(__file__).resolve().parent          # .../open-rppg/webapp
ROOT_DIR = THIS_DIR.parent                          # .../open-rppg
STATIC_DIR = THIS_DIR / "static"
INDEX_FILE = os.path.join(STATIC_DIR, "index.html")

# So "from rppg import Model" works when run from root
sys.path.insert(0, str(ROOT_DIR))

import asyncio
import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rppg import Model


app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] client connected")

    model = Model("ME-flow.rlap")
    frame_count = 0
    current_hr = None
    current_sqi = None
    last_hr_at = None

    with model:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    continue

                arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                now = time.monotonic()
                model.update_frame(frame, now)
                frame_count += 1

                payload = {
                    "frame": frame_count,
                    "ready": False,
                }

                if model.has_signal:
                    try:
                        bvp, ts = model.bvp()
                        if len(bvp) > 0:
                            payload["ready"] = True
                            payload["bvp"] = float(bvp[-1])
                    except Exception as e:
                        print(f"BVP error: {e}")

                if frame_count % 30 == 0 and model.has_signal:
                    try:
                        result = model.hr(start = -30)
                        if(
                            result
                            and result.get("hr")
                        ):
                            hr = result["hr"]

                            hrv = result.get("hrv") or {}
                            rmssd = hrv.get("rmssd")

                            br_hz = hrv.get("breathingrate")
                            breathing = round(br_hz * 60, 1) if br_hz else None

                            bvp_signal, _ = model.bvp()

                            sbp, dbp = estimate_bp(
                            bvp_signal,
                            hr,
                            )
                            
                            load = cardiac_load(
                                hr,
                                sbp,
                            )

                            current_hr = round(hr, 1)
                            current_sqi = round(
                                float(result.get("SQI") or 0),
                                2,
                            )
                            last_hr_at = now

                            print(
                                f"HR={current_hr} | "
                                f"RMSSD={rmssd} | "
                                f"BR={breathing} | "
                                f"BP={sbp}/{dbp} | "
                                f"LOAD={load} | "
                                f"SQI={current_sqi}"
                            )
                    except Exception as e:
                        print(f"HR error: {e}")


                if (
                    current_hr is not None
                    and last_hr_at is not None
                    and (now - last_hr_at) < 5.0
                ):
                    payload["hr"] = current_hr
                    payload["sqi"] = current_sqi

                await websocket.send_text(json.dumps(payload))

        except WebSocketDisconnect:
            print("[ws] client disconnected")
        except Exception as e:
            print(f"[ws] error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)