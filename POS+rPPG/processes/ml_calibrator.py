import cv2
import queue
import numpy as np
import rppg


def run_ml_calibrator(ipc):
    print("[ML_ENGINE] Process initializing. Loading deep learning model...")

    try:
        model = rppg.Model()
        print("[ML_ENGINE] Model loaded. Ready to receive frames.")
        ipc.ml_ready_event.set()

        collected_frames = []
        first_ts = None
        ACCUMULATE_SEC = 30.0

        while not ipc.ml_kill_event.is_set():
            try:
                ts, frame_bgr = ipc.frame_buffer_queue.get(timeout=0.5)

                if first_ts is None:
                    first_ts = ts

                frame_resized = cv2.resize(frame_bgr, (320, 240))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                collected_frames.append(frame_rgb)

                if ts - first_ts >= ACCUMULATE_SEC:
                    break

            except queue.Empty:
                continue

        if not ipc.ml_kill_event.is_set() and len(collected_frames) > 0:
            print(f"[ML_ENGINE] Running inference on {len(collected_frames)} frames (240x180)...")

            target_count = int(ACCUMULATE_SEC * 30.0)
            indices = np.linspace(0, len(collected_frames) - 1, target_count, dtype=int)
            uniform_frames = [collected_frames[i] for i in indices]
            tensor = np.array(uniform_frames, dtype=np.uint8)
            result = model.process_video_tensor(tensor, fps=30.0)
            print(f"[ML_ENGINE] Raw output: {result}")

            if result:
                sqi_val = float(result.get("SQI") or 0.0)
                hr_val  = result.get("hr")

                ipc.ml_sqi_value.value = sqi_val

                if hr_val and sqi_val > 0.3:
                    hr_float = float(hr_val)
                    ipc.anchor_hr_value.value = hr_float
                    print(f"[ML_ENGINE] >> LOCK ACQUIRED << HR: {hr_float:.1f} BPM (SQI: {sqi_val:.2f})")
                else:
                    print(f"[ML_ENGINE] Calibration rejected — SQI: {sqi_val:.2f} | HR: {hr_val}")
            else:
                print("[ML_ENGINE] Calibration failed. Model returned empty result.")

        ipc.ml_done_event.set()
        print("[ML_ENGINE] Calibration phase complete. Process exiting.")

    except Exception as e:
        print(f"[ML_ENGINE] CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        ipc.ml_done_event.set()