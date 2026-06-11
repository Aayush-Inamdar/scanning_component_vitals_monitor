import cv2
import time
import numpy as np
import queue

from config.settings import CAMERA_INDEX, RESOLUTION_W, RESOLUTION_H, MOTION_THRESHOLD_PIXELS, COOLDOWN_SEC
from services.face_mesh_processor import FaceMeshProcessor

def run_producer(ipc):
    """
    COMPONENT A: The Visual Frame Producer
    Captures video, isolates ROI, draws the visual mask, and feeds the UI, Math, and ML queues.
    """
    print("[PRODUCER] Securing camera hardware...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)
    
    face_centers = []
    cooldown_until = 0
    start_time = time.time()
    ML_FRAME_SIZE = (128, 128)

    print("[PRODUCER] Camera secured. Waiting for MediaPipe face lock...")

    with FaceMeshProcessor() as mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time() - start_time
            
            # 1. MediaPipe ROI Extraction
            masked_frame, roi_mask, face_found, nose_y = mesh.process(
                frame, timestamp_ms=int(time.time() * 1000)
            )
            
            # 2. Draw the Visual Overlay for the UI
            display_frame = mesh.draw_landmarks(frame, roi_mask)
            
            is_moving = False
            if face_found:
                x, y, w, h = cv2.boundingRect(roi_mask)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_centers.append((x + w/2, y + h/2))
                
                if len(face_centers) > 30:
                    face_centers.pop(0)
                    xs, ys = [c[0] for c in face_centers], [c[1] for c in face_centers]
                    if np.max(xs) - np.min(xs) > MOTION_THRESHOLD_PIXELS or np.max(ys) - np.min(ys) > MOTION_THRESHOLD_PIXELS:
                        is_moving = True
                        cooldown_until = current_time + COOLDOWN_SEC 
                        cv2.putText(display_frame, "MOTION DETECTED - HOLD STILL", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 3. Stream the annotated video to the Main UI Thread
            try:
                # We use put_nowait. If the UI is busy, we just drop the frame.
                ipc.ui_frame_queue.put_nowait(display_frame)
            except queue.Full:
                pass

            # 4. Pipeline Data Routing
            if face_found and current_time > cooldown_until:
                
                if not ipc.roi_locked_event.is_set():
                    print("[PRODUCER] ROI Locked. Opening data streams...")
                    ipc.roi_locked_event.set()

                # Fast Lane (POS Math)
                b_channel, g_channel, r_channel = cv2.split(frame)
                valid_pixels = roi_mask > 0
                
                if np.any(valid_pixels):
                    mean_r = np.mean(r_channel[valid_pixels])
                    mean_g = np.mean(g_channel[valid_pixels])
                    mean_b = np.mean(b_channel[valid_pixels])
                    ipc.rgb_queue.put((current_time, mean_r, mean_g, mean_b, nose_y))

                # Heavy Lane (ML Calibration)
                if not ipc.ml_done_event.is_set() and ipc.ml_ready_event.is_set():
                    try:
                        ipc.frame_buffer_queue.put_nowait((current_time, frame.copy()))
                    except queue.Full:
                        pass

    cap.release()