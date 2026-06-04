import cv2
import time
import numpy as np

from config.settings import CAMERA_INDEX, RESOLUTION_W, RESOLUTION_H, BUFFER_DURATION_SEC, HR_WINDOW_SEC, EXTENDED_WINDOW_SEC, MOTION_THRESHOLD_PIXELS, COOLDOWN_SEC
from services.face_mesh_processor import FaceMeshProcessor
from services.signal_processor import PosSignalProcessor
from services.data_logger import VitalsLogger

class VitalsCameraService:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)
        
        self.processor = PosSignalProcessor()
        self.logger = VitalsLogger()

    def run(self):
        print("Pure-Vitals-Engine Camera Service Active")
        
        # State Management
        times, r_buffer, g_buffer, b_buffer = [], [], [], []
        face_centers = [] 
        start_time = time.time()
        last_calc_time = 0
        cooldown_until = 0 
        
        # UI State
        ui_status = "Initializing..."
        ui_color = (0, 255, 255)
        current_metrics = {'hr': '--', 'hrv': '--', 'br': '--'}

        with FaceMeshProcessor() as mesh:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                current_time = time.time() - start_time

                masked_frame, roi_mask, face_found = mesh.process(
                    frame, timestamp_ms=int(time.time() * 1000)
                )

                is_moving = False
                if face_found:
                    x, y, w, h = cv2.boundingRect(roi_mask)
                    face_centers.append((x + w/2, y + h/2))
                    
                    # Check Macro-Motion
                    if len(face_centers) > 30:
                        face_centers.pop(0)
                        xs, ys = [c[0] for c in face_centers], [c[1] for c in face_centers]
                        if np.max(xs) - np.min(xs) > MOTION_THRESHOLD_PIXELS or np.max(ys) - np.min(ys) > MOTION_THRESHOLD_PIXELS:
                            is_moving = True
                            cooldown_until = current_time + COOLDOWN_SEC 

                # Buffer Logic
                if face_found and current_time > cooldown_until:
                    b_channel, g_channel, r_channel = cv2.split(frame)
                    valid_pixels = roi_mask > 0
                    if np.any(valid_pixels):
                        times.append(current_time)
                        r_buffer.append(np.mean(r_channel[valid_pixels]))
                        g_buffer.append(np.mean(g_channel[valid_pixels]))
                        b_buffer.append(np.mean(b_channel[valid_pixels]))
                
                elif is_moving:
                    # Total Burn: Delete the massive 30-second array if motion is detected
                    times.clear(); r_buffer.clear(); g_buffer.clear(); b_buffer.clear()
                    self.processor.reset()
                    ui_status = "MOTION DETECTED! Resetting..."
                    ui_color = (0, 0, 255)
                    current_metrics = {'hr': '--', 'hrv': '--', 'br': '--'}

                # Master Circular Buffer Cleanup (Keeps 30s)
                while len(times) > 0 and times[0] < current_time - BUFFER_DURATION_SEC:
                    times.pop(0); r_buffer.pop(0); g_buffer.pop(0); b_buffer.pop(0)

                # Execution Engine (Runs every 1 second)
                if current_time - last_calc_time > 1.0:
                    if current_time <= cooldown_until:
                        pass
                    elif len(times) > (HR_WINDOW_SEC * 15): # Minimum 10s collected for HR
                        
                        results = self.processor.process(np.array(times), np.array(r_buffer), np.array(g_buffer), np.array(b_buffer))
                        
                        if results['hr']:
                            current_metrics['hr'] = results['hr']
                            
                            # Determine UI state based on if we reached 30s yet
                            if results['hrv'] and results['br']:
                                current_metrics['hrv'] = results['hrv']
                                current_metrics['br'] = results['br']
                                ui_status = "Full Diagnostics Locked"
                                ui_color = (0, 255, 0)
                            else:
                                percent = min(100, int((len(times) / (EXTENDED_WINDOW_SEC * 30)) * 100))
                                ui_status = f"HR Locked. Acquiring advanced... {percent}%"
                                ui_color = (0, 255, 255)
                            
                            self.logger.log_metrics(current_metrics)
                                
                    else:
                        percent = min(100, int((len(times) / (HR_WINDOW_SEC * 30)) * 100))
                        ui_status = f"Acquiring HR Signal... {percent}%"
                        ui_color = (0, 255, 255)
                    
                    last_calc_time = current_time

                # --- HUD Rendering ---
                display = mesh.draw_landmarks(frame, roi_mask)
                
                # Draw Status Bar
                cv2.rectangle(display, (10, 10), (600, 140), (0, 0, 0, 0.6), -1)
                cv2.putText(display, ui_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_color, 2)
                
                # Draw Metrics
                cv2.putText(display, f"HR (BPM): {current_metrics['hr']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"HRV (ms): {current_metrics['hrv']}", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"Breathing: {current_metrics['br']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("POS Medical Engine", display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()