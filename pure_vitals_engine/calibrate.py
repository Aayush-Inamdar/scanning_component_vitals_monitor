import cv2
import time
import numpy as np
import json
import os
from scipy import signal

from config.settings import CAMERA_INDEX, RESOLUTION_W, RESOLUTION_H, TARGET_FPS, HR_MIN_HZ, HR_MAX_HZ
from services.face_mesh_processor import FaceMeshProcessor
from services.pos_engine import extract_pos_signal

class VitalsCalibrator:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)
        self.calibration_data = [] 

    def record_calibration_block(self, reading_num):
        t_arr, r_arr, g_arr, b_arr = [], [], [], []
        
        with FaceMeshProcessor() as mesh:
            # --- PHASE 1: WAITING ---
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                cv2.putText(frame, f"READING {reading_num} OF 3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'S' on keyboard to START", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Calibration Engine", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'): break
            
            # --- PHASE 2: RECORDING ---
            start_time = time.time()
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                
                current_time = time.time() - start_time
                masked_frame, roi_mask, face_found, _ = mesh.process(frame, int(time.time() * 1000))
                
                if face_found:
                    b, g, r = cv2.split(frame)
                    valid_pixels = roi_mask > 0
                    if np.any(valid_pixels):
                        t_arr.append(current_time)
                        r_arr.append(np.mean(r[valid_pixels]))
                        g_arr.append(np.mean(g[valid_pixels]))
                        b_arr.append(np.mean(b[valid_pixels]))
                        
                cv2.putText(frame, f"RECORDING {reading_num} / 3", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Press 'Q' to Capture | Time: {current_time:.1f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Calibration Engine", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        # --- THE FIX: FULL PIPELINE NORMALIZATION ---
        pos_signal = extract_pos_signal(np.array(t_arr), np.array(r_arr), np.array(g_arr), np.array(b_arr), TARGET_FPS)
        clean_pulse = signal.detrend(pos_signal)
        
        b_filt, a_filt = signal.butter(3, [HR_MIN_HZ, HR_MAX_HZ], btype='bandpass', fs=TARGET_FPS)
        clean_pulse = signal.filtfilt(b_filt, a_filt, clean_pulse)
        
        # Digital Iron
        sg_opt_window = int(TARGET_FPS * 0.1) | 1
        clean_pulse = signal.savgol_filter(clean_pulse, window_length=max(5, sg_opt_window), polyorder=3)

        # Hilbert Envelope
        from scipy.signal import hilbert
        analytic_opt = hilbert(clean_pulse)
        envelope_opt = np.abs(analytic_opt)
        env_window = int(TARGET_FPS * 1.5) | 1
        envelope_smooth_opt = signal.savgol_filter(envelope_opt, window_length=max(5, env_window), polyorder=3)

        warmup_frames = int(TARGET_FPS * 1.5)
        if len(envelope_smooth_opt) > warmup_frames * 2:
            stable_median = np.median(envelope_smooth_opt[warmup_frames:-warmup_frames])
            envelope_smooth_opt[:warmup_frames] = stable_median
            envelope_smooth_opt[-warmup_frames:] = stable_median

        clean_pulse = clean_pulse / (envelope_smooth_opt + 1e-8)

        # Z-Score Normalization (This was the missing piece!)
        clean_pulse = (clean_pulse - np.mean(clean_pulse)) / (np.std(clean_pulse) + 1e-8)
        
        return clean_pulse

    def get_user_hr_input_cv2(self, reading_num):
        input_str = ""
        error_msg = ""
        
        while True:
            ui_frame = np.zeros((int(RESOLUTION_H), int(RESOLUTION_W), 3), dtype=np.uint8)
            ui_frame[:] = (30, 30, 30) 
            
            cv2.putText(ui_frame, f"READING {reading_num} SAVED", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(ui_frame, "Type the EXACT Heart Rate from your device:", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(ui_frame, input_str + "_", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            if error_msg:
                cv2.putText(ui_frame, error_msg, (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            cv2.putText(ui_frame, "Press ENTER to confirm", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.imshow("Calibration Engine", ui_frame)
            
            key = cv2.waitKey(0)
            if key == 13 or key == 10: 
                try:
                    return float(input_str)
                except ValueError:
                    error_msg = "Invalid number. Try again."
                    input_str = ""
            elif key == 8 or key == 127: 
                input_str = input_str[:-1]
                error_msg = ""
            elif 0 <= key < 256:
                char = chr(key)
                if char.isdigit() or char == '.':
                    input_str += char
                    error_msg = ""

    def run_grid_search(self):
        print("\n" + "="*50)
        print("INITIATING PARAMETER GRID SEARCH...")
        print("="*50)
        
        best_error = float('inf')
        best_params = {"prominence": 0.35, "distance_mod": 1.0}
        
        prominences = np.arange(0.10, 0.70, 0.05)
        distance_mods = np.arange(0.7, 1.3, 0.05) 
        
        total_combinations = len(prominences) * len(distance_mods)
        tested = 0
        
        for prom in prominences:
            for d_mod in distance_mods:
                tested += 1
                total_error = 0
                valid_blocks = 0
                
                for bvp_wave, true_hr in self.calibration_data:
                    skip_frames = int(TARGET_FPS * 5.0)
                    if len(bvp_wave) <= skip_frames + int(TARGET_FPS * 5.0): 
                        continue 
                    
                    search_pulse = bvp_wave[skip_frames:]
                    calc_distance = int(TARGET_FPS * (60.0 / 150.0) * d_mod) 
                    
                    peaks, _ = signal.find_peaks(search_pulse, distance=calc_distance, prominence=prom, wlen=int(TARGET_FPS*3))
                    
                    if len(peaks) >= 3:
                        duration_sec = len(search_pulse) / TARGET_FPS
                        hr_count = (len(peaks) / duration_sec) * 60.0
                        
                        raw_rr_intervals = np.diff(peaks) * (1000.0 / TARGET_FPS)
                        median_rr = np.median(raw_rr_intervals)
                        
                        if median_rr > 0:
                            hr_median = 60000.0 / median_rr
                            predicted_hr = (hr_count + hr_median) / 2.0 if abs(hr_count - hr_median) <= 5.0 else hr_median
                            
                            error = abs(predicted_hr - true_hr)
                            total_error += error
                            valid_blocks += 1
                
                # Require at least 2 out of 3 blocks to be mathematically valid
                if valid_blocks >= len(self.calibration_data) - 1:
                    avg_error = total_error / valid_blocks
                    if avg_error < best_error:
                        best_error = avg_error
                        best_params = {"prominence": round(prom, 2), "distance_mod": round(d_mod, 2)}
        
        print(f"\n[SUCCESS] Grid Search Complete. Analyzed {tested} combinations.")
        if best_error == float('inf'):
            print("[FAILED] Mathematical lock failed. Please ensure your face is well-lit.")
        else:
            print(f"Optimal Prominence   : {best_params['prominence']}")
            print(f"Optimal Distance Mod : {best_params['distance_mod']}")
            print(f"Mean Absolute Error  : +/- {best_error:.1f} BPM")
            
            os.makedirs("config", exist_ok=True)
            with open("config/user_profile.json", "w") as f:
                json.dump(best_params, f, indent=4)
            print("\n[SAVED] Personalized settings saved to config/user_profile.json")

    def run_calibration_sequence(self):
        print("Welcome to the Pure-Vitals Auto-Calibrator.")
        print("Please focus entirely on the video window. No terminal input is required.\n")
        
        for i in range(3): # REDUCED TO 3 READINGS
            bvp_wave = self.record_calibration_block(reading_num=i+1) 
            true_hr = self.get_user_hr_input_cv2(reading_num=i+1)
            self.calibration_data.append((bvp_wave, true_hr))
            
        cv2.destroyAllWindows()
        self.cap.release()
        self.run_grid_search()

if __name__ == "__main__":
    calibrator = VitalsCalibrator()
    calibrator.run_calibration_sequence()