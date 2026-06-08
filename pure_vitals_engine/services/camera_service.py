import cv2
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal

from config.settings import CAMERA_INDEX, RESOLUTION_W, RESOLUTION_H, BUFFER_DURATION_SEC, HR_WINDOW_SEC, EXTENDED_WINDOW_SEC, MOTION_THRESHOLD_PIXELS, COOLDOWN_SEC, TARGET_FPS, HR_MIN_HZ, HR_MAX_HZ
from services.face_mesh_processor import FaceMeshProcessor
from services.signal_processor import PosSignalProcessor
from services.data_logger import VitalsLogger
from services.pos_engine import extract_pos_signal

class VitalsCameraService:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_H)
        
        self.live_processor = PosSignalProcessor()
        self.baseline_processor = PosSignalProcessor()
        
        self.logger = VitalsLogger()
        self.hist_t, self.hist_r, self.hist_g, self.hist_b = [], [], [], []
        self.hist_bcg_y = [] 
        
        self.live_hr_history = []
        self.live_hr_times = []

    def run(self):
        print("Pure-Vitals-Engine Camera Service Active")
        print("[INFO] Press 'q' on the video window or Ctrl+C in terminal to stop and save graph.")
        
        times, r_buffer, g_buffer, b_buffer = [], [], [], []
        face_centers = [] 
        start_time = time.time()
        
        last_live_calc = 0
        last_baseline_calc = 0
        cooldown_until = 0 

        ui_status = "Initializing..."
        ui_color = (0, 255, 255)
        
        current_metrics = {'hr': '--', 'hrv': '--', 'br': '--'}
        baseline_metrics = {'hr': '--', 'hrv': '--', 'br': '--'}

        with FaceMeshProcessor() as mesh:
            while True:
                ret, frame = self.cap.read()
                if not ret: break
                current_time = time.time() - start_time

                masked_frame, roi_mask, face_found, nose_y = mesh.process(
                    frame, timestamp_ms=int(time.time() * 1000)
                )

                is_moving = False
                if face_found:
                    x, y, w, h = cv2.boundingRect(roi_mask)
                    face_centers.append((x + w/2, y + h/2))
                    
                    if len(face_centers) > 30:
                        face_centers.pop(0)
                        xs, ys = [c[0] for c in face_centers], [c[1] for c in face_centers]
                        if np.max(xs) - np.min(xs) > MOTION_THRESHOLD_PIXELS or np.max(ys) - np.min(ys) > MOTION_THRESHOLD_PIXELS:
                            is_moving = True
                            cooldown_until = current_time + COOLDOWN_SEC 

                if face_found and current_time > cooldown_until:
                    b_channel, g_channel, r_channel = cv2.split(frame)
                    valid_pixels = roi_mask > 0
                    if np.any(valid_pixels):
                        mean_r = np.mean(r_channel[valid_pixels])
                        mean_g = np.mean(g_channel[valid_pixels])
                        mean_b = np.mean(b_channel[valid_pixels])
                        
                        times.append(current_time)
                        r_buffer.append(mean_r)
                        g_buffer.append(mean_g)
                        b_buffer.append(mean_b)
                        
                        self.hist_t.append(current_time)
                        self.hist_r.append(mean_r)
                        self.hist_g.append(mean_g)
                        self.hist_b.append(mean_b)
                        
                        if nose_y is not None:
                            self.hist_bcg_y.append(nose_y)
                        else:
                            self.hist_bcg_y.append(self.hist_bcg_y[-1] if len(self.hist_bcg_y) > 0 else 0)
                
                elif is_moving:
                    times.clear(); r_buffer.clear(); g_buffer.clear(); b_buffer.clear()
                    self.live_processor.reset()
                    ui_status = "MOTION DETECTED! Resetting..."
                    ui_color = (0, 0, 255)
                    current_metrics = {'hr': '--', 'hrv': '--', 'br': '--'}

                while len(times) > 0 and times[0] < current_time - BUFFER_DURATION_SEC:
                    times.pop(0); r_buffer.pop(0); g_buffer.pop(0); b_buffer.pop(0)

                if current_time - last_live_calc > 1.0:
                    if current_time <= cooldown_until:
                        pass
                    elif len(times) > (HR_WINDOW_SEC * 15):
                        results = self.live_processor.process(np.array(times), np.array(r_buffer), np.array(g_buffer), np.array(b_buffer))
                        if results['hr']:
                            current_metrics['hr'] = results['hr']
                            self.live_hr_history.append(results['hr'])
                            self.live_hr_times.append(current_time)
                            
                            if results['hrv']:
                                current_metrics['hrv'] = results['hrv']
                            if results['br']:
                                current_metrics['br'] = results['br']
                                
                            if current_metrics['hrv'] != '--' and current_metrics['br'] != '--':
                                ui_status = "Diagnostics Locked & Tracking"
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
                    last_live_calc = current_time

                if current_time - last_baseline_calc > 10.0:
                    if len(self.hist_t) > (EXTENDED_WINDOW_SEC * TARGET_FPS):
                        max_frames = int(TARGET_FPS * 300)
                        
                        b_times = np.array(self.hist_t[-max_frames:])
                        b_r = np.array(self.hist_r[-max_frames:])
                        b_g = np.array(self.hist_g[-max_frames:])
                        b_b = np.array(self.hist_b[-max_frames:])
                        
                        b_results = self.baseline_processor.process(b_times, b_r, b_g, b_b)
                        
                        if b_results.get('hr'): baseline_metrics['hr'] = b_results['hr']
                        if b_results.get('hrv'): baseline_metrics['hrv'] = b_results['hrv']
                        if b_results.get('br'): baseline_metrics['br'] = b_results['br']
                        
                    last_baseline_calc = current_time

                display = mesh.draw_landmarks(frame, roi_mask)
                cv2.rectangle(display, (10, 10), (600, 160), (0, 0, 0, 0.7), -1)
                cv2.putText(display, ui_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_color, 2)
                
                if isinstance(current_metrics['hr'], float):
                    hr_disp = f"{current_metrics['hr']:.1f}"
                else:
                    hr_disp = current_metrics['hr']
                    
                if isinstance(baseline_metrics['hr'], float):
                    b_hr_disp = f"{baseline_metrics['hr']:.1f}"
                else:
                    b_hr_disp = baseline_metrics['hr']
                    
                cv2.putText(display, f"HR  (BPM): {hr_disp:>5} (Live) | {b_hr_disp:>5} (Avg)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, f"HRV  (ms): {current_metrics['hrv']:>5} (Live) | {baseline_metrics['hrv']:>5} (True)", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(display, f"Breathing: {current_metrics['br']:>5} (Live) | {baseline_metrics['br']:>5} (Avg)", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                
                cv2.imshow("POS Medical Engine", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.generate_and_save_graph()
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def generate_and_save_graph(self):
        if len(self.hist_t) < int(TARGET_FPS * 10):
            print("[!] Not enough data to generate graph.")
            return

        print("[INFO] Generating diagnostic reports...")

        t_arr = np.array(self.hist_t)
        r_arr = np.array(self.hist_r)
        g_arr = np.array(self.hist_g)
        b_arr = np.array(self.hist_b)
        bcg_arr = np.array(self.hist_bcg_y)

        t_axis = np.linspace(0, t_arr[-1] - t_arr[0], len(t_arr))
        total_time = int(t_arr[-1] - t_arr[0])
        final_hr = f"{self.live_hr_history[-1]:.1f}" if len(self.live_hr_history) > 0 else "--"

        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # -------------------------------------------------------------
        # GRAPH 1: THE RAW RGB TIME SERIES
        # -------------------------------------------------------------
        plt.style.use('default')
        fig_rgb, ax_rgb = plt.subplots(figsize=(10, 4))
        fig_rgb.patch.set_facecolor('white')
        ax_rgb.set_facecolor('white')

        ax_rgb.plot(t_axis, r_arr, color='#ef4444', linewidth=0.8, label='Red Channel', alpha=0.9)
        ax_rgb.plot(t_axis, g_arr, color='#22c55e', linewidth=0.8, label='Green Channel', alpha=0.9)
        ax_rgb.plot(t_axis, b_arr, color='#3b82f6', linewidth=0.8, label='Blue Channel', alpha=0.9)
        
        ax_rgb.set_title(f"Raw RGB Intensity Signals | Duration: {total_time}s", color='black', fontsize=12, fontweight='bold', pad=10)
        ax_rgb.set_xlabel("Time (s)", color='black', fontsize=10)
        ax_rgb.set_ylabel("Mean Pixel Intensity", color='black', fontsize=10)
        
        ax_rgb.spines['top'].set_visible(False)
        ax_rgb.spines['right'].set_visible(False)
        ax_rgb.tick_params(colors='black')
        
        legend = ax_rgb.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')
        for text in legend.get_texts(): text.set_color("black")

        rgb_save_path = f"data/desktop_raw_rgb_signal_{timestamp}.png"
        plt.savefig(rgb_save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_rgb)

        # -------------------------------------------------------------
        # GRAPH 2: THE OSCILLATORY BVP TRACE (rPPG / Optical) - EVEN & PLAIN
        # -------------------------------------------------------------
        pos_signal = extract_pos_signal(t_arr, r_arr, g_arr, b_arr, TARGET_FPS)
        clean_pulse = signal.detrend(pos_signal)

        # 1. Bandpass Filter
        b_filt_opt, a_filt_opt = signal.butter(3, [0.8, 2.0], btype='bandpass', fs=TARGET_FPS)
        clean_pulse = signal.filtfilt(b_filt_opt, a_filt_opt, clean_pulse)
        
        # 2. The Digital Iron (Savitzky-Golay)
        sg_opt_window = int(TARGET_FPS * 0.1) | 1
        clean_pulse = signal.savgol_filter(clean_pulse, window_length=max(5, sg_opt_window), polyorder=3)

        # 3. Hilbert Envelope Normalization
        from scipy.signal import hilbert
        analytic_opt = hilbert(clean_pulse)
        envelope_opt = np.abs(analytic_opt)
        
        env_window = int(TARGET_FPS * 1.5) | 1
        envelope_smooth_opt = signal.savgol_filter(envelope_opt, window_length=max(5, env_window), polyorder=3)

        # --- THE FIX: EDGE ARTIFACT CLAMPING ---
        # Prevents the Hilbert transform from exploding the first and last beats
        warmup_frames = int(TARGET_FPS * 1.5)
        if len(envelope_smooth_opt) > warmup_frames * 2:
            stable_median = np.median(envelope_smooth_opt[warmup_frames:-warmup_frames])
            envelope_smooth_opt[:warmup_frames] = stable_median
            envelope_smooth_opt[-warmup_frames:] = stable_median

        clean_pulse = clean_pulse / (envelope_smooth_opt + 1e-8)

        # 4. Standard Z-Score mapping for plotting
        clean_pulse = (clean_pulse - np.mean(clean_pulse)) / (np.std(clean_pulse) + 1e-8)

        bvp_t_axis = np.linspace(0, total_time, len(clean_pulse))

        fig_bvp, ax_bvp = plt.subplots(figsize=(10, 4))
        fig_bvp.patch.set_facecolor('white')
        ax_bvp.set_facecolor('white')

        ax_bvp.plot(bvp_t_axis, clean_pulse, color='#e8633a', linewidth=0.8)
        
        y_max_bvp = np.max(np.abs(clean_pulse)) * 1.15
        ax_bvp.set_ylim(-y_max_bvp, y_max_bvp)
        
        ax_bvp.set_title(f"Oscillatory rPPG Trace (Optical) | Duration: {total_time}s | Final HR: {final_hr} BPM", color='black', fontsize=12, fontweight='bold', pad=10)
        ax_bvp.set_xlabel("Time (s)", color='black', fontsize=10)
        ax_bvp.set_ylabel("Amplitude (Uniform Z-Score)", color='black', fontsize=10)
        
        ax_bvp.spines['top'].set_visible(False)
        ax_bvp.spines['right'].set_visible(False)
        ax_bvp.tick_params(colors='black')

        bvp_save_path = f"data/desktop_bvp_signal_{timestamp}.png"
        plt.savefig(bvp_save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_bvp)
        
        # -------------------------------------------------------------
        # GRAPH 3: THE BALLISTOCARDIOGRAM (BCG / Mechanical)
        # -------------------------------------------------------------
        clean_bcg = signal.detrend(bcg_arr)
        
        if len(clean_bcg) > 11:
            clean_bcg = signal.savgol_filter(clean_bcg, window_length=11, polyorder=3)
        
        b_filt_bcg, a_filt_bcg = signal.butter(6, [0.8, 8.0], btype='bandpass', fs=TARGET_FPS)
        clean_bcg = signal.filtfilt(b_filt_bcg, a_filt_bcg, clean_bcg)

        envelope_bcg = np.abs(hilbert(clean_bcg))
        envelope_smooth_bcg = signal.savgol_filter(envelope_bcg, 
            window_length=max(5, int(TARGET_FPS * 2) | 1),
            polyorder=3)

        # --- THE FIX: EDGE ARTIFACT CLAMPING (BCG) ---
        if len(envelope_smooth_bcg) > warmup_frames * 2:
            stable_median_bcg = np.median(envelope_smooth_bcg[warmup_frames:-warmup_frames])
            envelope_smooth_bcg[:warmup_frames] = stable_median_bcg
            envelope_smooth_bcg[-warmup_frames:] = stable_median_bcg

        clean_bcg = clean_bcg / (envelope_smooth_bcg + 1e-8)

        energy_window = int(TARGET_FPS * 1.5)
        local_energy = np.convolve(np.abs(clean_bcg), np.ones(energy_window)/energy_window, mode='same')
        
        median_energy = np.median(local_energy)
        mad_energy = np.median(np.abs(local_energy - median_energy))
        threshold_energy = median_energy + (4.0 * mad_energy) 
        
        attenuation = np.where(local_energy > threshold_energy, threshold_energy / (local_energy + 1e-8), 1.0)
        
        mask_window = int(TARGET_FPS * 1.5)
        mask_window = mask_window if mask_window % 2 != 0 else mask_window + 1
        smooth_mask = signal.savgol_filter(attenuation, max(5, mask_window), 3)
        clean_bcg = clean_bcg * smooth_mask

        median_bcg = np.median(clean_bcg)
        mad_bcg = np.median(np.abs(clean_bcg - median_bcg))
        robust_std_bcg = 1.4826 * mad_bcg + 1e-8
        clean_bcg = (clean_bcg - median_bcg) / robust_std_bcg

        fig_bcg, ax_bcg = plt.subplots(figsize=(10, 4))
        fig_bcg.patch.set_facecolor('white')
        ax_bcg.set_facecolor('white')

        ax_bcg.plot(t_axis, clean_bcg, color='#9333ea', linewidth=0.8)
        
        y_max_bcg = np.max(np.abs(clean_bcg)) * 1.15
        ax_bcg.set_ylim(-y_max_bcg, y_max_bcg)
        
        ax_bcg.set_title(f"Ballistocardiogram Trace (Mechanical) | Duration: {total_time}s", color='black', fontsize=12, fontweight='bold', pad=10)
        ax_bcg.set_xlabel("Time (s)", color='black', fontsize=10)
        ax_bcg.set_ylabel("Micro-Displacement (Robust Z-Score)", color='black', fontsize=10)
        
        ax_bcg.spines['top'].set_visible(False)
        ax_bcg.spines['right'].set_visible(False)
        ax_bcg.tick_params(colors='black')

        bcg_save_path = f"data/desktop_bcg_signal_{timestamp}.png"
        plt.savefig(bcg_save_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close(fig_bcg)
        print(f"[SUCCESS] Saved Mechanical BCG graph to: {bcg_save_path}")

        # -------------------------------------------------------------
        # GRAPH 4: THE HEARTBEAT INTERVAL TACHOGRAM (RR Plot)
        # -------------------------------------------------------------
        min_dist_frames = int(TARGET_FPS * (60.0 / 150.0))
        peaks, _ = signal.find_peaks(clean_pulse, distance=min_dist_frames, prominence=0.25)

        if len(peaks) > 1:
            raw_rr_intervals = np.diff(peaks) * (1000.0 / TARGET_FPS)
            
            median_rr = np.median(raw_rr_intervals)
            valid_mask = (raw_rr_intervals >= median_rr * 0.75) & (raw_rr_intervals <= median_rr * 1.25)
            clean_rr_intervals = raw_rr_intervals[valid_mask]
            
            if len(clean_rr_intervals) > 0:
                beat_numbers = np.arange(1, len(clean_rr_intervals) + 1)
                
                fig_tacho, ax_tacho = plt.subplots(figsize=(10, 4))
                fig_tacho.patch.set_facecolor('white')
                ax_tacho.set_facecolor('white')

                ax_tacho.plot(beat_numbers, clean_rr_intervals, color='#14b8a6', linewidth=1.0, marker='o', markersize=3)
                
                mean_rr = np.mean(clean_rr_intervals)
                ax_tacho.set_title(f"Heartbeat Intervals Tachogram | Mean RR: {mean_rr:.0f} ms", color='black', fontsize=12, fontweight='bold', pad=10)
                ax_tacho.set_xlabel("Beat Number", color='black', fontsize=10)
                ax_tacho.set_ylabel("Interval Duration (ms)", color='black', fontsize=10)
                
                ax_tacho.set_ylim(max(400, min(clean_rr_intervals)-50), min(1500, max(clean_rr_intervals)+50))
                
                ax_tacho.spines['top'].set_visible(False)
                ax_tacho.spines['right'].set_visible(False)
                ax_tacho.tick_params(colors='black')
                ax_tacho.grid(True, linestyle='--', alpha=0.3)

                tacho_save_path = f"data/desktop_tachogram_{timestamp}.png"
                plt.savefig(tacho_save_path, bbox_inches='tight', facecolor='white', dpi=300)
                plt.close(fig_tacho)
        
        print("[SUCCESS] All diagnostic reports generated.")