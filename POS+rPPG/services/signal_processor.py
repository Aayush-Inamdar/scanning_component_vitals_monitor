import numpy as np
from config.settings import TARGET_FPS, HR_WINDOW_SEC, EXTENDED_WINDOW_SEC
from services.pos_engine import extract_pos_signal
from services.metrics.heart_rate import calculate_live_hr
from services.metrics.hrv import calculate_hrv
from services.metrics.breathing import calculate_breathing

class PosSignalProcessor:
    def __init__(self, fps=TARGET_FPS):
        self.fps = fps
        self.last_known_hr = None 
        self.anchor_hr = None       
        self.anchor_candidates = [] # THE VERIFICATION WAITING ROOM
        self.last_known_hrv = None  
        self.last_known_br = None 
        self.lost_lock_counter = 0 

    def process(self, t_raw, r_raw, g_raw, b_raw):
        results = {'hr': None, 'hrv': None, 'br': None}
        total_time = t_raw[-1] - t_raw[0]

        # 1. Fast Lane (10s)
        if total_time >= HR_WINDOW_SEC:
            fast_frames = int(HR_WINDOW_SEC * self.fps)
            pos_10s = extract_pos_signal(
                t_raw[-fast_frames:], r_raw[-fast_frames:], 
                g_raw[-fast_frames:], b_raw[-fast_frames:], self.fps
            )
            
            hr, _, self.lost_lock_counter, _ = calculate_live_hr(
                pos_10s, self.fps, self.last_known_hr, self.lost_lock_counter, self.anchor_hr
            )
            
            if hr:
                # --- THE VERIFICATION PROTOCOL ---
                if self.anchor_hr is None:
                    self.anchor_candidates.append(hr)
                    # Require 3 consecutive stable readings to prove it is a biological pulse
                    if len(self.anchor_candidates) >= 5:
                        recent = self.anchor_candidates[-5:]
                        if np.max(recent) - np.min(recent) <= 4.0: # Must be stable within 2.5 BPM
                            self.anchor_hr = np.mean(recent)
                            print(f"[ENGINE] Baseline Verified. Absolute Anchor Locked at: {self.anchor_hr:.1f} BPM")
                        else:
                            # Too much variance, pop the oldest and keep waiting
                            self.anchor_candidates.pop(0)
                self.last_known_hr = hr 
                results['hr'] = round(hr, 1)

        # 2. Slow Lane (30s Live / 5min Baseline)
        if total_time >= EXTENDED_WINDOW_SEC * 0.95: 
            pos_30s = extract_pos_signal(t_raw, r_raw, g_raw, b_raw, self.fps)
            
            hr_30s, clean_pulse_30s, _, _ = calculate_live_hr(
                pos_30s, self.fps, self.last_known_hr, self.lost_lock_counter, self.anchor_hr
            )
            
            if hr_30s:
                hrv = calculate_hrv(clean_pulse_30s, self.fps, hr_30s, self.last_known_hrv)
                if hrv: self.last_known_hrv = hrv  
                
                br = calculate_breathing(pos_30s, self.fps, self.last_known_br)
                if br: self.last_known_br = br
                
                results['hrv'] = round(hrv, 1) if hrv else None
                results['br'] = round(br, 1) if br else None

        return results

    def reset(self):
        self.last_known_hr = None
        self.anchor_hr = None
        self.anchor_candidates = []
        self.last_known_hrv = None 
        self.last_known_br = None
        self.lost_lock_counter = 0