import numpy as np
from scipy import signal

def calculate_hrv(clean_pulse, fps, current_hr_bpm, previous_hrv=None):
    """Calculates RMSSD with Dicrotic Notch Lockout and Display Inertia."""
    if not current_hr_bpm or current_hr_bpm <= 0:
        return previous_hrv # Return memory if HR is invalid

    # 1. The Dicrotic Notch Lockout 
    expected_gap_sec = 60.0 / current_hr_bpm
    min_distance_frames = int((expected_gap_sec * 0.65) * fps)
    
    # 2. Adaptive Prominence
    pulse_amplitude = np.max(clean_pulse) - np.min(clean_pulse)
    min_prominence = pulse_amplitude * 0.40

    peaks, _ = signal.find_peaks(
        clean_pulse, 
        distance=min_distance_frames,
        prominence=min_prominence
    )
    
    if len(peaks) < 4: 
        return previous_hrv 
    
    # Calculate intervals in milliseconds
    rr_intervals = np.diff(peaks) / fps * 1000.0
    
    # 3. Medical Outlier Rejection
    median_rr = np.median(rr_intervals)
    valid_rr = rr_intervals[
        (rr_intervals > median_rr * 0.8) & 
        (rr_intervals < median_rr * 1.2)
    ]
    
    if len(valid_rr) < 3:
        return previous_hrv

    # 4. RMSSD Calculation
    successive_diffs = np.diff(valid_rr)
    raw_rmssd = np.sqrt(np.mean(np.square(successive_diffs)))
    
    # 5. The 33ms Webcam Penalty Floor
    adjusted_rmssd = max(10.0, raw_rmssd - 15.0)
    
    if adjusted_rmssd > 120:
        return previous_hrv
        
    # --- THE FIX: HRV INERTIA ---
    # HRV is a slow-changing biological metric. We apply heavy smoothing 
    # to kill the violent 70ms camera-induced snaps.
    if previous_hrv is not None and previous_hrv > 0:
        final_hrv = (0.90 * previous_hrv) + (0.10 * adjusted_rmssd)
    else:
        final_hrv = adjusted_rmssd
        
    return final_hrv