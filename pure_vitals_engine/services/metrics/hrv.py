import numpy as np
from scipy import signal

def calculate_hrv(clean_pulse, fps, current_hr_bpm, previous_hrv=None):
    """Calculates RMSSD with Dicrotic Notch Lockout and Display Inertia."""
    if not current_hr_bpm or current_hr_bpm <= 0:
        return previous_hrv 

    # 1. The Dicrotic Notch Lockout (Expect a beat based on current HR)
    expected_gap_sec = 60.0 / current_hr_bpm
    min_distance_frames = int((expected_gap_sec * 0.65) * fps)
    
    # --- THE FIX: Robust Amplitude Estimation ---
    # Using percentiles ignores massive sudden camera glitches that 
    # would otherwise ruin the np.max() calculation.
    top_pulse = np.percentile(clean_pulse, 95)
    bottom_pulse = np.percentile(clean_pulse, 5)
    pulse_amplitude = top_pulse - bottom_pulse
    
    # We can safely lower this to 15% because the min_distance_frames 
    # is already doing 90% of the heavy lifting to prevent double-counting.
    min_prominence = pulse_amplitude * 0.15

    peaks, _ = signal.find_peaks(
        clean_pulse, 
        distance=min_distance_frames,
        prominence=min_prominence
    )
    
    # We need at least a few heartbeats to calculate variability
    if len(peaks) < 4: 
        return previous_hrv 
    
    # Calculate intervals in milliseconds
    rr_intervals = np.diff(peaks) / fps * 1000.0
    
    # 3. Medical Outlier Rejection
    median_rr = np.median(rr_intervals)
    
    # Widened to 0.75 / 1.25 to accommodate healthy Respiratory Sinus Arrhythmia
    valid_rr = rr_intervals[
        (rr_intervals > median_rr * 0.75) & 
        (rr_intervals < median_rr * 1.25)
    ]
    
    if len(valid_rr) < 3:
        return previous_hrv

    # 4. RMSSD Calculation (The clinical standard for short-term HRV)
    successive_diffs = np.diff(valid_rr)
    raw_rmssd = np.sqrt(np.mean(np.square(successive_diffs)))
    
    # 5. The 33ms Webcam Penalty Floor
    adjusted_rmssd = max(10.0, raw_rmssd - 15.0)
    
    # If the variance is impossibly high, it's noise. Reject it.
    if adjusted_rmssd > 130:
        return previous_hrv
        
    # 6. HRV Inertia
    # We sped up the convergence to 0.20 so the UI populates faster 
    # once a valid signal is acquired.
    if previous_hrv is not None and previous_hrv > 0:
        final_hrv = (0.80 * previous_hrv) + (0.20 * adjusted_rmssd)
    else:
        final_hrv = adjusted_rmssd
        
    return final_hrv