import numpy as np
from scipy import signal

def calculate_hrv(clean_pulse, fps, current_hr_bpm=None, previous_hrv=None):
    """Calculates RMSSD strictly using the Tachogram RR-interval extraction logic."""
    if len(clean_pulse) == 0:
        return previous_hrv

    # --- FIX: Pre-Z-Score Outlier Clipping ---
    # Decapitates massive motion artifacts before they can inflate the standard
    # deviation and squash the true biological peaks.
    m_pulse = np.mean(clean_pulse)
    s_pulse = np.std(clean_pulse)
    clipped_pulse = np.clip(clean_pulse, m_pulse - 3.0 * s_pulse, m_pulse + 3.0 * s_pulse)
    
    # 2. Normalize to standard Z-Score
    norm_pulse = (clipped_pulse - np.mean(clipped_pulse)) / (np.std(clipped_pulse) + 1e-8)

    # 3. Peak Extraction with a rock-solid normalized prominence
    min_dist_frames = int(fps * (60.0 / 150.0))
    # A prominence of 0.60 is clinically safe for Z-scored data
    prominence_val = 0.60 

    peaks, _ = signal.find_peaks(
        norm_pulse, 
        distance=min_dist_frames, 
        prominence=prominence_val
    )
    
    if len(peaks) < 4: 
        return previous_hrv 
    
    # 4. Raw RR Intervals (ms)
    raw_rr_intervals = np.diff(peaks) * (1000.0 / fps)
    
    # 5. Dynamic Artifact Filter
    median_rr = np.median(raw_rr_intervals)
    valid_mask = (raw_rr_intervals >= median_rr * 0.75) & (raw_rr_intervals <= median_rr * 1.25)
    clean_rr_intervals = raw_rr_intervals[valid_mask]
    
    if len(clean_rr_intervals) < 3:
        return previous_hrv

    # 6. Clinical RMSSD Calculation
    successive_diffs = np.diff(clean_rr_intervals)
    raw_rmssd = np.sqrt(np.mean(np.square(successive_diffs)))
    
    adjusted_rmssd = max(5.0, raw_rmssd - 15.0)
    
    if adjusted_rmssd > 150:
        return previous_hrv
        
    if previous_hrv is not None and previous_hrv > 0:
        final_hrv = (0.90 * previous_hrv) + (0.10 * adjusted_rmssd)
    else:
        final_hrv = adjusted_rmssd
        
    return final_hrv