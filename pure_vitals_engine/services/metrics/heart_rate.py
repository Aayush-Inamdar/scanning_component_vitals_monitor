import numpy as np
from scipy import signal

HR_MIN_HZ = 0.75  # 45 BPM
HR_MAX_HZ = 2.5   # 150 BPM

# Stores the history of peaks to ensure stability
peak_history = [] 

def calculate_hr(pos_signal, fps, previous_hr_bpm=None, lost_lock_counter=0, anchor_hr_bpm=None):
    global peak_history
    pos_detrended = signal.detrend(pos_signal)
    
    b_filt, a_filt = signal.butter(3, [HR_MIN_HZ, HR_MAX_HZ], btype='bandpass', fs=fps)
    clean_pulse = signal.filtfilt(b_filt, a_filt, pos_detrended)
    
    current_len = len(clean_pulse)
    safe_nfft = 16384 
    freqs, psd = signal.welch(clean_pulse, fs=fps, nperseg=current_len, nfft=safe_nfft)

    valid_mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]
    
    peaks, properties = signal.find_peaks(valid_psd, prominence=np.max(valid_psd) * 0.15)
    
    if len(peaks) == 0:
        return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, {"bpm": [], "power": []}

    candidates = valid_freqs[peaks] * 60.0
    prominences = properties['prominences']

    # --- THE PERSISTENCE LOGIC ---
    # We find the best peak, but we weight it against the previous second's peak.
    best_candidate = candidates[np.argmax(prominences)]
    peak_history.append(best_candidate)
    if len(peak_history) > 5: peak_history.pop(0)
    
    # Calculate a "Weighted Stability" HR
    stable_hr = np.median(peak_history)

    # ---------------------------------------------------------
    # ABSOLUTE ANCHOR LOGIC
    # ---------------------------------------------------------
    if anchor_hr_bpm is None:
        if 55.0 <= stable_hr <= 105.0 and not (59.0 < stable_hr < 61.0):
            return stable_hr, clean_pulse, 0, {"bpm": (valid_freqs * 60.0).tolist(), "power": valid_psd.tolist()}
        return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, {}

    # Tethering to the anchor
    if abs(stable_hr - anchor_hr_bpm) <= 5.0: # Wider tether for search, tighter for output
        return stable_hr, clean_pulse, 0, {}
    
    return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, {}