import numpy as np
from scipy import signal

HR_MIN_HZ = 0.75  # 45 BPM
HR_MAX_HZ = 2.5   # 150 BPM

def calculate_hr(pos_signal, fps, previous_hr_bpm=None, lost_lock_counter=0, anchor_hr_bpm=None):
    pos_detrended = signal.detrend(pos_signal)
    
    b_filt, a_filt = signal.butter(3, [HR_MIN_HZ, HR_MAX_HZ], btype='bandpass', fs=fps)
    clean_pulse = signal.filtfilt(b_filt, a_filt, pos_detrended)
    
    current_len = len(clean_pulse)
    safe_nfft = max(2048, current_len)
    freqs, psd = signal.welch(clean_pulse, fs=fps, nperseg=current_len, nfft=safe_nfft)

    peaks, properties = signal.find_peaks(psd, prominence=np.max(psd) * 0.15)
    valid_mask = (freqs[peaks] >= HR_MIN_HZ) & (freqs[peaks] <= HR_MAX_HZ)
    
    candidates = freqs[peaks][valid_mask] * 60.0
    prominences = properties['prominences'][valid_mask]

    graph_mask = (freqs >= HR_MIN_HZ) & (freqs <= HR_MAX_HZ)
    spectrum = {
        "bpm": (freqs[graph_mask] * 60.0).tolist(),
        "power": psd[graph_mask].tolist()
    }

    if len(candidates) == 0:
        return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, spectrum

    # ---------------------------------------------------------
    # 1. STRICT INITIAL LOCK (With 1 Hz Alias Blindspot)
    # ---------------------------------------------------------
    if anchor_hr_bpm is None:
        sorted_indices = np.argsort(prominences)[::-1]
        for idx in sorted_indices:
            candidate_hr = candidates[idx]
            candidate_power = prominences[idx]
            
            if 55.0 <= candidate_hr <= 105.0:
                # THE FIX: 60 BPM is exactly 1.0 Hz (A classic webcam/UI stutter alias).
                # We mathematically blind the engine to 59.0 - 61.0 ONLY during the initial lock.
                if not (59.0 < candidate_hr < 61.0):
                    if candidate_power > (np.max(psd) * 0.40): # Increased strictness to 40%
                        return candidate_hr, clean_pulse, 0, spectrum
        
        return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, spectrum

    # ---------------------------------------------------------
    # 2. THE ABSOLUTE ANCHOR (+/- 2.5 BPM)
    # ---------------------------------------------------------
    valid_jump_mask = (candidates >= anchor_hr_bpm - 2.5) & (candidates <= anchor_hr_bpm + 2.5)
    tethered_candidates = candidates[valid_jump_mask]
    tethered_prominences = prominences[valid_jump_mask]
    
    if len(tethered_candidates) > 0:
        best_idx = np.argmax(tethered_prominences)
        raw_hr = tethered_candidates[best_idx]
        
        if abs(raw_hr - previous_hr_bpm) <= 0.5:
            return previous_hr_bpm, clean_pulse, 0, spectrum
            
        final_hr = (0.90 * previous_hr_bpm) + (0.10 * raw_hr)
        return final_hr, clean_pulse, 0, spectrum
    
    # 3. COASTING MODE 
    return previous_hr_bpm, clean_pulse, lost_lock_counter + 1, spectrum