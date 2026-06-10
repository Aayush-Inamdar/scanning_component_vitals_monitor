import numpy as np
from scipy import signal

def calculate_live_hr(pos_signal, fps, last_known_hr, lost_lock_counter, anchor_hr):
    """
    Live Frequency-Domain Heart Rate Extraction.
    Receives a 1D optical signal (pos_10s) and outputs 4 variables to match signal_processor.
    """
    if len(pos_signal) < int(fps * 2):
        return None, None, lost_lock_counter, None

    # ---------------------------------------------------------
    # 1. LIVE DSP PIPELINE 
    # ---------------------------------------------------------
    # Base Detrend
    clean_pulse = signal.detrend(pos_signal)
    
    # Order 3, 0.75-2.5 Hz Bandpass (No Savitzky-Golay, No Hilbert)
    b_filt, a_filt = signal.butter(3, [0.75, 2.5], btype='bandpass', fs=fps)
    clean_pulse = signal.filtfilt(b_filt, a_filt, clean_pulse)
    
    # Uniform Z-Score Normalization
    clean_pulse = (clean_pulse - np.mean(clean_pulse)) / (np.std(clean_pulse) + 1e-8)
    
    # ---------------------------------------------------------
    # 2. WELCH'S POWER SPECTRAL DENSITY (FFT)
    # ---------------------------------------------------------
    nperseg = min(len(clean_pulse), int(fps * 10)) 
    
    f, pxx = signal.welch(clean_pulse, fs=fps, nperseg=nperseg, nfft=16384, scaling='spectrum')
    bpm_freqs = f * 60.0
    
    # ---------------------------------------------------------
    # 3. THE 1 HZ ALIAS BLINDSPOT 
    # ---------------------------------------------------------
    valid_idx = np.where(
        (bpm_freqs >= 45.0) & (bpm_freqs <= 150.0) & 
        ~((bpm_freqs >= 59.0) & (bpm_freqs <= 61.0))
    )[0]
    
    if len(valid_idx) == 0:
        return None, None, lost_lock_counter + 1, None
        
    valid_f = bpm_freqs[valid_idx]
    valid_p = pxx[valid_idx]
    
    max_idx = np.argmax(valid_p)
    raw_peak_bpm = valid_f[max_idx]

    if raw_peak_bpm < 65.0:
        double_target = raw_peak_bpm * 2.0
        double_idx = np.argmin(np.abs(valid_f - double_target))
        
        # If the upper frequency has at least 35% of the sub-harmonic's power, it's the real HR
        if valid_p[double_idx] > (0.35 * valid_p[max_idx]):
            max_idx = double_idx
            raw_peak_bpm = valid_f[max_idx]
    
    # ---------------------------------------------------------
    # 4. PARABOLIC PEAK INTERPOLATION
    # ---------------------------------------------------------
    if 0 < max_idx < len(valid_p) - 1:
        alpha = valid_p[max_idx - 1]
        beta = valid_p[max_idx]
        gamma = valid_p[max_idx + 1]
        
        denominator = (alpha - 2 * beta + gamma)
        if denominator != 0:
            shift = 0.5 * (alpha - gamma) / denominator
            bin_width = valid_f[1] - valid_f[0]
            true_peak_bpm = raw_peak_bpm + (shift * bin_width)
        else:
            true_peak_bpm = raw_peak_bpm
    else:
        true_peak_bpm = raw_peak_bpm
        
    # ---------------------------------------------------------
    # 5. LIVE VAULT CLAMP (Prevents harmonic drift in the FFT)
    # ---------------------------------------------------------
    if anchor_hr is not None:
        if abs(true_peak_bpm - anchor_hr) > 10.0:
            # If the FFT jumps to a wild harmonic, reject it and return the last safe reading
            return last_known_hr, None, lost_lock_counter + 1, None
            
    if 40.0 <= true_peak_bpm <= 150.0:
        return float(true_peak_bpm), None, 0, None
    else:
        return None, None, lost_lock_counter + 1, None