import numpy as np
from scipy import signal
from config.settings import BR_MIN_HZ, BR_MAX_HZ

def calculate_breathing(pos_signal, fps, previous_br=None):
    """Extracts Breathing Rate with Detrending and Strict Bandpass."""
    # Detrending kills the slow baseline camera shifts that confuse the breathing FFT
    pos_detrended = signal.detrend(pos_signal)
    
    b_filt, a_filt = signal.butter(3, [BR_MIN_HZ, BR_MAX_HZ], btype='bandpass', fs=fps)
    resp_wave = signal.filtfilt(b_filt, a_filt, pos_detrended)

    freqs, psd = signal.welch(
        resp_wave, 
        fs=fps, 
        nperseg=len(resp_wave),
        nfft=2048
    )
    
    valid_indices = np.where((freqs >= BR_MIN_HZ) & (freqs <= BR_MAX_HZ))[0]
    
    if len(valid_indices) == 0: 
        return previous_br
        
    peak_idx = valid_indices[np.argmax(psd[valid_indices])]
    raw_br = freqs[peak_idx] * 60.0
    
    if previous_br is not None and previous_br > 0:
        final_br = (0.85 * previous_br) + (0.15 * raw_br)
    else:
        final_br = raw_br
        
    return final_br