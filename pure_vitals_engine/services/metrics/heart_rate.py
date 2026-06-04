import numpy as np
from scipy import signal
from config.settings import HR_MIN_HZ, HR_MAX_HZ

def calculate_hr(pos_signal, fps, previous_hr_bpm=None, lost_lock_counter=0):
    pos_detrended = signal.detrend(pos_signal)
    b_filt, a_filt = signal.butter(3, [0.7, 3.0], btype='bandpass', fs=fps)
    clean_pulse = signal.filtfilt(b_filt, a_filt, pos_detrended)
    freqs, psd = signal.welch(clean_pulse, fs=fps, nperseg=len(clean_pulse), nfft=2048)

    # 1. FIND ALL POTENTIAL SPIKES (Candidates)
    peaks, properties = signal.find_peaks(psd, prominence=np.max(psd) * 0.15)
    valid_mask = (freqs[peaks] >= HR_MIN_HZ) & (freqs[peaks] <= HR_MAX_HZ)
    candidates = freqs[peaks][valid_mask] * 60.0

    # 2. TRACKING MODE (The "Safe" Path)
    if previous_hr_bpm and lost_lock_counter < 3:
        # Check if any candidate is within the +/- 8 BPM vault
        in_vault = candidates[(candidates >= previous_hr_bpm - 8) & (candidates <= previous_hr_bpm + 8)]
        if len(in_vault) > 0:
            raw_hr = in_vault[np.argmax(properties['prominences'][valid_mask][(candidates >= previous_hr_bpm - 8) & (candidates <= previous_hr_bpm + 8)])]
            final_hr = (0.9 * previous_hr_bpm) + (0.1 * raw_hr)
            return final_hr, clean_pulse, 0
        else:
            # Vault is empty, but don't panic yet. Hold for 3s.
            return previous_hr_bpm, clean_pulse, lost_lock_counter + 1

    # 3. HYPOTHESIS TESTING (The Startup Path)
    # If we have no lock, look at all candidates.
    # Return the one that is most prominent, BUT only if it is > 20% power.
    if len(candidates) > 0:
        best_candidate = candidates[np.argmax(properties['prominences'][valid_mask])]
        # Require 20% power to initialize
        if np.max(properties['prominences'][valid_mask]) > (np.max(psd) * 0.2):
            return best_candidate, clean_pulse, 0
            
    return previous_hr_bpm, clean_pulse, lost_lock_counter