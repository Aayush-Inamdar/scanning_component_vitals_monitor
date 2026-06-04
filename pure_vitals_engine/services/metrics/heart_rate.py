import numpy as np
from scipy import signal

HR_MIN_HZ = 0.75  # 45 BPM
HR_MAX_HZ = 2.5   # 150 BPM

def calculate_hr(pos_signal, fps, previous_hr_bpm=None, lost_lock_counter=0):
    pos_detrended = signal.detrend(pos_signal)
    
    # Tight bandpass to kill low-end movement noise
    b_filt, a_filt = signal.butter(3, [0.75, 2.5], btype='bandpass', fs=fps)
    clean_pulse = signal.filtfilt(b_filt, a_filt, pos_detrended)
    freqs, psd = signal.welch(clean_pulse, fs=fps, nperseg=len(clean_pulse), nfft=2048)

    peaks, properties = signal.find_peaks(psd, prominence=np.max(psd) * 0.15)
    valid_mask = (freqs[peaks] >= HR_MIN_HZ) & (freqs[peaks] <= HR_MAX_HZ)
    
    candidates = freqs[peaks][valid_mask] * 60.0
    prominences = properties['prominences'][valid_mask]

    # 1. TRACKING MODE (The Titanium Vault: +/- 8 BPM)
    if previous_hr_bpm and lost_lock_counter < 3:
        vault_mask = (candidates >= previous_hr_bpm - 8) & (candidates <= previous_hr_bpm + 8)
        in_vault = candidates[vault_mask]
        
        if len(in_vault) > 0:
            vault_prominences = prominences[vault_mask]
            max_inside_power = np.max(vault_prominences)
            max_overall_power = np.max(prominences) if len(prominences) > 0 else 0
            
            # --- THE OVERRIDE PROTOCOL ---
            # If a peak outside the vault is TWICE as strong as the one inside, 
            # we are tracking a ghost. Break the vault and let recovery mode grab the real pulse!
            if max_overall_power > (max_inside_power * 2.0):
                pass # Skip the vault logic, fall through to Startup/Recovery
            else:
                # Normal Vault Tracking (Safe and Stable)
                raw_hr = in_vault[np.argmax(vault_prominences)]
                final_hr = (0.9 * previous_hr_bpm) + (0.1 * raw_hr)
                return final_hr, clean_pulse, 0
        else:
            return previous_hr_bpm, clean_pulse, lost_lock_counter + 1

    # 2. STARTUP / RECOVERY MODE 
    if len(candidates) > 0:
        sorted_indices = np.argsort(prominences)[::-1]
        
        for idx in sorted_indices:
            candidate_hr = candidates[idx]
            candidate_power = prominences[idx]
            
            # Standard, proven 20% threshold
            if candidate_power > (np.max(psd) * 0.20):
                
                # --- BIOLOGICAL PLAUSIBILITY CHECK ---
                # Prevent instant drops to the 49 BPM lighting ghost
                # --- FIXED: SMART PLAUSIBILITY & HARMONIC REJECTION ---
                if previous_hr_bpm is not None:
                    # 1. Guard against the sudden 50Hz lighting drop
                    if candidate_hr < (previous_hr_bpm - 15.0):
                        continue 
                        
                    # 2. Guard against the 2x Harmonic Echo
                    # If the new peak is roughly double the current HR, it is mathematically a ghost.
                    if abs(candidate_hr - (previous_hr_bpm * 2.0)) < 15.0:
                        continue 
                
                return candidate_hr, clean_pulse, 0
            
    return previous_hr_bpm, clean_pulse, lost_lock_counter