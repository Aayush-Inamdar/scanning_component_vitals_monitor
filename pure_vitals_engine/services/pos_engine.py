import numpy as np
from scipy import signal

def extract_pos_signal(t, r, g, b, fps):
    """The Overlap-Add POS Engine (With Outlier Clipping)"""
    t_even = np.arange(t[0], t[-1], 1.0 / fps)
    r_even = signal.medfilt(np.interp(t_even, t, r), 5)
    g_even = signal.medfilt(np.interp(t_even, t, g), 5)
    b_even = signal.medfilt(np.interp(t_even, t, b), 5)

    window_length = int(fps * 1.6)
    pos_signal = np.zeros(len(r_even))
    window_overlap = np.zeros(len(r_even))
    
    for i in range(len(r_even) - window_length):
        r_win = r_even[i:i+window_length]
        g_win = g_even[i:i+window_length]
        b_win = b_even[i:i+window_length]
        
        r_norm = r_win / (np.mean(r_win) + 1e-8)
        g_norm = g_win / (np.mean(g_win) + 1e-8)
        b_norm = b_win / (np.mean(b_win) + 1e-8)
        
        X = 3 * r_norm - 2 * g_norm
        Y = 1.5 * r_norm + g_norm - 1.5 * b_norm
        alpha = (np.std(X) + 1e-8) / (np.std(Y) + 1e-8)
        
        H = X - (alpha * Y)
        H = H - np.mean(H)
        
        pos_signal[i:i+window_length] += H
        window_overlap[i:i+window_length] += 1

    pos_signal = pos_signal / (window_overlap + 1e-8)
    pos_signal = pos_signal[window_length:-window_length]
    
    # --- THE FIX: Standard Deviation Spike Clipper ---
    # Shaves off violent camera auto-exposure jumps to stop filter ringing
    std_val = np.std(pos_signal)
    if std_val > 0:
        pos_signal = np.clip(pos_signal, -3 * std_val, 3 * std_val)
        
    return pos_signal