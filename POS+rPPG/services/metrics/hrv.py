import numpy as np
from scipy import signal

def calculate_hrv_rmssd(input_dict):
    """
    Python equivalent of the TS/JS calculateHRVRMSSDValue function.
    Calculates RMSSD from an array of RR intervals (in milliseconds).
    """
    heartbeats = input_dict.get('correctedHeartbeats', input_dict.get('heartbeats', []))
    
    if not isinstance(heartbeats, (list, np.ndarray)) or len(heartbeats) < 2:
        return {'calculatedHrvRmssdMs': 0.0}
        
    rr_intervals = np.array([float(x) for x in heartbeats if np.isfinite(float(x))])
    
    if len(rr_intervals) < 2:
        return {'calculatedHrvRmssdMs': 0.0}
        
    diffs = np.diff(rr_intervals)
    sum_of_squared_differences = np.sum(np.square(diffs))
    calculated_hrv_rmssd_ms = np.sqrt(sum_of_squared_differences / len(diffs))
    
    input_dict['calculatedHrvRmssdMs'] = calculated_hrv_rmssd_ms
    
    return {'calculatedHrvRmssdMs': calculated_hrv_rmssd_ms}

def get_rmssd_from_array(clean_rr_intervals_array):
    """Convenience wrapper to feed our engine's NumPy arrays directly into the JS structure."""
    mock_input_object = {'heartbeats': clean_rr_intervals_array}
    result = calculate_hrv_rmssd(mock_input_object)
    return result['calculatedHrvRmssdMs']

def calculate_hrv(clean_pulse, fps, current_hr, last_known_hrv=None):
    """
    Restored bridge function for live UI processing. 
    Extracts RR intervals from the live pulse wave and routes them through the new JS-translated RMSSD calculator.
    """
    if current_hr is None or current_hr < 40:
        return last_known_hrv
    
    clean_pulse = np.ravel(clean_pulse)
        
    # Extract peaks from the live rolling window
    min_dist_frames = int(fps * (60.0 / 150.0))
    peaks, _ = signal.find_peaks(clean_pulse, distance=min_dist_frames, prominence=0.25)
    
    if len(peaks) < 3:
        return last_known_hrv
        
    # Convert peaks to RR intervals
    raw_rr_intervals = np.diff(peaks) * (1000.0 / fps)
    
    # Filter out wild variations
    median_rr = np.median(raw_rr_intervals)
    valid_mask = (raw_rr_intervals >= median_rr * 0.75) & (raw_rr_intervals <= median_rr * 1.25)
    clean_rr_intervals = raw_rr_intervals[valid_mask]
    
    if len(clean_rr_intervals) < 2:
        return last_known_hrv
        
    # Return the result from the new clinical math function
    return get_rmssd_from_array(clean_rr_intervals)