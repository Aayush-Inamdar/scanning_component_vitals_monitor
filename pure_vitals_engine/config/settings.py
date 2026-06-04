# Video Settings
CAMERA_INDEX = 0
RESOLUTION_W = 1280
RESOLUTION_H = 720
TARGET_FPS = 30.0

# Engine Settings (Multi-Tiered Buffers)
BUFFER_DURATION_SEC = 30.0      # Master memory limit
HR_WINDOW_SEC = 10.0            # Fast lane for Heart Rate
EXTENDED_WINDOW_SEC = 30.0      # Slow lane for HRV & Breathing

# Motion Defenses
MOTION_THRESHOLD_PIXELS = 15.0  
COOLDOWN_SEC = 1.0              

# Bandpass Filter Ranges
HR_MIN_HZ = 0.8  # 48 BPM
HR_MAX_HZ = 2.5  # 150 BPM
BR_MIN_HZ = 0.15 # 9 Breaths/min
BR_MAX_HZ = 0.4  # 30 Breaths/min


# Data Logging
CSV_LOG_FILE = "data/live_vitals_log.csv"