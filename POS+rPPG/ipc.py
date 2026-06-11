import multiprocessing as mp

class PipelineIPC:
    """
    Inter-Process Communication (IPC) primitives for the Tri-Process Vitals Pipeline.
    This serves as the shared memory backbone connecting the Producer, POS Tracker, and ML Calibrator.
    """
    def __init__(self):
        # ==========================================
        # 1. DATA STREAMS (Queues)
        # ==========================================
        
        # Carries lightweight (timestamp, r, g, b) tuples from the Producer to the POS Tracker.
        # Uncapped, as it needs to flow continuously at high speed.
        self.rgb_queue = mp.Queue()
        
        # Carries downscaled spatial frames from the Producer to the ML Calibrator.
        # We set a maxsize to act as a bounded ring buffer (e.g., 30fps * 15s = 450 frames maximum)
        # to prevent RAM blowouts if the ML engine is slow to start.
        self.frame_buffer_queue = mp.Queue(maxsize=1000) 
        
        # Carries the final calculated metrics dict ({'hr': x, 'hrv': y, 'br': z}) 
        # from the POS Tracker back to the Main UI/Server.
        self.results_queue = mp.Queue()

        # NEW: The Lossy UI Stream. 
        # Maxsize is tiny so if the UI lags, it drops frames rather than choking RAM.
        self.ui_frame_queue = mp.Queue(maxsize=2)

        # ==========================================
        # 2. SHARED MEMORY VARIABLES
        # ==========================================
        
        # The critical injection point. The ML Calibrator writes the verified HR here.
        # The POS Tracker reads this. 'd' stands for double-precision float. 
        # Initialized to 0.0 to indicate no lock has occurred yet.
        self.anchor_hr_value = mp.Value('d', 0.0)

        # NEW: We need to pass the ML SQI value to the UI so you can see it on screen.
        self.ml_sqi_value = mp.Value('d', 0.0)

        # ==========================================
        # 3. STATE & CONTROL EVENTS
        # ==========================================
        
        # Signalled by the Producer process once MediaPipe has successfully 
        # identified and locked onto the facial ROI. Gates the start of the 15s race.
        self.roi_locked_event = mp.Event()
        
        # Signalled by the ML Calibrator process when it finishes its run, 
        # either successfully (SQI > 0.4) or unsuccessfully (failed lock).
        self.ml_done_event = mp.Event()
        
        # Signalled by the Orchestrator to forcefully terminate the ML Calibrator process
        # if the 15-second hard timeout is reached.
        self.ml_kill_event = mp.Event()

        # NEW: Added to ensure the 15-second race doesn't start until PyTorch/CUDA 
        # is fully loaded and ready to accept frames.
        self.ml_ready_event = mp.Event()

        self.shutdown_event = mp.Event()