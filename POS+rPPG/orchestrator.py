import multiprocessing as mp
import threading
import time

from ipc import PipelineIPC
from processes.frame_producer import run_producer
from processes.pos_tracker import run_pos_tracker
from processes.ml_calibrator import run_ml_calibrator

class VitalsOrchestrator:
    def __init__(self):
        self.ipc = PipelineIPC()
        self.processes = {}

    def start_session(self):
        print("[ORCHESTRATOR] Booting Tri-Process Architecture...")
        self.processes['producer'] = mp.Process(target=run_producer, args=(self.ipc,), daemon=True)
        self.processes['tracker'] = mp.Process(target=run_pos_tracker, args=(self.ipc,), daemon=True)
        self.processes['ml'] = mp.Process(target=run_ml_calibrator, args=(self.ipc,), daemon=True)

        for name, p in self.processes.items():
            p.start()
            print(f"[ORCHESTRATOR] Started {name.upper()} process (PID: {p.pid})")

        # NEW: Spin up the referee in the background so the UI isn't blocked!
        threading.Thread(target=self._referee_race, daemon=True).start()
        return self.ipc.results_queue, self.processes['producer'], self.processes['tracker']

    def _referee_race(self):
        """Runs in the background to manage the 15-second ML race."""
        print("[ORCHESTRATOR] Waiting for Deep Learning initialization...")
        self.ipc.ml_ready_event.wait() 
        
        print("[ORCHESTRATOR] Waiting for MediaPipe Face Lock...")
        self.ipc.roi_locked_event.wait()
        
        HARD_TIMEOUT_SEC = 55.0 
        print(f"[ORCHESTRATOR] Conditions met. 15-second data accumulation begun.")
        
        success = self.ipc.ml_done_event.wait(timeout=HARD_TIMEOUT_SEC)

        if not success:
            print("[ORCHESTRATOR] TIMEOUT EXCEEDED: ML Engine hung or failed to lock.")
            self.ipc.ml_kill_event.set()
        
        self.processes['ml'].join(timeout=3.0)
        if self.processes['ml'].is_alive():
            self.processes['ml'].terminate()
            
        print("[ORCHESTRATOR] ML Calibration phase concluded.")

    def shutdown(self):
        print("[ORCHESTRATOR] Shutting down. Waiting for Tracker to save graphs...")
        # Tell the tracker to stop processing and start graphing
        self.ipc.shutdown_event.set() 
        
        if self.processes['tracker'] and self.processes['tracker'].is_alive():
            self.processes['tracker'].join(timeout=10.0) # Give it 10s to render graphs
            
        for p in self.processes.values():
            if p and p.is_alive():
                p.terminate()
                p.join()
        print("[ORCHESTRATOR] Backend shutdown complete.")