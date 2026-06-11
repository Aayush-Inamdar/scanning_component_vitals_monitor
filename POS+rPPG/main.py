import cv2
import queue
from orchestrator import VitalsOrchestrator

def overlay_telemetry(frame, hr, hrv, br, anchor_hr, sqi, ml_done, ml_kill):
    """Draws a semi-transparent dashboard directly over the live video feed."""
    
    # Determine System State for UI
    if not ml_done:
        status = "CALIBRATING: Running ML Warm-Start..."
        color = (0, 165, 255) # Orange
    elif ml_kill:
        status = f"TRACKING: Math Fallback (Failed SQI: {sqi:.2f})"
        color = (0, 255, 255) # Yellow
    else:
        status = f"TRACKING: ML Anchor Locked ({anchor_hr:.1f} BPM | SQI: {sqi:.2f})"
        color = (0, 255, 0) # Green

    # Draw semi-transparent background box for text readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Render Metrics on Video
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Live HR: {hr if hr else '--'} BPM", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"HRV: {hrv if hrv else '--'} ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Resp: {br if br else '--'} rpm", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

def main():
    print("\n[SYSTEM] Booting POS+rPPG Visual Backend...")
    orchestrator = VitalsOrchestrator()
    
    try:
        results_queue, _, _ = orchestrator.start_session()
        
        latest_hr, latest_hrv, latest_br = None, None, None
        print("[MAIN] Visual Dashboard is live.")
        
        while True:
            # 1. Drain the Results Queue (Math Output)
            try:
                while True:
                    res = results_queue.get_nowait()
                    if res['hr'] is not None: latest_hr = res['hr']
                    if res['hrv'] is not None: latest_hrv = res['hrv']
                    if res['br'] is not None: latest_br = res['br']
            except queue.Empty:
                pass

            # 2. Get the Live Video Frame
            try:
                # Use a short timeout so the UI doesn't hang if the camera drops a frame
                frame = orchestrator.ipc.ui_frame_queue.get(timeout=0.1)
                
                # 3. Read IPC State Data
                anchor = orchestrator.ipc.anchor_hr_value.value
                sqi = orchestrator.ipc.ml_sqi_value.value
                ml_done = orchestrator.ipc.ml_done_event.is_set()
                ml_kill = orchestrator.ipc.ml_kill_event.is_set()

                # 4. Render and Display
                final_frame = overlay_telemetry(frame, latest_hr, latest_hrv, latest_br, anchor, sqi, ml_done, ml_kill)
                cv2.imshow("Vitals Telemetry", final_frame)

            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[MAIN] Exit command received.")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        orchestrator.shutdown()
        print("[SYSTEM] Safely offline.")

if __name__ == "__main__":
    main()