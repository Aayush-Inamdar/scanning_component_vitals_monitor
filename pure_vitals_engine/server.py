import asyncio
import json
import base64
import cv2
import numpy as np
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from services.signal_processor import PosSignalProcessor
from services.face_mesh_processor import FaceMeshProcessor

app = FastAPI()

@app.get("/")
async def get_index():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws/vitals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    processor = PosSignalProcessor() 
    face_processor = FaceMeshProcessor()
    
    scan_start_time = None
    phase = "IDLE"
    frame_count = 0
    t_buf, r_buf, g_buf, b_buf = [], [], [], []
    
    # --- Motion Tracking State ---
    prev_face_center = None
    motion_detected = False
    
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            if data['type'] == 'command':
                if data['action'] == 'START_SCAN':
                    scan_start_time = time.time()
                    processor.reset()
                    t_buf.clear(); r_buf.clear(); g_buf.clear(); b_buf.clear()
                    frame_count = 0
                    prev_face_center = None
                    phase = "CALIBRATING (Wait 5s)"
                    await websocket.send_json({"phase": phase})
                    continue
                elif data['action'] == 'STOP_SCAN':
                    scan_start_time = None
                    phase = "IDLE"
                    await websocket.send_json({"phase": phase, "progress": 0})
                    continue
                
            # --- HANDLE INCOMING FRAMES ---
            if data['type'] == 'frame' and scan_start_time is not None:
                elapsed = time.time() - scan_start_time
                frame_count += 1
                
                img_data = base64.b64decode(data['image'].split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                _, mask, face_found = face_processor.process(frame, frame_count * 33)
                
                # --- FORGIVING MOTION DETECTION ---
                if not face_found:
                    motion_detected = True
                elif np.any(mask):
                    M = cv2.moments(mask)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        if prev_face_center is None:
                            prev_face_center = (cx, cy)
                        else:
                            dist = np.sqrt((cx - prev_face_center[0])**2 + (cy - prev_face_center[1])**2)
                            
                            # 80 pixels allows for natural breathing and slight swaying
                            if dist > 80.0:
                                motion_detected = True
                        
                        mean_bgr = cv2.mean(frame, mask=mask)
                        t_buf.append(elapsed)
                        b_buf.append(mean_bgr[0])
                        g_buf.append(mean_bgr[1])
                        r_buf.append(mean_bgr[2])
                        
                        if len(t_buf) > 1200:
                            t_buf.pop(0); r_buf.pop(0); g_buf.pop(0); b_buf.pop(0)

                # --- VISIBLE MOTION RESET ---
                if motion_detected:
                    processor.reset()
                    t_buf.clear(); r_buf.clear(); g_buf.clear(); b_buf.clear()
                    motion_detected = False
                    prev_face_center = None 
                    
                    phase = "MOTION DETECTED - RESTARTING"
                    await websocket.send_json({"phase": phase, "progress": 0})
                    
                    # PAUSE: Force the UI to show the warning for 1.5 seconds before restarting
                    await asyncio.sleep(1.5)
                    scan_start_time = time.time() 
                    continue

                # --- THE CONVERGENCE STATE MACHINE ---
                if elapsed < 5.0:
                    phase = "STABILIZING SENSORS..."
                    await websocket.send_json({"phase": phase, "progress": int((elapsed/5.0)*100)})
                    
                else:
                    results = None
                    if len(t_buf) > 30: 
                        results = processor.process(
                            np.array(t_buf), np.array(r_buf), 
                            np.array(g_buf), np.array(b_buf)
                        )
                    
                    # Normal 30-second scan window
                    if elapsed < 35.0:
                        phase = "SCANNING..."
                        progress = int(((elapsed - 5.0)/30.0)*100)
                        
                    # Overtime: 30s to 45s (Waiting for HRV)
                    elif elapsed >= 35.0 and elapsed < 50.0:
                        if results and results.get('hrv') and results.get('br') and results.get('hr'):
                            phase = "CONVERGED"
                            progress = 100
                        else:
                            phase = "EXTENDING SCAN (Acquiring HRV)..."
                            progress = 99
                            
                    # Hard Stop: 45 seconds maximum. Do not hold the user hostage.
                    else:
                        phase = "CONVERGED"
                        progress = 100

                    response = {"phase": phase, "progress": progress}
                    
                    if results and results.get('hr'):
                        response['hr'] = results['hr']
                        if results.get('hrv'): response['hrv'] = results['hrv']
                        if results.get('br'): response['br'] = results['br']
                        
                    await websocket.send_json(response)
                    
                    if phase == "CONVERGED":
                        scan_start_time = None 
                    
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)