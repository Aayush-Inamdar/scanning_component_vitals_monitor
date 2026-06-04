"""MediaPipe Face Landmarker processor for precise skin ROI extraction.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.x) with the FaceLandmarker
model. On first run the model file is downloaded automatically to the
project's data/ directory (~5 MB, one-time).

ROI regions (forehead + cheeks) give the strongest BVP signal.
Eyes, lips, hair, and background are masked out to reduce noise.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceMeshProcessor:
    def __init__(self, model_asset_path='data/face_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # NEW: Jitter Smoothing State
        self.prev_radius = None
        self.prev_anchors = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detector.close()

    def process(self, frame, timestamp_ms):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        face_found = False

        if detection_result.face_landmarks:
            face_found = True
            landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape
            
            face_width = abs(landmarks[454].x - landmarks[234].x) * w
            target_radius = face_width * 0.10 

            target_anchors = [
                (landmarks[151].x * w, landmarks[151].y * h), # Forehead
                (landmarks[205].x * w, landmarks[205].y * h), # Left Cheek
                (landmarks[425].x * w, landmarks[425].y * h)  # Right Cheek
            ]

            # --- THE FIX: Exponential Moving Average (EMA) Smoothing ---
            if self.prev_radius is None:
                self.prev_radius = target_radius
                self.prev_anchors = target_anchors
            else:
                # 80% old value, 20% new value. Completely dampens micro-jitter.
                self.prev_radius = 0.8 * self.prev_radius + 0.2 * target_radius
                for i in range(3):
                    cx = 0.8 * self.prev_anchors[i][0] + 0.2 * target_anchors[i][0]
                    cy = 0.8 * self.prev_anchors[i][1] + 0.2 * target_anchors[i][1]
                    self.prev_anchors[i] = (cx, cy)

            # Draw the rock-solid isolated regions
            for pt in self.prev_anchors:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), int(self.prev_radius), 255, -1)
                
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame, mask, face_found

    def draw_landmarks(self, frame, roi_mask):
        """Draws a clean UI overlay showing exactly what the engine is reading."""
        # SAFETY GATE: If no face is detected yet, just return the normal video frame
        if not np.any(roi_mask):
            return frame.copy()
            
        # Create a solid green image the same size as the webcam feed
        green_overlay = np.zeros_like(frame)
        green_overlay[:] = (0, 255, 0)
        
        # Blend the entire frame with the green overlay
        blended = cv2.addWeighted(frame, 0.7, green_overlay, 0.3, 0)
        
        # Use NumPy to elegantly stamp the blended green pixels ONLY where the mask is active
        display_frame = np.where(roi_mask[:, :, np.newaxis] > 0, blended, frame)
        
        return display_frame