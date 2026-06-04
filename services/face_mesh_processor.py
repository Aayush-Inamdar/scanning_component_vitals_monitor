"""MediaPipe Face Landmarker processor for precise skin ROI extraction.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.x) with the FaceLandmarker
model. On first run the model file is downloaded automatically to the
project's data/ directory (~5 MB, one-time).

ROI regions (forehead + cheeks) give the strongest BVP signal.
Eyes, lips, hair, and background are masked out to reduce noise.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Stored alongside the project's data folder
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "face_landmarker.task"
)


def _ensure_model(model_path: Path = _DEFAULT_MODEL_PATH) -> Path:
    """Download the FaceLandmarker model if it isn't already on disk."""
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FaceMesh] Downloading face landmarker model -> {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[FaceMesh] Model downloaded ({model_path.stat().st_size // 1024} KB)")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download MediaPipe face landmarker model.\n"
            f"Please download it manually from:\n  {MODEL_URL}\n"
            f"and place it at:\n  {model_path}\n"
            f"Error: {e}"
        ) from e
    return model_path


# ---------------------------------------------------------------------------
# Landmark index groups
# Indices into MediaPipe's 478-point face mesh (468 base + 10 iris).
# Chosen to cover forehead and cheeks while avoiding eyes/lips/nose.
# ---------------------------------------------------------------------------

FOREHEAD_LANDMARKS: list[int] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]

LEFT_CHEEK_LANDMARKS: list[int] = [
    117, 118, 119, 120, 121, 128, 245, 188, 174,
    236, 134, 131, 123, 116, 143, 156, 139, 166,
]

RIGHT_CHEEK_LANDMARKS: list[int] = [
    346, 347, 348, 349, 350, 357, 465, 412, 399,
    456, 363, 360, 352, 345, 372, 383, 368, 393,
]

ALL_ROI_LANDMARKS: list[int] = (
    FOREHEAD_LANDMARKS + LEFT_CHEEK_LANDMARKS + RIGHT_CHEEK_LANDMARKS
)


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------

class FaceMeshProcessor:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API) to extract skin ROI per frame.

    Usage:
        with FaceMeshProcessor() as proc:
            masked_frame, roi_mask, face_found = proc.process(frame_bgr)
    """

    def __init__(
        self,
        model_path: Path | None = None,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        resolved_path = _ensure_model(
            model_path if model_path is not None else _DEFAULT_MODEL_PATH
        )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0  # monotonic timestamp for VIDEO mode

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """
        Detect face landmarks and return a skin-ROI-masked frame.

        Args:
            frame_bgr: Raw BGR frame from OpenCV.

        Returns:
            masked_frame : BGR frame — ROI pixels kept, rest blacked out.
            roi_mask     : uint8 mask (255 = ROI, 0 = excluded), or None.
            face_found   : True if a face was detected.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]

        # Tasks API needs RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode requires a strictly increasing timestamp in ms.
        if timestamp_ms is None:
            timestamp_ms = self._frame_ts_ms + 1
        if timestamp_ms <= self._frame_ts_ms:
            timestamp_ms = self._frame_ts_ms + 1
        self._frame_ts_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        if not result.face_landmarks:
            return frame_bgr, None, False

        landmarks = result.face_landmarks[0]  # first (only) face

        roi_points = self._landmarks_to_pixels(landmarks, ALL_ROI_LANDMARKS, w, h)
        roi_mask = self._build_roi_mask(roi_points, h, w)
        masked_frame = cv2.bitwise_and(frame_bgr, frame_bgr, mask=roi_mask)

        return masked_frame, roi_mask, True

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        roi_mask: np.ndarray | None,
        color: tuple[int, int, int] = (0, 255, 128),
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw the ROI contour on a frame for visual debugging."""
        display = frame_bgr.copy()
        if roi_mask is None:
            return display
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, color, thickness)
        return display

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_pixels(
        landmarks: list,
        indices: list[int],
        width: int,
        height: int,
    ) -> np.ndarray:
        points = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                x = max(0, min(width - 1, int(lm.x * width)))
                y = max(0, min(height - 1, int(lm.y * height)))
                points.append([x, y])
        return np.array(points, dtype=np.int32)

    @staticmethod
    def _build_roi_mask(
        roi_points: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(roi_points) < 3:
            return mask
        hull = cv2.convexHull(roi_points)
        cv2.fillConvexPoly(mask, hull, 255)
        return mask

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshProcessor":
        return self

    def __exit__(self, *_) -> None:
        self.close()
