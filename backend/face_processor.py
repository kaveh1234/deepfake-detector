"""Face processing using MediaPipe Face Landmarker (Tasks API)."""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import Optional
import os
import urllib.request


@dataclass
class FaceROI:
    """Region of Interest data for face analysis."""
    bbox: tuple[int, int, int, int]  # x, y, w, h
    face_roi: np.ndarray  # Cropped face image
    nose_roi: np.ndarray  # Cropped nose region
    cheek_roi: np.ndarray  # Cropped cheek region for color sampling
    nose_width: float  # Distance between nostril landmarks
    face_width: float  # Distance between face edges
    landmarks: list[tuple[float, float, float]]  # All 478 landmarks


class FaceProcessor:
    """Process faces using MediaPipe Face Landmarker."""

    # MediaPipe Face Mesh landmark indices (same as before)
    NOSE_TIP = 1
    NOSE_LEFT = 279  # Left nostril
    NOSE_RIGHT = 49  # Right nostril
    FACE_LEFT = 234  # Left side of face
    FACE_RIGHT = 454  # Right side of face
    LEFT_CHEEK = 50
    RIGHT_CHEEK = 280
    FOREHEAD = 10

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

    def __init__(self):
        """Initialize MediaPipe Face Landmarker."""
        self._ensure_model_downloaded()

        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_downloaded(self):
        """Download the model file if it doesn't exist."""
        if not os.path.exists(self.MODEL_PATH):
            print(f"Downloading face landmarker model to {self.MODEL_PATH}...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print("Model downloaded successfully.")

    def process_frame(self, frame: np.ndarray) -> Optional[FaceROI]:
        """
        Process a frame and extract face ROI data.

        Args:
            frame: BGR image from OpenCV

        Returns:
            FaceROI data or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        face_landmarks = results.face_landmarks[0]
        h, w, _ = frame.shape

        # Extract all landmarks as list
        landmarks = [
            (lm.x, lm.y, lm.z)
            for lm in face_landmarks
        ]

        # Calculate bounding box
        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Extract face ROI
        face_roi = frame[y_min:y_max, x_min:x_max].copy()

        # Calculate nose width (distance between nostrils)
        nose_left = face_landmarks[self.NOSE_LEFT]
        nose_right = face_landmarks[self.NOSE_RIGHT]
        nose_width = np.sqrt(
            ((nose_left.x - nose_right.x) * w) ** 2 +
            ((nose_left.y - nose_right.y) * h) ** 2
        )

        # Calculate face width
        face_left = face_landmarks[self.FACE_LEFT]
        face_right = face_landmarks[self.FACE_RIGHT]
        face_width = np.sqrt(
            ((face_left.x - face_right.x) * w) ** 2 +
            ((face_left.y - face_right.y) * h) ** 2
        )

        # Extract nose ROI for squint detection
        nose_roi = self._extract_nose_roi(frame, face_landmarks, w, h)

        # Extract cheek ROI for color sampling (Disco trap)
        cheek_roi = self._extract_cheek_roi(frame, face_landmarks, w, h)

        return FaceROI(
            bbox=bbox,
            face_roi=face_roi,
            nose_roi=nose_roi,
            cheek_roi=cheek_roi,
            nose_width=nose_width,
            face_width=face_width,
            landmarks=landmarks
        )

    def _extract_nose_roi(
        self,
        frame: np.ndarray,
        landmarks,
        w: int,
        h: int
    ) -> np.ndarray:
        """Extract the nose region for texture analysis."""
        nose_tip = landmarks[self.NOSE_TIP]
        nose_left = landmarks[self.NOSE_LEFT]
        nose_right = landmarks[self.NOSE_RIGHT]

        center_x = int(nose_tip.x * w)
        center_y = int(nose_tip.y * h)

        nose_w = abs(int((nose_left.x - nose_right.x) * w))
        nose_h = int(nose_w * 1.2)

        x1 = max(0, center_x - nose_w)
        y1 = max(0, center_y - nose_h // 2)
        x2 = min(w, center_x + nose_w)
        y2 = min(h, center_y + nose_h)

        return frame[y1:y2, x1:x2].copy()

    def _extract_cheek_roi(
        self,
        frame: np.ndarray,
        landmarks,
        w: int,
        h: int
    ) -> np.ndarray:
        """Extract cheek region for color sampling."""
        left_cheek = landmarks[self.LEFT_CHEEK]
        right_cheek = landmarks[self.RIGHT_CHEEK]
        forehead = landmarks[self.FOREHEAD]
        nose_tip = landmarks[self.NOSE_TIP]

        center_x = int((left_cheek.x + right_cheek.x) / 2 * w)
        center_y = int((forehead.y + nose_tip.y) / 2 * h)

        size = 50
        x1 = max(0, center_x - size)
        y1 = max(0, center_y - size)
        x2 = min(w, center_x + size)
        y2 = min(h, center_y + size)

        return frame[y1:y2, x1:x2].copy()

    def get_face_color(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """
        Get the average color of the face region.

        Returns:
            Tuple of (R, G, B) normalized to 0-1, or None if no face
        """
        roi_data = self.process_frame(frame)
        if roi_data is None or roi_data.cheek_roi.size == 0:
            return None

        rgb_roi = cv2.cvtColor(roi_data.cheek_roi, cv2.COLOR_BGR2RGB)
        mean_color = np.mean(rgb_roi, axis=(0, 1)) / 255.0

        return tuple(mean_color)

    def draw_landmarks(
        self,
        frame: np.ndarray,
        roi_data: FaceROI,
        draw_all: bool = False
    ) -> np.ndarray:
        """Draw face landmarks and ROI on frame."""
        h, w, _ = frame.shape
        output = frame.copy()

        x, y, bw, bh = roi_data.bbox
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        if draw_all:
            for lm_x, lm_y, _ in roi_data.landmarks:
                px, py = int(lm_x * w), int(lm_y * h)
                cv2.circle(output, (px, py), 1, (0, 255, 255), -1)
        else:
            key_indices = [
                self.NOSE_TIP, self.NOSE_LEFT, self.NOSE_RIGHT,
                self.FACE_LEFT, self.FACE_RIGHT,
                self.LEFT_CHEEK, self.RIGHT_CHEEK
            ]
            for idx in key_indices:
                if idx < len(roi_data.landmarks):
                    lm_x, lm_y, _ = roi_data.landmarks[idx]
                    px, py = int(lm_x * w), int(lm_y * h)
                    cv2.circle(output, (px, py), 3, (255, 0, 0), -1)

        if len(roi_data.landmarks) > max(self.NOSE_LEFT, self.NOSE_RIGHT):
            nl = roi_data.landmarks[self.NOSE_LEFT]
            nr = roi_data.landmarks[self.NOSE_RIGHT]
            cv2.line(
                output,
                (int(nl[0] * w), int(nl[1] * h)),
                (int(nr[0] * w), int(nr[1] * h)),
                (255, 0, 255), 2
            )

        if len(roi_data.landmarks) > max(self.FACE_LEFT, self.FACE_RIGHT):
            fl = roi_data.landmarks[self.FACE_LEFT]
            fr = roi_data.landmarks[self.FACE_RIGHT]
            cv2.line(
                output,
                (int(fl[0] * w), int(fl[1] * h)),
                (int(fr[0] * w), int(fr[1] * h)),
                (0, 255, 255), 2
            )

        return output

    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()
