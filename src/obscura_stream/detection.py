"""Detection utilities for YOLO and MediaPipe.

This module provides common detection functions used across different detection modes:
    - YOLO person detection
    - MediaPipe face detection
    - Combined detection utilities
"""

from typing import Tuple, List, Optional
import numpy as np
import cv2
import mediapipe as mp
from ultralytics import YOLO
import torch

from obscura_stream.config import YOLO_MODEL_PATH, DETECTION_MODE


class DetectionResult:
    """Container for detection results."""
    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        label: str,
        size: Optional[Tuple[int, int]] = None
    ):
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.confidence = confidence
        self.label = label
        self.size = size or (bbox[2] - bbox[0], bbox[3] - bbox[1])

    @property
    def display_text(self) -> str:
        """Returns formatted display text for the detection."""
        w, h = self.size
        return f"{self.label} conf={self.confidence:.2f} size={w}x{h}"


class YOLODetector:
    """YOLO detector wrapper for person detection."""
    def __init__(self, model_path: str = YOLO_MODEL_PATH):
        self.model = YOLO(model_path)
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def detect_persons(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """Detect persons in the frame using YOLO.

        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold for detections

        Returns:
            List of DetectionResult objects for person detections
        """
        results = self.model.predict(frame, device=self.device, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if (self.model.names[cls_id] == "person" and 
                conf >= conf_threshold):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection = DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    label="person"
                )
                detections.append(detection)
        
        return detections

    def get_best_person(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.5
    ) -> Optional[DetectionResult]:
        """Get the highest confidence person detection.

        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold for detections

        Returns:
            DetectionResult for highest confidence person, or None if no detection
        """
        detections = self.detect_persons(frame, conf_threshold)
        if not detections:
            return None
            
        return max(detections, key=lambda x: x.confidence) 


class FaceDetector:
    """MediaPipe face detector wrapper."""
    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.5
    ):
        """Initialize MediaPipe face detector.
        
        Args:
            model_selection: 0 for short-range, 1 for full-range detection
            min_detection_confidence: Minimum confidence threshold
        """
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(
        self,
        frame: np.ndarray
    ) -> List[DetectionResult]:
        """Detect faces in the frame using MediaPipe.

        Args:
            frame: Input image frame in BGR format

        Returns:
            List of DetectionResult objects for face detections
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        if not results.detections:
            return []

        ih, iw, _ = frame.shape
        detections = []

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            x2 = x1 + w
            y2 = y1 + h

            detection_result = DetectionResult(
                bbox=(x1, y1, x2, y2),
                confidence=float(detection.score[0]),
                label="face"
            )
            detections.append(detection_result)

        return detections

    def get_best_face(
        self,
        frame: np.ndarray
    ) -> Optional[DetectionResult]:
        """Get the highest confidence face detection.

        Args:
            frame: Input image frame in BGR format

        Returns:
            DetectionResult for highest confidence face, or None if no detection
        """
        detections = self.detect_faces(frame)
        if not detections:
            return None
            
        return max(detections, key=lambda x: x.confidence)


class CombinedDetector:
    """Combined or single-mode detector based on configuration."""
    def __init__(self, mode: str = DETECTION_MODE):
        """Initialize appropriate detector(s) based on mode.
        
        Args:
            mode: Detection mode ("hybrid" or "person_only")
        """
        self.mode = mode
        self.person_detector = YOLODetector()
        self.face_detector = FaceDetector() if mode == "hybrid" else None

    def detect(
        self,
        frame: np.ndarray
    ) -> Optional[DetectionResult]:
        """Detect objects based on configured mode.

        Args:
            frame: Input image frame

        Returns:
            Best detection result based on mode:
                - hybrid: face preferred over person
                - person_only: only person detection
        """
        if self.mode == "hybrid":
            # Try face detection first
            face_result = self.face_detector.get_best_face(frame)
            if face_result is not None:
                return face_result

        # Fall back to person detection
        return self.person_detector.get_best_person(frame) 