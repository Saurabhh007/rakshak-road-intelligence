"""YOLO inference interface with a deterministic, clearly-labelled demo fallback."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .config import DetectorConfig
from .models import Detection

logger = logging.getLogger(__name__)


class FrameDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]: ...


class MockPotholeDetector:
    """Deterministic demo implementation; it performs no ML inference."""

    is_mock = True

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        _validate_frame(frame)
        # A stable, opt-in-by-frame-index visual demo for development and tests.
        if not 110 <= frame_index <= 150:
            return []
        height, width = frame.shape[:2]
        progress = (frame_index - 110) / 40.0
        box_width, box_height = 30 + progress * 150, 15 + progress * 70
        center_x, center_y = width * 0.48, height * (0.55 + progress * 0.25)
        return [Detection(
            class_name="pothole",
            confidence=round(0.91 + progress * 0.04, 4),
            bbox=(center_x - box_width / 2, center_y - box_height / 2,
                  center_x + box_width / 2, center_y + box_height / 2),
        )]


class PotholeDetector:
    """Run configured Ultralytics YOLO weights on OpenCV images and video frames.

    A real detector is only activated when ``RAKSHAK_MODEL_PATH`` points to an
    existing weights file. This project does not ship or claim to ship a trained
    pothole model; without supplied pothole-capable weights, ``is_mock`` is true.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig.from_environment()
        self.model = None
        self.is_mock = True
        self._fallback = MockPotholeDetector()
        self._load_model()

    def _load_model(self) -> None:
        model_path = self.config.model_path
        if model_path is None:
            logger.warning("No RAKSHAK_MODEL_PATH configured; using mock detector")
            return
        if not model_path.is_file():
            logger.warning("Configured model weights were not found at %s; using mock detector", model_path)
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.is_mock = False
            logger.info("Loaded configured YOLO weights from %s", model_path)
        except Exception as error:
            raise RuntimeError(f"Could not load configured YOLO weights from {model_path}") from error

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        """Run inference for one BGR OpenCV frame."""
        _validate_frame(frame)
        if self.is_mock:
            return self._fallback.detect(frame, frame_index)
        assert self.model is not None
        result = self.model(frame, verbose=False, conf=self.config.confidence_threshold)[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(names[class_id])
            if class_name.casefold() != self.config.pothole_class_name.casefold():
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(class_name, confidence, (x1, y1, x2, y2)))
        return detections


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR NumPy array with shape (height, width, 3)")
