"""Video-frame integration for the perception detector."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .detector import FrameDetector
from .models import Detection


class VideoProcessor:
    """Read a video and yield each frame with its structured detections."""

    def __init__(self, detector: FrameDetector) -> None:
        self.detector = detector

    def process_frame(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]:
        return self.detector.detect(frame, frame_index)

    def process_video(self, video_path: str | Path) -> Iterator[tuple[int, np.ndarray, list[Detection]]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        try:
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame_index, frame, self.process_frame(frame, frame_index)
                frame_index += 1
        finally:
            capture.release()
