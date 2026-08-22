import numpy as np
import pytest

from ai.config import DetectorConfig
from ai.detector import PotholeDetector
from ai.models import Detection
from ai.video_processor import VideoProcessor


def test_missing_weights_use_the_explicit_mock_detector(tmp_path):
    detector = PotholeDetector(DetectorConfig(model_path=tmp_path / "missing.pt"))
    assert detector.is_mock is True


def test_mock_detector_returns_structured_pothole_detection():
    detector = PotholeDetector(DetectorConfig())
    detections = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8), frame_index=120)

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].class_name == "pothole"
    assert 0 <= detections[0].confidence <= 1
    assert len(detections[0].bbox) == 4


def test_detector_rejects_invalid_image_shape():
    with pytest.raises(ValueError, match="BGR"):
        PotholeDetector(DetectorConfig()).detect(np.zeros((10, 10), dtype=np.uint8))


def test_video_processor_delegates_to_detector():
    detector = PotholeDetector(DetectorConfig())
    detections = VideoProcessor(detector).process_frame(np.zeros((100, 100, 3), dtype=np.uint8), 120)
    assert detections[0].class_name == "pothole"
