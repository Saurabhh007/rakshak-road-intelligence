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


def test_real_detector_filters_d40_correctly_based_on_config():
    import torch
    from unittest.mock import MagicMock
    
    detector = PotholeDetector(DetectorConfig(pothole_class_name="D40"))
    detector.is_mock = False
    
    mock_box_d40 = MagicMock()
    mock_box_d40.conf = [torch.tensor(0.85)]
    mock_box_d40.cls = [torch.tensor(3.0)]  # class ID for D40 is 3
    mock_box_d40.xyxy = [torch.tensor([10.0, 20.0, 30.0, 40.0])]
    
    mock_box_d20 = MagicMock()
    mock_box_d20.conf = [torch.tensor(0.90)]
    mock_box_d20.cls = [torch.tensor(2.0)]  # class ID for D20 is 2
    mock_box_d20.xyxy = [torch.tensor([50.0, 60.0, 70.0, 80.0])]
    
    mock_result = MagicMock()
    mock_result.names = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40', 4: 'Repair'}
    mock_result.boxes = [mock_box_d40, mock_box_d20]
    
    detector.model = MagicMock(return_value=[mock_result])
    
    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    
    assert len(detections) == 1
    assert detections[0].class_name == "D40"
    assert detections[0].confidence == pytest.approx(0.85)
    assert detections[0].bbox == (10.0, 20.0, 30.0, 40.0)
