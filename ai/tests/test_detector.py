import numpy as np
import pytest
from src.detector import HazardDetector

def test_mock_detector_initialization():
    """
    Test that detector defaults to mock/simulated mode when no weights exist.
    """
    detector = HazardDetector(model_path="missing_weights.pt")
    assert detector.is_mock is True

def test_mock_detector_frame_inference():
    """
    Test that mock detections are returned for specific frame ranges.
    """
    detector = HazardDetector(model_path="missing_weights.pt")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Frame 0 should return no potholes
    detections = detector.detect(frame, frame_index=0)
    assert len(detections) == 0
    
    # Frame 120 should return a pothole
    detections = detector.detect(frame, frame_index=120)
    assert len(detections) == 1
    assert detections[0]["class_name"] == "pothole"
    assert detections[0]["confidence"] > 0.80
    assert len(detections[0]["bbox"]) == 4
