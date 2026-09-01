"""Test suite for RAKSHAK perception detection pipeline, upload equivalence, and temporal verification."""

from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai.config import DetectorConfig
from ai.detector import PotholeDetector, _patch_yolo12_attention
from ai.test_webcam import CameraSource, format_detection_dict
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import Hazard, Observation
from app.schemas import ObservationCreate
from app.services.verification import process_observation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROAD11_PATH = PROJECT_ROOT / "ai" / "test_images" / "sample_road11.jpg"
MODEL_PATH = PROJECT_ROOT / "ai" / "model" / "yolo12s_RDD2022_best.pt"


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_real_model_detection_sample_road11():
    """Verify sample_road11.jpg produces genuine D40 pothole detection at ~0.33 confidence."""
    assert MODEL_PATH.exists(), f"Model weights not found at {MODEL_PATH}"
    assert SAMPLE_ROAD11_PATH.exists(), f"Test image not found at {SAMPLE_ROAD11_PATH}"

    _patch_yolo12_attention()
    detector = PotholeDetector(
        DetectorConfig(
            model_path=MODEL_PATH,
            confidence_threshold=0.25,
            pothole_class_name="D40",
        )
    )
    assert not detector.is_mock, "Detector should be real (not mock)"

    img = cv2.imread(str(SAMPLE_ROAD11_PATH))
    assert img is not None, "Failed to read sample_road11.jpg"

    # Inference at candidate threshold 0.25
    detections = detector.detect(img, confidence_threshold=0.25)
    assert len(detections) >= 1, "Expected at least 1 D40 detection at 0.25 threshold"

    d40 = detections[0]
    assert d40.class_name == "D40"
    assert 0.25 <= d40.confidence <= 0.45, f"Expected confidence around 0.33, got {d40.confidence}"
    assert len(d40.bbox) == 4

    # Baseline production threshold 0.60 should yield 0 detections
    detections_high = detector.detect(img, confidence_threshold=0.60)
    assert len(detections_high) == 0, "Expected 0 detections at 0.60 threshold"


def test_upload_api_equivalence(client):
    """Verify Upload API endpoint matches direct python inference for sample_road11.jpg."""
    with open(SAMPLE_ROAD11_PATH, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/upload/inference?threshold=0.25",
        files={"file": ("sample_road11.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ai_mode"] in {"REAL", "real_inference"}
    assert data["is_simulated"] is False
    assert len(data["detections"]) >= 1

    det = data["detections"][0]
    assert det["class_name"] == "D40"
    assert 0.25 <= det["confidence"] <= 0.45
    assert data["annotated_image"] is not None
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")


def test_temporal_verification_cycle():
    """Verify that candidate observations transition DETECTED -> VERIFIED after required real observations."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Clean test hazards/observations
        db.query(Observation).delete()
        db.query(Hazard).delete()
        db.commit()

        lat, lon = 18.61140, 73.75010
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # 1. First real candidate observation (e.g. from camera frame 1)
        obs1 = ObservationCreate(
            source="ai_camera",
            latitude=lat,
            longitude=lon,
            confidence=0.33,
            timestamp=now_iso,
            sensor_evidence=None,
            is_simulated=False,
            gps_source="live",
            gps_is_simulated=False,
        )
        hazard1 = process_observation(db, obs1)
        assert hazard1.status == "DETECTED", "1st observation should have status DETECTED"
        assert hazard1.source == "ai_camera"
        assert bool(hazard1.is_simulated) is False
        assert len(hazard1.observations) == 1

        # 2. Second real candidate observation nearby (within 5 meters)
        obs2 = ObservationCreate(
            source="ai_camera",
            latitude=lat + 0.00002,
            longitude=lon + 0.00002,
            confidence=0.34,
            timestamp=now_iso,
            sensor_evidence=None,
            is_simulated=False,
            gps_source="live",
            gps_is_simulated=False,
        )
        hazard2 = process_observation(db, obs2)
        assert hazard2.id == hazard1.id, "Observations within cluster radius must group under same hazard"
        assert hazard2.status == "DETECTED", "2nd observation should still be DETECTED (pending 3rd confirmation)"
        assert len(hazard2.observations) == 2

        # 3. Third real candidate observation -> triggers transition to VERIFIED
        obs3 = ObservationCreate(
            source="ai_camera",
            latitude=lat - 0.00001,
            longitude=lon - 0.00001,
            confidence=0.35,
            timestamp=now_iso,
            sensor_evidence=None,
            is_simulated=False,
            gps_source="live",
            gps_is_simulated=False,
        )
        hazard3 = process_observation(db, obs3)
        assert hazard3.id == hazard1.id
        assert hazard3.status == "VERIFIED", "3rd observation must promote status to VERIFIED"
        assert hazard3.confidence == 0.35, "Hazard confidence should take maximum observation score"
        assert len(hazard3.observations) == 3

    finally:
        db.close()


def test_camera_source_abstraction():
    """Verify CameraSource modular interface correctly accepts video files and device indices."""
    cam = CameraSource("ai/test_videos/road_video.mp4")
    assert cam.open() is True
    ret, frame = cam.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.ndim == 3
    cam.release()
    assert cam.is_opened is False


def test_webcam_detection_with_test_fallback_gps():
    """Verify webcam detections attach test fallback GPS and persist to DB when live GPS is absent."""
    from app.services.video_processor import VideoProcessor
    from app.config import settings
    from ai.models import Detection

    vp = VideoProcessor()
    # Ensure live coordinates are None (laptop state)
    vp.latitude = None
    vp.longitude = None
    vp.gps_source = "N/A"
    vp.last_save_time = 0.0

    test_det = Detection(bbox=(50.0, 60.0, 150.0, 180.0), confidence=0.35, class_name="D40")
    vp._save_detections([test_det], source_tag="ai_camera")

    db = SessionLocal()
    try:
        latest = db.query(Hazard).order_by(Hazard.id.desc()).first()
        assert latest is not None
        assert latest.type == "pothole"
        assert latest.latitude == settings.DEFAULT_TEST_LATITUDE
        assert latest.longitude == settings.DEFAULT_TEST_LONGITUDE
        assert latest.gps_source == "test_fallback"
        assert bool(latest.gps_is_simulated) is True
        assert bool(latest.is_simulated) is False
        assert latest.source == "ai_camera"
    finally:
        db.close()
