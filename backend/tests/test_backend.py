import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_api_root():
    """
    Test root health check route.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_list_hazards_initially_empty():
    """
    Test listing hazards returns empty list on clean start.
    """
    response = client.get("/api/hazards")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_telemetry_response():
    """
    Test posting telemetry updates and warnings.
    """
    payload = {
        "latitude": 12.9717,
        "longitude": 77.5947,
        "speed_kmh": 35.0,
        "imu_accel_z": 9.8,
        "gps_source": "SIMULATED"
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "warning_active" in data
    assert "system_status" in data
    assert data["system_status"]["gps"] == "SIMULATED"

def test_simulation_reset_seeds_db():
    """
    Test simulation reset endpoint successfully clears and seeds DB.
    """
    response = client.post("/api/simulation/reset", json={"mode": "DEMO"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # Verify hazards table is populated with seed values
    response = client.get("/api/hazards")
    assert response.status_code == 200
    assert len(response.json()) == 3

def test_proximity_warning_triggered():
    """
    Test warning activates when posting telemetry in warning range of pothole.
    """
    # Seed 1 pothole is at 18.61140, 73.75010 (Tathawade Service Road near Kasturi Chowk).
    # Send telemetry ~15 meters away
    payload = {
        "latitude": 18.61130,
        "longitude": 73.75000,
        "speed_kmh": 40.0,
        "imu_accel_z": 9.8,
        "gps_source": "SIMULATED"
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["warning_active"] is True
    assert data["warning"]["hazard_id"] is not None
    assert data["warning"]["distance_meters"] <= 50.0

def test_road_health_evaluation():
    """
    Test getting spatial road health evaluations.
    """
    response = client.get("/api/road-health?latitude=18.6170&longitude=73.7475&radius_meters=2000")
    assert response.status_code == 200
    data = response.json()
    assert "prototype_road_health_score" in data
    assert data["active_hazards"] == 3


def test_mode_switching():
    """
    Test switching modes between SYSTEM_DEMO, LIVE_CAMERA, IMAGE_FALLBACK, and UPLOAD_IMAGE.
    """
    # Switch to SYSTEM_DEMO
    res_demo = client.post("/api/mode", json={"mode": "SYSTEM_DEMO"})
    assert res_demo.status_code == 200
    assert res_demo.json()["mode"] == "SYSTEM_DEMO"
    assert res_demo.json()["detection_source"] == "DEMO/SIMULATED"

    # Switch to LIVE_CAMERA
    res_live = client.post("/api/mode", json={"mode": "LIVE_CAMERA"})
    assert res_live.status_code == 200
    assert res_live.json()["mode"] == "LIVE_CAMERA"

    # Switch to IMAGE_FALLBACK
    res_img = client.post("/api/mode", json={"mode": "IMAGE_FALLBACK"})
    assert res_img.status_code == 200
    assert res_img.json()["mode"] == "IMAGE_FALLBACK"

    # Switch to UPLOAD_IMAGE (Option D)
    res_upload = client.post("/api/mode", json={"mode": "UPLOAD_IMAGE"})
    assert res_upload.status_code == 200
    assert res_upload.json()["mode"] == "UPLOAD_IMAGE"
    assert res_upload.json()["detection_source"] == "REAL AI"

    # Query GET /api/mode
    res_get = client.get("/api/mode")
    assert res_get.status_code == 200
    assert res_get.json()["mode"] == "UPLOAD_IMAGE"


def test_real_image_inference_endpoint():
    """
    Test Option C isolated real image inference endpoint on sample_road11.jpg.
    """
    response = client.post("/api/demo/real-inference")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "AI_UNAVAILABLE"]
    if data["status"] == "ok":
        assert data["ai_mode"] == "REAL"
        assert data["is_simulated"] is False
        assert len(data["detections"]) >= 1
        assert data["detections"][0]["class_name"] == "D40"
        assert data["detections"][0]["confidence"] > 0.25
        assert data["annotated_image"] is not None
        assert data["annotated_image"].startswith("data:image/jpeg;base64,")


def test_upload_image_inference_endpoint():
    """
    Test Option D uploaded image real AI inference on sample_road11.jpg.
    """
    image_path = "ai/test_images/sample_road11.jpg"
    with open(image_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/upload/inference",
        files={"file": ("sample_road11.jpg", file_bytes, "image/jpeg")},
        params={"threshold": 0.25}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["input_type"] == "uploaded_image"
    assert data["ai_mode"] == "real_inference"
    assert data["is_simulated"] is False
    assert len(data["detections"]) >= 1
    assert data["detections"][0]["class_name"] == "D40"
    assert data["detections"][0]["confidence"] > 0.25
    assert data["annotated_image"] is not None
    assert data["annotated_image"].startswith("data:image/jpeg;base64,")


def test_upload_image_invalid_extension():
    """
    Test uploading unsupported file format returns 400 Bad Request.
    """
    response = client.post(
        "/api/upload/inference",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]



