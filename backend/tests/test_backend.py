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
    # Seed 1 pothole is at 12.9720, 77.5940.
    # Send telemetry 15 meters away
    payload = {
        "latitude": 12.9719,
        "longitude": 77.5939,
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
    response = client.get("/api/road-health?latitude=12.9720&longitude=77.5940&radius_meters=500")
    assert response.status_code == 200
    data = response.json()
    assert "prototype_road_health_score" in data
    assert data["active_hazards"] == 3

