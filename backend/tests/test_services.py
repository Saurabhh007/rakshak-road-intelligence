import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app import models, schemas
from app.services.geofence import haversine_distance, calculate_bearing
from app.services.severity import calculate_severity
from app.services.road_health import calculate_road_health
from app.services.verification import process_observation

# Configure an in-memory SQLite DB for testing services
engine = create_engine("sqlite:///:memory:")
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

def test_haversine_distance():
    # MG Road to Kanteerava Stadium Bangalore (approx 1.1 km)
    dist = haversine_distance(12.9716, 77.5946, 12.9700, 77.5850)
    assert 1000 <= dist <= 1200

def test_bearing():
    # Driving directly north
    bearing = calculate_bearing(12.9000, 77.5000, 12.9100, 77.5000)
    assert abs(bearing - 0.0) < 1.0 or abs(bearing - 360.0) < 1.0

def test_calculate_severity():
    # High area ratio -> high severity
    assert calculate_severity(0.06, 0.85) == "high"
    # Medium area ratio -> medium severity
    assert calculate_severity(0.02, 0.70) == "medium"
    # Low area ratio -> low severity
    assert calculate_severity(0.005, 0.50) == "low"

def test_calculate_road_health():
    # Test prototype scoring reductions
    h1 = models.Hazard(type="pothole", severity="high", status="ACTIVE")
    h2 = models.Hazard(type="pothole", severity="medium", status="VERIFIED")
    h3 = models.Hazard(type="pothole", severity="low", status="DETECTED")
    h4 = models.Hazard(type="pothole", severity="high", status="RESOLVED") # should be ignored
    
    score_data = calculate_road_health([h1, h2, h3, h4])
    # Deductions: high (10) + medium (5) + low (2) = 17 pts
    assert score_data["prototype_road_health_score"] == 83.0
    assert score_data["active_hazards"] == 3
    assert score_data["breakdown"]["high_severity"] == 1
    assert "PROTOTYPE ONLY" in score_data["notice"]

def test_proximity_aggregation_new_hazard(db):
    obs = schemas.ObservationCreate(
        source="ai_camera",
        latitude=12.9720,
        longitude=77.5940,
        confidence=0.85,
        timestamp="2026-08-22T01:30:00Z"
    )
    
    hazard = process_observation(db, obs)
    assert hazard.id is not None
    assert hazard.status == "DETECTED"
    assert hazard.latitude == 12.9720
    assert hazard.longitude == 77.5940
    assert len(hazard.observations) == 1

def test_proximity_aggregation_existing_hazard(db):
    # Create first observation (creates hazard)
    obs1 = schemas.ObservationCreate(
        source="ai_camera",
        latitude=12.97200,
        longitude=77.59400,
        confidence=0.85,
        timestamp="2026-08-22T01:30:00Z"
    )
    process_observation(db, obs1)
    
    # Create second observation within 5 meters (aggregates coordinates)
    obs2 = schemas.ObservationCreate(
        source="ai_camera",
        latitude=12.97202,
        longitude=77.59402,
        confidence=0.90,
        timestamp="2026-08-22T01:30:05Z"
    )
    hazard = process_observation(db, obs2)
    
    # Proximity cluster matches existing hazard
    assert db.query(models.Hazard).count() == 1
    assert db.query(models.Observation).count() == 2
    assert hazard.status == "DETECTED"  # Still DETECTED (requires 3 observations to trigger VERIFIED)
    assert hazard.confidence == 0.90  # Max aggregated confidence

def test_prototype_verification_lifecycle(db):
    # Create 3 observations at the same location to verify DETECTED -> VERIFIED transition
    for i in range(3):
        obs = schemas.ObservationCreate(
            source="ai_camera",
            latitude=12.9720,
            longitude=77.5940,
            confidence=0.80,
            timestamp=f"2026-08-22T01:30:0{i}Z"
        )
        hazard = process_observation(db, obs)
        
    assert hazard.status == "VERIFIED"
    assert len(hazard.observations) == 3
