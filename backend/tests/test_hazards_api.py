from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


def hazard_payload(**overrides):
    payload = {
        "type": "pothole",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "confidence": 0.91,
        "severity": "high",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "DETECTED",
        "source": "ai_camera",
    }
    payload.update(overrides)
    return payload


def test_health(client):
    assert client.get("/api/health").json() == {"status": "healthy"}


def test_hazard_lifecycle_and_nearby_search(client):
    created = client.post("/api/hazards", json=hazard_payload())
    assert created.status_code == 201
    hazard = created.json()
    assert hazard["id"] > 0
    assert hazard["source"] == "ai_camera"

    assert len(client.get("/api/hazards", params={"status": "DETECTED"}).json()) == 1
    assert client.get(f"/api/hazards/{hazard['id']}").status_code == 200
    assert len(client.get("/api/hazards/nearby", params={"latitude": 12.9716, "longitude": 77.5946, "radius_meters": 20}).json()) == 1

    updated = client.patch(f"/api/hazards/{hazard['id']}/status", json={"status": "RESOLVED"})
    assert updated.status_code == 200
    assert updated.json()["status"] == "RESOLVED"


def test_hazard_validation_and_not_found(client):
    assert client.post("/api/hazards", json=hazard_payload(latitude=91)).status_code == 422
    assert client.patch("/api/hazards/999/status", json={"status": "ACTIVE"}).status_code == 404
    assert client.get("/api/hazards/999").status_code == 404
