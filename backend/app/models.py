from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from app.database import Base

class Hazard(Base):
    __tablename__ = "hazards"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False, default="pothole")
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    severity = Column(String, nullable=False)  # 'low', 'medium', 'high'
    status = Column(String, nullable=False, default="DETECTED")  # DETECTED, VERIFIED, ACTIVE, REPORTED_REPAIRED, RESOLVED
    confidence = Column(Float, nullable=False, default=0.0)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    source = Column(String(100), nullable=False, default="manual")
    # Retained for the existing observation-processing service.
    first_detected = Column(String, nullable=False)
    last_detected = Column(String, nullable=False)

    # Relationship to individual observations
    observations = relationship("Observation", back_populates="hazard", cascade="all, delete-orphan")


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True)
    hazard_id = Column(Integer, ForeignKey("hazards.id", ondelete="CASCADE"), nullable=True)
    source = Column(String, nullable=False)  # 'ai_camera', 'manual_report', etc.
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(String, nullable=False)
    sensor_evidence = Column(String, nullable=True)  # Filepath/URL of captured screenshot

    # Relationship back to aggregated hazard
    hazard = relationship("Hazard", back_populates="observations")
