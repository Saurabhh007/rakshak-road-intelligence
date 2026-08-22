from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HazardSeverity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class HazardStatus(str, Enum):
    detected = "DETECTED"
    verified = "VERIFIED"
    active = "ACTIVE"
    reported_repaired = "REPORTED_REPAIRED"
    resolved = "RESOLVED"


class HazardCreate(BaseModel):
    type: str = Field(default="pothole", min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    confidence: float = Field(ge=0, le=1)
    severity: HazardSeverity
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: HazardStatus = HazardStatus.detected
    source: str = Field(min_length=1, max_length=100)

    @field_validator("type", "source")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str | HazardStatus) -> str | HazardStatus:
        return value.upper() if isinstance(value, str) else value


class Hazard(HazardCreate):
    id: int
    model_config = ConfigDict(from_attributes=True)


class HazardStatusUpdate(BaseModel):
    status: HazardStatus

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value: str | HazardStatus) -> str | HazardStatus:
        return value.upper() if isinstance(value, str) else value

# Observations Schemas
class ObservationBase(BaseModel):
    source: str
    latitude: float
    longitude: float
    confidence: float
    timestamp: str
    sensor_evidence: Optional[str] = None

class ObservationCreate(ObservationBase):
    hazard_id: Optional[int] = None

class Observation(ObservationBase):
    id: int
    hazard_id: Optional[int] = None

    class Config:
        from_attributes = True


# Telemetry Schemas
class TelemetryRequest(BaseModel):
    latitude: float
    longitude: float
    speed_kmh: float = 0.0
    imu_accel_z: float = 9.8
    gps_source: str = "SIMULATED"  # "LIVE" or "SIMULATED"

class TelemetryWarning(BaseModel):
    hazard_id: int
    type: str
    distance_meters: float
    bearing: float
    severity: str
    status: str

class SystemStatus(BaseModel):
    ai_engine: str  # "REAL" or "SIMULATED"
    gps: str        # "LIVE" or "SIMULATED"
    backend: str    # "CONNECTED" or "OFFLINE"

class TelemetryResponse(BaseModel):
    warning_active: bool
    warning: Optional[TelemetryWarning] = None
    system_status: SystemStatus


# Road Health Schemas
class SeverityBreakdown(BaseModel):
    high_severity: int = 0
    medium_severity: int = 0
    low_severity: int = 0

class RoadHealthResponse(BaseModel):
    prototype_road_health_score: float
    evaluation_radius_meters: float
    active_hazards: int
    breakdown: SeverityBreakdown
    notice: str = "PROTOTYPE ONLY - NOT FOR ENGINEERING USE"


# Simulation Control Schemas
class SimulationResetRequest(BaseModel):
    mode: str = "DEMO"  # "DEMO" or "REAL"

class SimulationResetResponse(BaseModel):
    status: str
    ai_mode: str
    gps_source: str
