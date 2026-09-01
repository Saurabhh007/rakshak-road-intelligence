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
    is_simulated: bool = False
    gps_source: str = "unknown"
    gps_is_simulated: bool = False

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

    @field_validator("timestamp", mode="after")
    @classmethod
    def normalize_timestamp_to_utc(cls, value: datetime) -> datetime:
        """SQLite drops tzinfo; all stored hazard times are UTC by contract."""
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


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
    is_simulated: bool = False
    gps_source: str = "unknown"
    gps_is_simulated: bool = False

class ObservationCreate(ObservationBase):
    hazard_id: Optional[int] = None

class Observation(ObservationBase):
    id: int
    hazard_id: Optional[int] = None

    class Config:
        from_attributes = True


# Telemetry Schemas
class TelemetryRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
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
    ai_engine: str  # "REAL" or "SIMULATED" or "AI_UNAVAILABLE"
    gps: str        # "LIVE" or "SIMULATED" or "DEMO TELEMETRY" or "N/A"
    backend: str    # "CONNECTED" or "OFFLINE"
    camera: Optional[str] = "N/A"  # "ACTIVE", "OFFLINE", "UNAVAILABLE", "N/A"
    mode: Optional[str] = "SYSTEM_DEMO"  # "LIVE_CAMERA", "SYSTEM_DEMO", "IMAGE_FALLBACK"
    verification: Optional[str] = "ACTIVE"
    map: Optional[str] = "ACTIVE"
    warning_status: Optional[str] = "ACTIVE"

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


# Mode Selector Schemas
class ModeSwitchRequest(BaseModel):
    mode: str  # "LIVE_CAMERA", "SYSTEM_DEMO", "IMAGE_FALLBACK", "A", "B", "C"

class ModeSwitchResponse(BaseModel):
    status: str
    mode: str
    input_type: str
    ai_mode: str
    gps_mode: str
    camera_status: str
    detection_source: str


# Option C Real AI Image Inference Schemas
class RealInferenceDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: list[float]

class RealInferenceResponse(BaseModel):
    status: str  # "ok" or "AI_UNAVAILABLE"
    image_name: str
    threshold: float
    detections: list[RealInferenceDetection]
    annotated_image: Optional[str] = None
    ai_mode: str
    is_simulated: bool
    source: str


# Option D Upload Image Inference Schemas
class UploadInferenceDetection(BaseModel):
    class_id: Optional[int] = None
    class_name: str
    confidence: float
    bbox: list[float]

class UploadInferenceResponse(BaseModel):
    status: str  # "ok" or "AI_UNAVAILABLE" or "error"
    input_type: str = "uploaded_image"
    ai_mode: str = "real_inference"
    is_simulated: bool = False
    filename: Optional[str] = None
    threshold: float
    detections: list[UploadInferenceDetection]
    annotated_image: Optional[str] = None
    message: Optional[str] = None


