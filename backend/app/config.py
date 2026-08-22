import os
from pathlib import Path
from pydantic import model_validator
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_video_source(value: str) -> str:
    """Keep OpenCV network URLs intact and resolve local files from the project root."""
    if "://" in value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def resolve_project_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)

class Settings(BaseSettings):
    # App General Config
    APP_NAME: str = "RAKSHAK Road Intelligence Engine"
    API_V1_STR: str = "/api"
    
    # DB config
    DATABASE_URL: str = "sqlite:///./rakshak.db"
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    LOG_LEVEL: str = "INFO"
    
    # AI Config
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "ai" / "model" / "yolo12s_RDD2022_best.pt"))
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "yolo")  # e.g., yolov8, yolo11, etc.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    DETECTION_CLASS: str = os.getenv("DETECTION_CLASS", "D40")
    AI_FRAME_INTERVAL: int = int(os.getenv("AI_FRAME_INTERVAL", "5")) # Process 1 in every N frames
    AI_MAX_FPS: int = int(os.getenv("AI_MAX_FPS", "5")) # Max FPS for AI processing
    
    # Geofence & Routing Config
    HAZARD_CLUSTER_RADIUS_METERS: float = float(os.getenv("HAZARD_CLUSTER_RADIUS_METERS", "10.0"))
    WARNING_DISTANCE_METERS: float = float(os.getenv("WARNING_DISTANCE_METERS", "120.0"))
    ROAD_HEALTH_RADIUS_METERS: float = float(os.getenv("ROAD_HEALTH_RADIUS_METERS", "500.0"))
    
    # Simulation/Inputs Config
    VIDEO_SOURCE: str = resolve_video_source(os.getenv("VIDEO_SOURCE", "ai/test_videos/road_video.mp4"))
    GPS_MODE: str = os.getenv("GPS_MODE", "SIMULATED")  # LIVE or SIMULATED
    
    class Config:
        case_sensitive = True

    @model_validator(mode="after")
    def resolve_configured_paths(self):
        self.MODEL_PATH = resolve_project_path(self.MODEL_PATH)
        self.VIDEO_SOURCE = resolve_video_source(self.VIDEO_SOURCE)
        return self

settings = Settings()
