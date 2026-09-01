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
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:5175,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,http://127.0.0.1:5175,*"
    LOG_LEVEL: str = "INFO"
    
    # AI Config
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "ai" / "model" / "yolo12s_RDD2022_best.pt"))
    MODEL_TYPE: str = os.getenv("MODEL_TYPE", "yolo")  # e.g., yolov8, yolo11, etc.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    LIVE_CAMERA_CANDIDATE_THRESHOLD: float = float(os.getenv("LIVE_CAMERA_CANDIDATE_THRESHOLD", "0.25"))
    UPLOAD_IMAGE_THRESHOLD: float = float(os.getenv("UPLOAD_IMAGE_THRESHOLD", "0.25"))
    DETECTION_CLASS: str = os.getenv("DETECTION_CLASS", "D40")
    AI_FRAME_INTERVAL: int = int(os.getenv("AI_FRAME_INTERVAL", "5")) # Process 1 in every N frames
    AI_MAX_FPS: int = int(os.getenv("AI_MAX_FPS", "5")) # Max FPS for AI processing
    TEMPORAL_WINDOW_SECONDS: float = float(os.getenv("TEMPORAL_WINDOW_SECONDS", "5.0"))
    
    # Geofence & Routing Config
    HAZARD_CLUSTER_RADIUS_METERS: float = float(os.getenv("HAZARD_CLUSTER_RADIUS_METERS", "10.0"))
    WARNING_DISTANCE_METERS: float = float(os.getenv("WARNING_DISTANCE_METERS", "120.0"))
    ROAD_HEALTH_RADIUS_METERS: float = float(os.getenv("ROAD_HEALTH_RADIUS_METERS", "500.0"))
    
    # Simulation/Inputs Config
    VIDEO_SOURCE: str = resolve_video_source(os.getenv("VIDEO_SOURCE", "ai/test_videos/road_video.mp4"))
    # A live camera is opt-in.  Never treat the demo-video setting or a private
    # developer IP as a live camera fallback.
    CAMERA_URL: str = os.getenv("CAMERA_URL", os.getenv("DEFAULT_CAMERA_URL", "0"))
    DEMO_VIDEO_SOURCE: str = resolve_video_source(os.getenv("DEMO_VIDEO_SOURCE", "ai/test_videos/road_video.mp4"))
    DEMO_IMAGE_PATH: str = resolve_project_path(os.getenv("DEMO_IMAGE_PATH", "ai/test_images/sample_road11.jpg"))
    DEMO_IMAGE_THRESHOLD: float = float(os.getenv("DEMO_IMAGE_THRESHOLD", "0.25"))
    # Prototype Test Fallback Location (used ONLY when hardware GPS is unavailable on laptop)
    DEFAULT_TEST_LATITUDE: float = float(os.getenv("DEFAULT_TEST_LATITUDE", "18.6056"))
    DEFAULT_TEST_LONGITUDE: float = float(os.getenv("DEFAULT_TEST_LONGITUDE", "73.7525"))
    DEFAULT_GPS_SOURCE: str = os.getenv("DEFAULT_GPS_SOURCE", "TEST_FALLBACK")
    
    # Keep the normal production verification rule.  A judge may explicitly
    # configure a smaller, still-real observation count for a short live demo.
    REAL_CAMERA_VERIFICATION_OBSERVATIONS: int = int(os.getenv("REAL_CAMERA_VERIFICATION_OBSERVATIONS", "3"))
    DEMO_VERIFICATION_OBSERVATIONS: int = int(os.getenv("DEMO_VERIFICATION_OBSERVATIONS", "3"))
    
    class Config:
        case_sensitive = True

    @model_validator(mode="after")
    def resolve_configured_paths(self):
        self.MODEL_PATH = resolve_project_path(self.MODEL_PATH)
        self.VIDEO_SOURCE = resolve_video_source(self.VIDEO_SOURCE)
        self.DEMO_VIDEO_SOURCE = resolve_video_source(self.DEMO_VIDEO_SOURCE)
        self.DEMO_IMAGE_PATH = resolve_project_path(self.DEMO_IMAGE_PATH)
        if not self.CAMERA_URL and ("://" in self.VIDEO_SOURCE or self.VIDEO_SOURCE.isdigit()):
            self.CAMERA_URL = self.VIDEO_SOURCE
        return self

settings = Settings()
