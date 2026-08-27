"""Environment-based configuration for the perception module."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DetectorConfig:
    """Detector settings. ``model_path`` is intentionally unset by default."""

    model_path: Path | None = None
    confidence_threshold: float = 0.60
    pothole_class_name: str = "D40"

    @classmethod
    def from_environment(cls) -> "DetectorConfig":
        configured_path = os.getenv("RAKSHAK_MODEL_PATH")
        threshold = float(os.getenv("RAKSHAK_CONFIDENCE_THRESHOLD", "0.60"))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("RAKSHAK_CONFIDENCE_THRESHOLD must be between 0 and 1")
        return cls(
            model_path=Path(configured_path).expanduser() if configured_path else None,
            confidence_threshold=threshold,
            pothole_class_name=os.getenv("RAKSHAK_POTHOLE_CLASS_NAME", "D40"),
        )
