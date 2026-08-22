"""Structured data models returned by the perception module."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One model prediction, with ``bbox`` in pixel ``[x1, y1, x2, y2]`` order."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation for API or logging boundaries."""
        value = asdict(self)
        value["bbox"] = list(self.bbox)
        return value
