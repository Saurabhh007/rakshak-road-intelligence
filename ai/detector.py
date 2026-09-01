import logging
from typing import Protocol

import numpy as np
import torch

from .config import DetectorConfig
from .models import Detection

logger = logging.getLogger(__name__)


def _patch_yolo12_attention() -> None:
    """Patch Ultralytics YOLOv12 area attention for compatibility across PyTorch versions."""
    try:
        from ultralytics.nn.modules.block import AAttn

        def compatibility_forward(self, x):
            B, C, H, W = x.shape
            N = H * W
            all_head_dim = self.num_heads * self.head_dim
            qkv = self.qkv(x)
            qk, v = qkv.split([all_head_dim * 2, all_head_dim], dim=1)
            pp = self.pe(v)
            qk = qk.flatten(2).transpose(1, 2)
            v = v.flatten(2).transpose(1, 2)

            if self.area > 1:
                qk = qk.reshape(B * self.area, N // self.area, all_head_dim * 2)
                v = v.reshape(B * self.area, N // self.area, all_head_dim)
                B, N, _ = qk.shape

            q, k = qk.split([all_head_dim, all_head_dim], dim=2)
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = v @ attn.transpose(-2, -1)
            x = x.permute(0, 3, 1, 2)

            if self.area > 1:
                x = x.reshape(B // self.area, N * self.area, C)
                B, N, _ = x.shape

            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return self.proj(x + pp)

        AAttn.forward = compatibility_forward
    except Exception as e:
        logger.debug("YOLOv12 attention patch not needed or skipped: %s", e)


_patch_yolo12_attention()


class FrameDetector(Protocol):
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[Detection]: ...


class MockPotholeDetector:
    """Deterministic demo implementation; it performs no ML inference."""

    is_mock = True

    def detect(self, frame: np.ndarray, frame_index: int = 0, confidence_threshold: float | None = None) -> list[Detection]:
        _validate_frame(frame)
        # A stable, opt-in-by-frame-index visual demo for development and tests.
        if not 110 <= frame_index <= 150:
            return []
        height, width = frame.shape[:2]
        progress = (frame_index - 110) / 40.0
        box_width, box_height = 30 + progress * 150, 15 + progress * 70
        center_x, center_y = width * 0.48, height * (0.55 + progress * 0.25)
        return [Detection(
            class_name="pothole",
            confidence=round(0.91 + progress * 0.04, 4),
            bbox=(center_x - box_width / 2, center_y - box_height / 2,
                  center_x + box_width / 2, center_y + box_height / 2),
        )]


class PotholeDetector:
    """Run configured Ultralytics YOLO weights on OpenCV images and video frames.

    A real detector is only activated when ``RAKSHAK_MODEL_PATH`` points to an
    existing weights file. Without supplied weights, ``is_mock`` is true.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig.from_environment()
        self.model = None
        self.is_mock = True
        self._fallback = MockPotholeDetector()
        self._load_model()

    def _load_model(self) -> None:
        model_path = self.config.model_path
        if model_path is None:
            logger.warning("No RAKSHAK_MODEL_PATH configured; using mock detector")
            return
        if not model_path.is_file():
            logger.warning("Configured model weights were not found at %s; using mock detector", model_path)
            return
        try:
            _patch_yolo12_attention()
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.is_mock = False
            logger.info("Loaded configured YOLO weights from %s", model_path)
        except Exception as error:
            raise RuntimeError(f"Could not load configured YOLO weights from {model_path}") from error

    def detect(self, frame: np.ndarray, frame_index: int = 0, confidence_threshold: float | None = None) -> list[Detection]:
        """Run inference for one BGR OpenCV frame with optional confidence threshold override."""
        _validate_frame(frame)
        if self.is_mock:
            return self._fallback.detect(frame, frame_index, confidence_threshold)
        assert self.model is not None
        conf_to_use = confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
        result = self.model(frame, verbose=False, conf=conf_to_use)[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = str(names[class_id])
            if class_name.casefold() != self.config.pothole_class_name.casefold():
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(class_name, confidence, (x1, y1, x2, y2), class_id=class_id))
        return detections


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR NumPy array with shape (height, width, 3)")

