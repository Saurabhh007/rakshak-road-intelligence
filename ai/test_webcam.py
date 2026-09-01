"""RAKSHAK Live Webcam Perception Runner.

Supports laptop webcam (index 0 / 1), video files, and network streams.
Runs real RDD2022 YOLOv12 inference and renders live annotated preview with FPS.

Usage:
    python ai/test_webcam.py                 # Default webcam (index 0)
    python ai/test_webcam.py --camera 1      # Secondary webcam
    python ai/test_webcam.py --video ai/test_videos/road_video.mp4  # Video file
    python ai/test_webcam.py --conf 0.25     # Set confidence threshold (default: 0.25)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.config import DetectorConfig
from ai.detector import PotholeDetector, _patch_yolo12_attention
from ai.models import Detection
from ai.src.utils import draw_bounding_boxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAKSHAK.Webcam")


class CameraSource:
    """Modular camera source supporting laptop webcam indices, video files, and URLs."""

    def __init__(self, source: Union[int, str, Path] = 0):
        self.requested_source = source
        self.active_source: Optional[Union[int, str]] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> bool:
        """Attempt to open requested source with automatic webcam fallback (0 -> 1)."""
        candidate_sources: List[Union[int, str]] = []

        if isinstance(self.requested_source, int) or (
            isinstance(self.requested_source, str) and self.requested_source.isdigit()
        ):
            idx = int(self.requested_source)
            candidate_sources.append(idx)
            # Add alternate webcam index as fallback
            candidate_sources.append(1 if idx == 0 else 0)
        else:
            candidate_sources.append(str(self.requested_source))

        for src in candidate_sources:
            logger.info("Attempting to open video source: %s ...", src)
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                # Verify we can actually read a valid frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.cap = cap
                    self.active_source = src
                    self.is_opened = True
                    h, w = frame.shape[:2]
                    logger.info(
                        "Successfully opened video source '%s' (Resolution: %dx%d)",
                        src,
                        w,
                        h,
                    )
                    return True
                cap.release()
            logger.warning("Failed to open or capture from source: %s", src)

        self.is_opened = False
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the camera."""
        if not self.is_opened or self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Release OpenCV video capture safely."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info("Camera source released.")


def format_detection_dict(det: Detection) -> Dict[str, Any]:
    """Return clean structured detection result."""
    return {
        "type": "pothole" if det.class_name.upper() == "D40" else det.class_name.lower(),
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 4),
        "bbox": [round(float(coord), 1) for coord in det.bbox],
    }


def draw_hud_banner(
    frame: np.ndarray,
    fps: float,
    detection_count: int,
    conf_threshold: float,
    source_name: str,
) -> np.ndarray:
    """Draw a top HUD status banner onto the frame."""
    h, w = frame.shape[:2]
    banner_height = 42

    # Draw semi-transparent dark banner overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 17, 23), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & Engine Info
    title_text = "RAKSHAK ROAD INTELLIGENCE  |  YOLOv12s RDD2022"
    cv2.putText(
        frame,
        title_text,
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Metrics & Hotkeys
    metrics_text = (
        f"FPS: {fps:.1f}  |  Conf: {conf_threshold:.2f}  |  Potholes: {detection_count}  |  "
        f"Src: {source_name}  |  [Q] Quit  [C] Toggle Threshold"
    )
    cv2.putText(
        frame,
        metrics_text,
        (12, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Visual detection alert pill if potholes present
    if detection_count > 0:
        alert_w = 170
        cv2.rectangle(
            frame,
            (w - alert_w - 10, 6),
            (w - 10, banner_height - 6),
            (0, 0, 180),
            -1,
        )
        cv2.putText(
            frame,
            f"! {detection_count} POTHOLE(S) !",
            (w - alert_w + 2, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def run_webcam_pipeline(
    source_arg: Union[int, str] = 0,
    model_path: Optional[Path] = None,
    confidence_threshold: float = 0.25,
) -> None:
    """Main loop for running live webcam detection."""
    _patch_yolo12_attention()

    resolved_model_path = model_path or (PROJECT_ROOT / "ai" / "model" / "yolo12s_RDD2022_best.pt")
    if not resolved_model_path.exists():
        logger.error("Model weights file not found at: %s", resolved_model_path)
        print(f"\nERROR: Model weights not found at '{resolved_model_path}'.")
        return

    logger.info("Initializing RAKSHAK PotholeDetector with weights: %s", resolved_model_path)
    detector = PotholeDetector(
        DetectorConfig(
            model_path=resolved_model_path,
            confidence_threshold=confidence_threshold,
            pothole_class_name="D40",
        )
    )

    if detector.is_mock or detector.model is None:
        logger.error("Failed to load real YOLO model weights.")
        print("\nERROR: Could not load real YOLO model weights.")
        return

    logger.info("Real YOLOv12 RDD2022 detector loaded successfully.")

    # Initialize camera source
    camera = CameraSource(source_arg)
    if not camera.open():
        print("\n" + "=" * 60)
        print("ERROR: Could not open laptop webcam or video source.")
        print(f"Tried: {source_arg} and fallback index.")
        print("Please verify:")
        print(" 1. Your laptop webcam is connected and enabled.")
        print(" 2. Camera permissions are granted to Python/Windows apps.")
        print(" 3. No other application (Zoom, Teams, Browser) is locking the webcam.")
        print("=" * 60 + "\n")
        return

    window_name = "RAKSHAK - Live Webcam Road Intelligence"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 640)

    fps_history = []
    fps_smooth = 0.0
    frame_index = 0
    current_threshold = confidence_threshold

    print("\n" + "=" * 65)
    print(" RAKSHAK LIVE WEBCAM PERCEPTION RUNNING")
    print(" Model: YOLOv12s RDD2022 (Real Pothole Detector)")
    print(f" Active Source: {camera.active_source}")
    print(f" Confidence Threshold: {current_threshold}")
    print(" Controls:")
    print("   [Q] or [ESC] - Quit")
    print("   [C] - Toggle threshold between 0.25 and 0.60")
    print("   [S] - Save current frame snapshot")
    print("=" * 65 + "\n")

    try:
        while True:
            start_time = time.time()

            ret, frame = camera.read_frame()
            if not ret or frame is None:
                # If video file, loop back to start
                if isinstance(camera.active_source, str) and Path(camera.active_source).exists():
                    camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                logger.warning("Failed to read frame from webcam; exiting.")
                break

            # 1. Run real inference
            detections: List[Detection] = detector.detect(
                frame,
                frame_index=frame_index,
                confidence_threshold=current_threshold,
            )

            # 2. Print structured detection outputs to console when potholes detected
            if detections:
                structured = [format_detection_dict(d) for d in detections]
                logger.info(
                    "Frame #%04d: Detected %d pothole(s) -> %s",
                    frame_index,
                    len(detections),
                    structured,
                )

            # 3. Draw bounding boxes (displaying 'POTHOLE' for D40)
            annotated_frame = draw_bounding_boxes(
                frame, [d.to_dict() for d in detections]
            )

            # 4. Measure FPS
            elapsed = time.time() - start_time
            instant_fps = 1.0 / max(elapsed, 1e-4)
            fps_history.append(instant_fps)
            if len(fps_history) > 15:
                fps_history.pop(0)
            fps_smooth = sum(fps_history) / len(fps_history)

            # 5. Render HUD Banner
            src_label = (
                f"Webcam #{camera.active_source}"
                if isinstance(camera.active_source, int)
                else "Video File"
            )
            annotated_frame = draw_hud_banner(
                annotated_frame,
                fps=fps_smooth,
                detection_count=len(detections),
                conf_threshold=current_threshold,
                source_name=src_label,
            )

            # 6. Display frame
            cv2.imshow(window_name, annotated_frame)

            frame_index += 1

            # Check key presses
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                logger.info("User requested exit.")
                break
            elif key in (ord("c"), ord("C")):
                # Toggle threshold between 0.25 and 0.60
                current_threshold = 0.60 if current_threshold == 0.25 else 0.25
                logger.info("Confidence threshold toggled to: %.2f", current_threshold)
            elif key in (ord("s"), ord("S")):
                snap_path = f"webcam_snapshot_frame_{frame_index}.jpg"
                cv2.imwrite(snap_path, annotated_frame)
                logger.info("Saved snapshot: %s", snap_path)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        logger.info("RAKSHAK Webcam runner cleanly shut down.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAKSHAK Live Webcam & Video Pothole Perception Runner"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Laptop webcam device index (e.g. 0 or 1, default: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to prerecorded video file (e.g. ai/test_videos/road_video.mp4)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO model weights (default: ai/model/yolo12s_RDD2022_best.pt)",
    )

    args = parser.parse_args()

    source = args.video if args.video else args.camera
    model_path = Path(args.model) if args.model else None

    run_webcam_pipeline(
        source_arg=source,
        model_path=model_path,
        confidence_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
