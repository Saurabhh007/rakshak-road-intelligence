import base64
import logging
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ai.config import DetectorConfig
from ai.detector import MockPotholeDetector, PotholeDetector
from ai.models import Detection
from ai.src.utils import draw_bounding_boxes
from app import schemas
from app.config import settings
from app.database import SessionLocal
from app.services.verification import process_observation

logger = logging.getLogger(__name__)


class VideoProcessor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(VideoProcessor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Thread safety controls
        self.frame_lock = threading.RLock()
        self.latest_frame_bytes: Optional[bytes] = None

        # Mode state: "LIVE_CAMERA", "SYSTEM_DEMO", "IMAGE_FALLBACK"
        # Determine initial mode from settings
        initial_mode = "LIVE_CAMERA" if "://" in settings.VIDEO_SOURCE else "SYSTEM_DEMO"
        self.mode = initial_mode
        self.restart_source = False

        # Vehicle coordinates telemetry state
        self.latitude: Optional[float] = None
        self.longitude: Optional[float] = None
        self.gps_source = "N/A"
        self.last_save_time: float = 0.0
        self.diagnostics = {"frames_received": 0, "inference_executed": 0, "raw_detections_count": 0,
                            "filtered_detections_count": 0, "pothole_detected": False, "last_error": None}

        # RDD2022 YOLOv12 real detector (production model)
        self.detector = PotholeDetector(
            DetectorConfig(
                model_path=Path(settings.MODEL_PATH),
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                pothole_class_name=settings.DETECTION_CLASS,
            )
        )
        self.mock_detector = MockPotholeDetector()

        self.camera_connected = False
        self.camera_status_str = "CAMERA_CONNECTING" if initial_mode == "LIVE_CAMERA" else "N/A"
        # Daemon Thread State Controls
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

    def start(self):
        with self.frame_lock:
            if self.is_running:
                return
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Video processor started in mode %s", self.mode)

    def stop(self):
        with self.frame_lock:
            self.is_running = False
            self.restart_source = True
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            logger.info("Video processor stopped")

    def switch_mode(self, mode_name: str) -> dict:
        """Thread-safe switch between LIVE_CAMERA, SYSTEM_DEMO, IMAGE_FALLBACK, and UPLOAD_IMAGE."""
        mode_upper = mode_name.upper().strip()
        if mode_upper in ["A", "LIVE", "LIVE_CAMERA"]:
            target_mode = "LIVE_CAMERA"
        elif mode_upper in ["B", "DEMO", "SYSTEM_DEMO", "VIDEO"]:
            target_mode = "SYSTEM_DEMO"
        elif mode_upper in ["C", "IMAGE", "REAL_IMAGE", "IMAGE_FALLBACK"]:
            target_mode = "IMAGE_FALLBACK"
        elif mode_upper in ["D", "UPLOAD", "UPLOAD_IMAGE", "IMAGE_UPLOAD"]:
            target_mode = "UPLOAD_IMAGE"
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        with self.frame_lock:
            if self.mode == target_mode:
                return self.get_camera_status()
            self.mode = target_mode
            self.restart_source = True
            if target_mode in ["IMAGE_FALLBACK", "UPLOAD_IMAGE"]:
                self.camera_connected = False
                self.camera_status_str = "N/A"
            elif target_mode == "SYSTEM_DEMO":
                self.camera_connected = False
                self.camera_status_str = "N/A"
            elif target_mode == "LIVE_CAMERA":
                self.camera_connected = False
                self.camera_status_str = "CAMERA_CONNECTING"

        logger.info("Switched VideoProcessor mode to %s", target_mode)
        return self.get_camera_status()

    def set_coordinates(self, latitude: float, longitude: float, gps_source: str = "LIVE"):
        """Updates the processor's vehicle coordinate telemetry state."""
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("GPS coordinates are outside valid latitude/longitude ranges")
        with self.frame_lock:
            self.latitude = latitude
            self.longitude = longitude
            self.gps_source = gps_source.upper()

    def get_latest_frame(self) -> Optional[bytes]:
        """Thread-safe retrieval of the latest processed JPEG frame."""
        with self.frame_lock:
            return self.latest_frame_bytes

    @property
    def source_type(self) -> str:
        if self.mode == "LIVE_CAMERA":
            if not settings.CAMERA_URL:
                return "none"
            if settings.CAMERA_URL.isdigit():
                return "webcam"
            return "network" if "://" in settings.CAMERA_URL else "file"
        elif self.mode == "SYSTEM_DEMO":
            return "file"
        return "none"

    def get_camera_status(self) -> dict[str, object]:
        """Return truthful camera & mode state without exposing stream URLs or IP addresses."""
        with self.frame_lock:
            ai_status = "REAL" if not self.detector.is_mock else "AI_UNAVAILABLE"
            if self.diagnostics["last_error"]:
                ai_status = "AI_ERROR"
            if self.mode == "SYSTEM_DEMO":
                ai_engine = "SIMULATED"
                detection_source = "DEMO/SIMULATED"
                gps_source = "SIMULATED DEMO ROUTE" if self.gps_source == "DEMO_SIMULATED" else "N/A"
                input_type = "VIDEO"
            elif self.mode == "LIVE_CAMERA":
                ai_engine = ai_status
                detection_source = "REAL INFERENCE"
                if self.latitude is None:
                    gps_source = "N/A"
                elif self.gps_source == "DEMO_SIMULATED":
                    gps_source = "SIMULATED DEMO ROUTE"
                else:
                    gps_source = "LIVE" if self.gps_source == "LIVE" else "N/A"
                input_type = "LAPTOP WEBCAM" if (settings.CAMERA_URL.isdigit() or not settings.CAMERA_URL) else "LIVE CAMERA"
            elif self.mode == "UPLOAD_IMAGE":
                ai_engine = ai_status
                detection_source = "REAL AI"
                gps_source = "N/A"
                input_type = "UPLOADED IMAGE"
            else:  # IMAGE_FALLBACK
                ai_engine = ai_status
                detection_source = "REAL INFERENCE"
                gps_source = "N/A"
                input_type = "IMAGE (sample_road11.jpg)"

            return {
                "mode": self.mode,
                "connected": self.camera_connected,
                "camera_status": self.camera_status_str,
                "source_type": self.source_type,
                "ai_engine": ai_engine,
                "input_type": input_type,
                "detection_source": detection_source,
                "gps_source": gps_source,
                "backend": "CONNECTED",
                "diagnostics": dict(self.diagnostics),
            }

    def run_real_image_inference(
        self, image_path: Optional[str] = None, threshold: Optional[float] = None,
        associate_demo_route: bool = False
    ) -> dict:
        """Execute real RDD2022 YOLO inference on sample_road11.jpg with isolated threshold 0.25.

        Does NOT write to production DB and does NOT place markers on the live map.
        """
        img_path = Path(image_path or settings.DEMO_IMAGE_PATH)
        conf_threshold = threshold if threshold is not None else settings.DEMO_IMAGE_THRESHOLD

        if self.detector.is_mock or self.detector.model is None:
            return {
                "status": "AI_UNAVAILABLE",
                "image_name": img_path.name,
                "threshold": conf_threshold,
                "detections": [],
                "annotated_image": None,
                "ai_mode": "AI_UNAVAILABLE",
                "is_simulated": False,
                "source": "none",
            }

        if not img_path.exists():
            return {
                "status": "AI_UNAVAILABLE",
                "image_name": img_path.name,
                "threshold": conf_threshold,
                "detections": [],
                "annotated_image": None,
                "ai_mode": "AI_UNAVAILABLE",
                "is_simulated": False,
                "source": "image_not_found",
            }

        img = cv2.imread(str(img_path))
        if img is None:
            return {
                "status": "AI_UNAVAILABLE",
                "image_name": img_path.name,
                "threshold": conf_threshold,
                "detections": [],
                "annotated_image": None,
                "ai_mode": "AI_UNAVAILABLE",
                "is_simulated": False,
                "source": "image_decode_error",
            }

        # Run real inference with isolated threshold
        detections: List[Detection] = self.detector.detect(img, confidence_threshold=conf_threshold)
        if associate_demo_route and detections:
            self._save_detections(detections, source_tag="ai_image")

        # Draw bounding boxes on image copy
        annotated_img = draw_bounding_boxes(
            img.copy(), [detection.to_dict() for detection in detections]
        )

        # Encode to JPEG base64 data URI
        ret, buffer = cv2.imencode(".jpg", annotated_img)
        annotated_b64 = None
        if ret:
            b64_str = base64.b64encode(buffer).decode("utf-8")
            annotated_b64 = f"data:image/jpeg;base64,{b64_str}"

        return {
            "status": "ok",
            "image_name": img_path.name,
            "threshold": conf_threshold,
            "detections": [
                {
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": [round(x, 1) for x in d.bbox],
                }
                for d in detections
            ],
            "annotated_image": annotated_b64,
            "ai_mode": "REAL",
            "is_simulated": False,
            "source": "real_inference",
        }

    def run_uploaded_image_inference(
        self, image_bytes: bytes, filename: Optional[str] = None, threshold: Optional[float] = None,
        associate_demo_route: bool = False
    ) -> dict:
        """OPTION D: Execute real RDD2022 YOLO inference on an uploaded image.

        Processes image in-memory. Does NOT write to production DB and does NOT place markers on the live map.
        """
        conf_threshold = threshold if threshold is not None else settings.UPLOAD_IMAGE_THRESHOLD

        if self.detector.is_mock or self.detector.model is None:
            return {
                "status": "AI_UNAVAILABLE",
                "input_type": "uploaded_image",
                "ai_mode": "AI_UNAVAILABLE",
                "is_simulated": False,
                "filename": filename,
                "threshold": conf_threshold,
                "detections": [],
                "annotated_image": None,
                "message": "Real AI model weights are unavailable.",
            }

        # Safe in-memory OpenCV decoding
        try:
            img_np = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Failed to decode uploaded image bytes: %s", e)
            img = None

        if img is None:
            return {
                "status": "error",
                "input_type": "uploaded_image",
                "ai_mode": "real_inference",
                "is_simulated": False,
                "filename": filename,
                "threshold": conf_threshold,
                "detections": [],
                "annotated_image": None,
                "message": "Invalid or unreadable image file.",
            }

        # Run real YOLO inference with isolated demo/upload threshold
        detections: List[Detection] = self.detector.detect(img, confidence_threshold=conf_threshold)
        logger.info("UPLOAD INFERENCE_EXECUTED FILTERED_DETECTIONS_COUNT=%d", len(detections))
        if associate_demo_route and detections:
            self._save_detections(detections, source_tag="ai_uploaded_image")

        # Draw bounding boxes on image copy
        annotated_img = draw_bounding_boxes(
            img.copy(), [detection.to_dict() for detection in detections]
        )
        ret, encoded_img = cv2.imencode(".jpg", annotated_img)
        annotated_b64 = None
        if ret:
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(
                encoded_img.tobytes()
            ).decode("utf-8")

        # Map detections to output schema
        formatted_detections = []
        for d in detections:
            class_id = getattr(d, "class_id", None)
            if class_id is None:
                class_id = 3 if d.class_name == "D40" else 0
            formatted_detections.append({
                "class_id": class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": [round(x, 1) for x in d.bbox],
            })

        return {
            "status": "ok",
            "input_type": "uploaded_image",
            "ai_mode": "real_inference",
            "is_simulated": False,
            "filename": filename,
            "threshold": conf_threshold,
            "detections": formatted_detections,
            "annotated_image": annotated_b64,
            "message": f"Detected {len(formatted_detections)} pothole(s)" if formatted_detections else "NO DETECTION ABOVE CONFIDENCE THRESHOLD",
        }

    def _set_camera_connected(self, connected: bool, status_str: str = "OFFLINE") -> None:
        with self.frame_lock:
            self.camera_connected = connected
            self.camera_status_str = status_str
            if not connected:
                self.latest_frame_bytes = None

    def _record_frame(self) -> None:
        with self.frame_lock:
            self.diagnostics["frames_received"] += 1

    def _run_real_inference(
        self, frame: np.ndarray, frame_index: int, confidence_threshold: Optional[float] = None
    ) -> List[Detection]:
        """Execute only the loaded real model; expose failures rather than simulating them."""
        try:
            conf_to_use = (
                confidence_threshold
                if confidence_threshold is not None
                else settings.LIVE_CAMERA_CANDIDATE_THRESHOLD
            )
            detections = self.detector.detect(frame, frame_index, confidence_threshold=conf_to_use)
            with self.frame_lock:
                self.diagnostics["inference_executed"] += 1
                self.diagnostics["raw_detections_count"] = len(detections)
                self.diagnostics["filtered_detections_count"] = len(detections)
                self.diagnostics["pothole_detected"] = bool(detections)
                self.diagnostics["last_error"] = None
            logger.info(
                "INFERENCE_EXECUTED RAW_DETECTIONS_COUNT=%d FILTERED_DETECTIONS_COUNT=%d (conf>=%.2f)",
                len(detections),
                len(detections),
                conf_to_use,
            )
            if detections:
                logger.info("POTHOLE_DETECTED count=%d", len(detections))
            return detections
        except Exception as error:
            with self.frame_lock:
                self.diagnostics["last_error"] = "AI inference failed"
            logger.exception("AI inference failed on live frame")
            return []

    def _run_loop(self):
        while True:
            with self.frame_lock:
                if not self.is_running:
                    break
                current_mode = self.mode
                self.restart_source = False

            if current_mode in ["IMAGE_FALLBACK", "UPLOAD_IMAGE"]:
                time.sleep(0.2)
                continue

            if current_mode == "LIVE_CAMERA":
                self._run_live_camera_loop()
            elif current_mode == "SYSTEM_DEMO":
                self._run_system_demo_loop()

    def _run_live_camera_loop(self):
        camera_url = settings.CAMERA_URL
        if not camera_url:
            self._set_camera_connected(False, "CAMERA_OFFLINE")
            logger.warning("CAMERA_OFFLINE: no CAMERA_URL/DEFAULT_CAMERA_URL is configured")
            time.sleep(1.0)
            return
        is_mjpeg = camera_url.startswith(("http://", "https://"))
        if is_mjpeg:
            try:
                stream = urllib.request.urlopen(camera_url, timeout=2)
                self._set_camera_connected(False, "CAMERA_CONNECTING")
                logger.info("CAMERA_CONNECTING: HTTP camera stream opened; awaiting JPEG frame")
            except Exception as e:
                self._set_camera_connected(False, "CAMERA_OFFLINE")
                logger.error("CAMERA_OFFLINE: unable to open configured camera stream (%s)", type(e).__name__)
                # Sleep and retry, checking if mode changed
                for _ in range(25):
                    if not self.is_running or self.restart_source:
                        return
                    time.sleep(0.2)
                return

            bytes_data = bytearray()
            frame_delay = 1.0 / 30.0
            frame_index = 0

            while self.is_running and not self.restart_source:
                start_time = time.time()
                frame = None
                ret = False
                try:
                    while True:
                        a = bytes_data.find(b"\xff\xd8")
                        if a != -1:
                            b = bytes_data.find(b"\xff\xd9", a + 2)
                            if b != -1:
                                jpg_bytes = bytes_data[a : b + 2]
                                del bytes_data[: b + 2]
                                frame = cv2.imdecode(
                                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                                )
                                if frame is not None:
                                    ret = True
                                    break
                                continue
                        else:
                            if len(bytes_data) > 1024 * 1024:
                                bytes_data.clear()

                        chunk = stream.read(8192)
                        if not chunk:
                            break
                        bytes_data.extend(chunk)
                except Exception as e:
                    logger.error("Error reading from MJPEG stream: %s", e)

                if not ret or frame is None:
                    self._set_camera_connected(False, "CAMERA_OFFLINE")
                    logger.warning("Camera stream dropped; reconnecting")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    break

                self._record_frame()
                self._set_camera_connected(True, "CAMERA_ACTIVE")
                logger.debug("FRAME_RECEIVED from live camera")
                # Real RDD2022 inference on camera frames only (no simulation fallback)
                detections: List[Detection] = []
                if frame_index % settings.AI_FRAME_INTERVAL == 0:
                    if not self.detector.is_mock:
                        detections = self._run_real_inference(frame, frame_index, confidence_threshold=settings.LIVE_CAMERA_CANDIDATE_THRESHOLD)
                        if len(detections) > 0:
                            self._save_detections(detections, source_tag="ai_camera")

                annotated_frame = draw_bounding_boxes(
                    frame, [detection.to_dict() for detection in detections]
                )
                ret_enc, encoded_frame = cv2.imencode(".jpg", annotated_frame)
                if ret_enc:
                    with self.frame_lock:
                        self.latest_frame_bytes = encoded_frame.tobytes()

                frame_index += 1
                elapsed = time.time() - start_time
                time.sleep(max(0.001, frame_delay - elapsed))

            try:
                stream.close()
            except Exception:
                pass
        else:
            candidates = [int(camera_url)] if camera_url.isdigit() else [camera_url]
            if camera_url.isdigit() and int(camera_url) == 0:
                candidates.append(1)
            cap = None
            for src in candidates:
                test_cap = cv2.VideoCapture(src)
                if test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap = test_cap
                        break
                    test_cap.release()

            if cap is None:
                self._set_camera_connected(False, "CAMERA_OFFLINE")
                logger.error("CAMERA_OFFLINE: unable to open camera source (%s)", camera_url)
                for _ in range(25):
                    if not self.is_running or self.restart_source:
                        return
                    time.sleep(0.2)
                return

            self._set_camera_connected(False, "CAMERA_CONNECTING")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_delay = 1.0 / max(fps, 1.0)
            frame_index = 0

            while self.is_running and not self.restart_source:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    self._set_camera_connected(False, "CAMERA_OFFLINE")
                    break

                self._record_frame()
                self._set_camera_connected(True, "CAMERA_ACTIVE")
                logger.debug("FRAME_RECEIVED from live camera")
                detections: List[Detection] = []
                if frame_index % settings.AI_FRAME_INTERVAL == 0:
                    if not self.detector.is_mock:
                        detections = self._run_real_inference(frame, frame_index, confidence_threshold=settings.LIVE_CAMERA_CANDIDATE_THRESHOLD)
                        if len(detections) > 0:
                            self._save_detections(detections, source_tag="ai_camera")

                annotated_frame = draw_bounding_boxes(
                    frame, [detection.to_dict() for detection in detections]
                )
                ret_enc, encoded_frame = cv2.imencode(".jpg", annotated_frame)
                if ret_enc:
                    with self.frame_lock:
                        self.latest_frame_bytes = encoded_frame.tobytes()

                frame_index += 1
                elapsed = time.time() - start_time
                time.sleep(max(0.001, frame_delay - elapsed))

            cap.release()

    def _run_system_demo_loop(self):
        demo_video_path = settings.DEMO_VIDEO_SOURCE
        cap = cv2.VideoCapture(demo_video_path)
        if not cap.isOpened():
            logger.error("DEMO VIDEO FAILED: Unable to open %s", demo_video_path)
            time.sleep(1.0)
            return

        with self.frame_lock:
            self.camera_connected = False
            self.camera_status_str = "N/A"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / max(fps, 1.0)
        frame_index = 0

        while self.is_running and not self.restart_source:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                # Loop video seamlessly from start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_index = 0
                continue

            detections: List[Detection] = []

            if frame_index % settings.AI_FRAME_INTERVAL == 0:
                # In Option B (SYSTEM DEMO), use demo mock detector
                detections = self.mock_detector.detect(frame, frame_index)
                if len(detections) > 0:
                    self._save_detections(detections, source_tag="demo_simulated")

            annotated_frame = draw_bounding_boxes(
                frame, [detection.to_dict() for detection in detections]
            )
            ret_enc, encoded_frame = cv2.imencode(".jpg", annotated_frame)
            if ret_enc:
                with self.frame_lock:
                    self.latest_frame_bytes = encoded_frame.tobytes()

            frame_index += 1
            elapsed = time.time() - start_time
            time.sleep(max(0.001, frame_delay - elapsed))

        cap.release()

    def _save_detections(self, detections: List[Detection], source_tag: str = "ai_camera"):
        """Helper method running database write aggregation on a separate thread session."""
        now_ts = time.time()
        # Prevent duplicate database write flooding (minimum 0.5s between consecutive webcam save cycles)
        if now_ts - self.last_save_time < 0.5:
            return
        self.last_save_time = now_ts

        db = SessionLocal()
        try:
            is_simulated = (source_tag == "demo_simulated")
            with self.frame_lock:
                live_lat, live_lon, live_gps_source = self.latitude, self.longitude, self.gps_source

            for det in detections:
                logger.info(
                    "[AI DETECTION] Class: %s (Pothole), Confidence: %.4f, BBox: %s",
                    det.class_name,
                    det.confidence,
                    [round(float(c), 1) for c in det.bbox],
                )

                # Determine coordinates: preserve live GPS if available; otherwise use test fallback
                if live_gps_source in {"LIVE", "DEMO_SIMULATED"} and live_lat is not None and live_lon is not None:
                    latitude = live_lat
                    longitude = live_lon
                    gps_source = "demo_simulated" if live_gps_source == "DEMO_SIMULATED" else "live"
                    gps_is_simulated = (live_gps_source == "DEMO_SIMULATED")
                else:
                    # Laptop webcam prototype fallback (clearly tagged as test fallback)
                    latitude = live_lat if live_lat is not None else settings.DEFAULT_TEST_LATITUDE
                    longitude = live_lon if live_lon is not None else settings.DEFAULT_TEST_LONGITUDE
                    gps_source = "test_fallback"
                    gps_is_simulated = True

                logger.info(
                    "[LOCATION ATTACHED] Lat: %.6f, Lon: %.6f, GPS Source: %s (Simulated: %s)",
                    latitude,
                    longitude,
                    gps_source,
                    gps_is_simulated,
                )

                now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                obs = schemas.ObservationCreate(
                    source=source_tag,
                    latitude=latitude,
                    longitude=longitude,
                    confidence=det.confidence,
                    timestamp=now_iso,
                    sensor_evidence=None,
                    is_simulated=is_simulated,
                    gps_source=gps_source,
                    gps_is_simulated=gps_is_simulated,
                )
                logger.info(
                    "[HAZARD EVENT CREATED] Source: %s, Real AI: %s, GPS Source: %s",
                    source_tag,
                    not is_simulated,
                    gps_source,
                )

                hazard = process_observation(db, obs)
                logger.info(
                    "[EVENT STORED] Hazard ID: PTH-%03d, Status: %s, Severity: %s, Lat: %.6f, Lon: %.6f",
                    hazard.id,
                    hazard.status,
                    hazard.severity,
                    hazard.latitude,
                    hazard.longitude,
                )
        except Exception:
            logger.exception("VideoProcessor database write error")
        finally:
            db.close()


video_processor = VideoProcessor()
