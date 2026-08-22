import threading
import time
import cv2
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from app.config import settings
from app.database import SessionLocal
from app import schemas
from app.services.verification import process_observation
from ai.config import DetectorConfig
from ai.detector import PotholeDetector
from ai.models import Detection
from ai.src.utils import draw_bounding_boxes

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
        self.frame_lock = threading.Lock()
        self.latest_frame_bytes: Optional[bytes] = None
        
        # Vehicle coordinates telemetry state
        self.latitude: float = 12.9710
        self.longitude: float = 77.5930
        
        # The configured RDD2022 YOLOv12 detector is used for every captured frame.
        self.detector = PotholeDetector(
            DetectorConfig(
                model_path=Path(settings.MODEL_PATH),
                confidence_threshold=settings.CONFIDENCE_THRESHOLD,
                pothole_class_name=settings.DETECTION_CLASS,
            )
        )

        self.camera_connected = False
        
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
            logger.info("Video processor started with %s source", self.source_type)

    def stop(self):
        with self.frame_lock:
            self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            logger.info("Video processor stopped")

    def set_coordinates(self, latitude: float, longitude: float):
        """
        Updates the processor's vehicle coordinate telemetry state.
        Called by the API telemetry endpoint.
        """
        self.latitude = latitude
        self.longitude = longitude

    def get_latest_frame(self) -> Optional[bytes]:
        """
        Thread-safe retrieval of the latest processed JPEG frame.
        """
        with self.frame_lock:
            return self.latest_frame_bytes

    @property
    def source_type(self) -> str:
        return "network" if "://" in settings.VIDEO_SOURCE else "file"

    def get_camera_status(self) -> dict[str, object]:
        """Return camera state without exposing the configured stream URL or credentials."""
        with self.frame_lock:
            return {"connected": self.camera_connected, "source_type": self.source_type}

    def _set_camera_connected(self, connected: bool) -> None:
        with self.frame_lock:
            self.camera_connected = connected
            if not connected:
                # Do not keep serving a stale frame as though a camera were live.
                self.latest_frame_bytes = None

    def _run_loop(self):
        while True:
            # Check thread termination request
            with self.frame_lock:
                if not self.is_running:
                    break
            
            # Initialise Video capture device / file
            cap = cv2.VideoCapture(settings.VIDEO_SOURCE)
            if not cap.isOpened():
                self._set_camera_connected(False)
                logger.error("CAMERA CONNECTION FAILED: Unable to open configured %s VIDEO_SOURCE", self.source_type)
                time.sleep(5.0)
                continue
            self._set_camera_connected(True)
            logger.info("Camera source connected (%s)", self.source_type)
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            frame_delay = 1.0 / fps
            
            frame_index = 0
            
            while True:
                with self.frame_lock:
                    if not self.is_running:
                        break
                        
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    self._set_camera_connected(False)
                    logger.warning("Camera frame read failed or source ended (%s); reconnecting", self.source_type)
                    break
                    
                detections: List[Detection] = []
                # Frame Sampling logic: run YOLO inference at configurable intervals
                if frame_index % settings.AI_FRAME_INTERVAL == 0:
                    detections = self.detector.detect(frame, frame_index)
                    if len(detections) > 0:
                        self._save_detections(detections)
                
                # Annotate overlay bounding box graphics
                annotated_frame = draw_bounding_boxes(frame, [detection.to_dict() for detection in detections])
                
                # Encode frame back to JPEG bytes array
                ret_enc, encoded_frame = cv2.imencode('.jpg', annotated_frame)
                if ret_enc:
                    with self.frame_lock:
                        self.latest_frame_bytes = encoded_frame.tobytes()
                
                frame_index += 1
                
                # Maintain frame rates mapping CPU execution time
                elapsed = time.time() - start_time
                sleep_time = max(0.001, frame_delay - elapsed)
                time.sleep(sleep_time)
                
            cap.release()

    def _save_detections(self, detections: List[Detection]):
        """
        Helper method running database write aggregation on a separate thread connection.
        """
        db = SessionLocal()
        try:
            for det in detections:
                now_iso = datetime.utcnow().isoformat() + "Z"
                source_tag = "ai_camera"
                obs = schemas.ObservationCreate(
                    source=source_tag,
                    latitude=self.latitude,
                    longitude=self.longitude,
                    confidence=det.confidence,
                    timestamp=now_iso,
                    sensor_evidence=None
                )
                process_observation(db, obs)
        except Exception:
            logger.exception("VideoProcessor database write error")
        finally:
            db.close()

video_processor = VideoProcessor()
