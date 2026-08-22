import os
import cv2
import numpy as np
from typing import List, Dict, Any

class HazardDetector:
    def __init__(self, model_path: str = None, model_type: str = "yolo", confidence_threshold: float = 0.60):
        """
        Initializes the road hazard/pothole detector.
        Loads configured weights if path exists and imports ultralytics YOLO.
        Falls back to SIMULATED detector if weights or libraries are missing.
        
        Args:
            model_path: Path to the YOLO weights file.
            model_type: Name/type of the model.
            confidence_threshold: Bounding box probability threshold.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.is_mock = True
        self.model = None

        if model_path and os.path.exists(model_path):
            try:
                # Attempt to load ultralytics library
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.is_mock = False
                print(f"AI ENGINE: [REAL] YOLO loaded successfully from {model_path}")
            except Exception as e:
                print(f"AI ENGINE: [SIMULATED] fallback. Error loading weights: {e}")
        else:
            print(f"AI ENGINE: [SIMULATED] fallback. Weights file not found at: '{model_path}'")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Dict[str, Any]]:
        """
        Detects road hazards (potholes) in a BGR numpy frame.
        
        Args:
            frame: opencv BGR frame (H, W, 3)
            frame_index: Current frame counter index (used for high-fidelity simulation sync)
            
        Returns:
            List[dict]: Detections with class name, confidence, and bounding boxes.
        """
        if not self.is_mock and self.model is not None:
            return self._real_detect(frame)
        else:
            return self._mock_detect(frame, frame_index)

    def _real_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs real YOLO inference on the frame.
        """
        # Run inference in verbose-free mode
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf >= self.confidence_threshold:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                # Check for standard pothole or road damage classes
                # Supports custom trained naming (e.g. 'pothole', 'damage', or road-damage codes like 'D00')
                xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [xmin, ymin, xmax, ymax]
                })
        return detections

    def _mock_detect(self, frame: np.ndarray, frame_index: int) -> List[Dict[str, Any]]:
        """
        Generates simulated pothole detections synchronized with demo_video.mp4 frame counts.
        Simulates the visual approach path by growing the bounding boxes as frame count increases.
        """
        h, w = frame.shape[:2]
        detections = []
        
        # Pothole 1: Sighted between frame 110 and 150
        if 110 <= frame_index <= 150:
            progress = (frame_index - 110) / 40.0
            box_w = int(30 + progress * 150)
            box_h = int(15 + progress * 70)
            cx = int(w * 0.48 + progress * 10)
            cy = int(h * 0.55 + progress * 200)
            
            detections.append({
                "class_name": "pothole",
                "confidence": 0.91 + (progress * 0.04),
                "bbox": [cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2]
            })
            
        # Pothole 2: Sighted between frame 290 and 330
        elif 290 <= frame_index <= 330:
            progress = (frame_index - 290) / 40.0
            box_w = int(40 + progress * 180)
            box_h = int(20 + progress * 90)
            cx = int(w * 0.52 - progress * 50)
            cy = int(h * 0.52 + progress * 220)
            
            detections.append({
                "class_name": "pothole",
                "confidence": 0.88 + (progress * 0.07),
                "bbox": [cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2]
            })

        # Pothole 3: Sighted between frame 490 and 530
        elif 490 <= frame_index <= 530:
            progress = (frame_index - 490) / 40.0
            box_w = int(35 + progress * 160)
            box_h = int(18 + progress * 80)
            cx = int(w * 0.45 + progress * 60)
            cy = int(h * 0.54 + progress * 210)
            
            detections.append({
                "class_name": "pothole",
                "confidence": 0.89 + (progress * 0.06),
                "bbox": [cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2]
            })
            
        return detections
