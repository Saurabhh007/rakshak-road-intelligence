import cv2
import numpy as np
from typing import List, Dict, Any

def draw_bounding_boxes(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes and class labels with confidence on the image.
    
    Args:
        image: numpy BGR frame (H, W, 3)
        detections: List of dicts:
            [
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox": [xmin, ymin, xmax, ymax]
                }
            ]
            
    Returns:
        np.ndarray: annotated image frame.
    """
    annotated = image.copy()
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
        
        xmin, ymin, xmax, ymax = map(int, bbox)
        
        # Color coding: Green for detections (or red for high alert)
        color = (0, 0, 255)  # Red for pothole hazard
        
        # Draw bounding box rectangle
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label banner
        label = f"{class_name.upper()} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_w, text_h = text_size
        
        # Make background black block for label readability
        cv2.rectangle(annotated, (xmin, ymin - text_h - 4), (xmin + text_w + 4, ymin), (0, 0, 0), -1)
        # Write text in white
        cv2.putText(annotated, label, (xmin + 2, ymin - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated
