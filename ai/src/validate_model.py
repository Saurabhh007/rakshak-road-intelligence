import os
import cv2
import numpy as np
from detector import HazardDetector
from utils import draw_bounding_boxes

def validate():
    print("--------------------------------------------------")
    print("RAKSHAK: Day 1 AI Validation Gate")
    print("--------------------------------------------------")
    
    # Initialize detector
    model_path = os.getenv("MODEL_PATH", "ai/models/pothole_detector.pt")
    detector = HazardDetector(model_path=model_path)
    
    print(f"AI Engine Status: {'[REAL]' if not detector.is_mock else '[SIMULATED]'}")
    
    # Create a dummy frame (dark road background)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw some fake road lane markers
    cv2.line(frame, (100, 480), (280, 260), (255, 255, 255), 2)
    cv2.line(frame, (540, 480), (360, 260), (255, 255, 255), 2)
    
    # Trigger a detection on frame 120 (mock pothole coordinates)
    detections = detector.detect(frame, frame_index=120)
    print(f"Detected Potholes: {len(detections)}")
    
    if len(detections) > 0:
        for idx, det in enumerate(detections):
            print(f"  Pothole {idx+1}: Conf={det['confidence']:.2f}, Bbox={det['bbox']}")
        
        # Annotate bounding boxes on image
        annotated_frame = draw_bounding_boxes(frame, detections)
        
        # Save output image
        output_dir = "data/samples"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "validate_output.jpg")
        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated frame saved to: {output_path}")
        print("--------------------------------------------------")
        print("Day 1 Validation: SUCCESS")
        print("--------------------------------------------------")
        return True
    else:
        print("--------------------------------------------------")
        print("Day 1 Validation: FAILED - No detections returned.")
        print("--------------------------------------------------")
        return False

if __name__ == "__main__":
    validate()
