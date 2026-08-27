import torch
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

def _patch_yolo12_attention():
    from ultralytics.nn.modules.block import AAttn

    def compatibility_forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        all_head_dim = self.num_heads * self.head_dim

        qkv = self.qkv(x)

        qk, v = qkv.split(
            [all_head_dim * 2, all_head_dim],
            dim=1,
        )

        pp = self.pe(v)

        qk = qk.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(
                B * self.area,
                N // self.area,
                all_head_dim * 2,
            )
            v = v.reshape(
                B * self.area,
                N // self.area,
                all_head_dim,
            )
            B, N, _ = qk.shape

        q, k = qk.split(
            [all_head_dim, all_head_dim],
            dim=2,
        )

        q = q.transpose(1, 2).view(
            B, self.num_heads, self.head_dim, N
        )
        k = k.transpose(1, 2).view(
            B, self.num_heads, self.head_dim, N
        )
        v = v.transpose(1, 2).view(
            B, self.num_heads, self.head_dim, N
        )

        attn = (q.transpose(-2, -1) @ k) * (
            self.head_dim ** -0.5
        )

        max_attn = attn.max(
            dim=-1,
            keepdim=True,
        ).values

        exp_attn = torch.exp(attn - max_attn)

        attn = exp_attn / exp_attn.sum(
            dim=-1,
            keepdim=True,
        )

        x = v @ attn.transpose(-2, -1)

        x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(
                B // self.area,
                N * self.area,
                C,
            )
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(
            0, 3, 1, 2
        )

        return self.proj(x + pp)

    AAttn.forward = compatibility_forward

_patch_yolo12_attention()

def main():
    model_path_str = "ai/model/yolo12s_RDD2022_best.pt"
    
    print("==================================================")
    print("CHECK 1: MODEL IDENTITY")
    print("==================================================")
    
    # 1. Load weights using torch.load to inspect raw checkpoint
    try:
        ckpt = torch.load(model_path_str, map_location="cpu")
        print(f"Checkpoint loaded successfully from {model_path_str}")
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        if 'epoch' in ckpt:
            print(f"Checkpoint trained for epochs: {ckpt['epoch']}")
        if 'train_args' in ckpt:
            args = ckpt['train_args']
            print(f"Train args (imgsz): {args.get('imgsz', 'Not specified')}")
            print(f"Train args (model): {args.get('model', 'Not specified')}")
            print(f"Train args (task): {args.get('task', 'Not specified')}")
        if 'date' in ckpt:
            print(f"Checkpoint Date: {ckpt['date']}")
        if 'version' in ckpt:
            print(f"Checkpoint YOLO/Ultralytics Version: {ckpt['version']}")
    except Exception as e:
        print(f"Error loading checkpoint dict: {e}")

    # 2. Load model via Ultralytics API
    model = YOLO(model_path_str)
    print(f"Ultralytics version in current environment: {ultralytics.__version__}")
    print(f"Model task: {model.task}")
    print(f"Model class names: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Check model architecture info
    # model.info() prints to stdout, let's call it
    print("\n--- Model Info ---")
    model.info()
    
    # Input image size
    img_size = model.args.get('imgsz', 640)
    print(f"Model input image size (from args): {img_size}")
    
    print("\n==================================================")
    print("CHECK 2: CLASS MAPPING")
    print("==================================================")
    expected_classes = {0: 'D00', 1: 'D10', 2: 'D20', 3: 'D40', 4: 'Repair'}
    print(f"Expected class mapping: {expected_classes}")
    print(f"Actual model class mapping: {model.names}")
    
    mapping_matches = (model.names == expected_classes)
    print(f"Class mapping matches expected: {mapping_matches}")
    if 3 in model.names:
        print(f"Class index 3 name: '{model.names[3]}' (Should be D40)")
    else:
        print("Class index 3 not found in model!")

    print("\n==================================================")
    print("CHECK 3: INFERENCE SANITY TEST")
    print("==================================================")
    
    # Load production detector classes dynamically
    sys.path.append(str(Path(".").resolve()))
    from ai.detector import PotholeDetector
    from ai.config import DetectorConfig
    
    test_image_path = "ai/test_images/sample_road.jpg"
    print(f"Testing image: {test_image_path}")
    
    # Load image
    cv_img = cv2.imread(test_image_path)
    
    # Path A: Production Detector (forcing model path and confidence threshold = 0.0)
    # We configure pothole_class_name = "D40" (since model has "D40", not "pothole")
    config_a = DetectorConfig(
        model_path=Path(model_path_str),
        confidence_threshold=0.0,
        pothole_class_name="D40"
    )
    detector_a = PotholeDetector(config_a)
    detections_a = detector_a.detect(cv_img)
    
    # Path B: Direct Ultralytics inference
    results_b = model(test_image_path, conf=0.0, verbose=False)[0]
    
    print(f"Production detector is_mock: {detector_a.is_mock}")
    print(f"Number of production detections (conf=0.0, pothole_class_name='D40'): {len(detections_a)}")
    
    # Compare raw outputs
    # Get direct detections for class D40
    direct_d40_dets = []
    for box in results_b.boxes:
        cls_id = int(box.cls[0].item())
        class_name = model.names[cls_id]
        if class_name == "D40":
            conf = float(box.conf[0].item())
            bbox = [float(x) for x in box.xyxy[0].tolist()]
            direct_d40_dets.append((conf, bbox))
            
    print(f"Number of direct D40 detections (conf=0.0): {len(direct_d40_dets)}")
    
    # Compare first few detections
    if len(detections_a) > 0 and len(direct_d40_dets) > 0:
        # Sort both by confidence descending
        detections_a.sort(key=lambda x: x.confidence, reverse=True)
        direct_d40_dets.sort(key=lambda x: x[0], reverse=True)
        
        print("\nProduction path top D40 detection:")
        print(f"  Confidence: {detections_a[0].confidence:.6f}")
        print(f"  BBox: {[round(x, 2) for x in detections_a[0].bbox]}")
        
        print("\nDirect path top D40 detection:")
        print(f"  Confidence: {direct_d40_dets[0][0]:.6f}")
        print(f"  BBox: {[round(x, 2) for x in direct_d40_dets[0][1]]}")
        
        conf_diff = abs(detections_a[0].confidence - direct_d40_dets[0][0])
        print(f"Confidence difference: {conf_diff:.8f}")
    else:
        print("No D40 detections to compare.")

    print("\n==================================================")
    print("CHECK 4: PREPROCESSING")
    print("==================================================")
    print(f"Original image dimensions: {cv_img.shape[1]}x{cv_img.shape[0]} (WxH)")
    print(f"Model training/inference image size: {img_size}")
    
    # Check YOLO inference default arguments
    print(f"YOLO conf threshold (default args): {model.predictor.args.get('conf') if hasattr(model, 'predictor') else 'N/A'}")
    print(f"YOLO iou/NMS threshold (default args): {model.predictor.args.get('iou') if hasattr(model, 'predictor') else 'N/A'}")
    print(f"YOLO augment (default args): {model.predictor.args.get('augment') if hasattr(model, 'predictor') else 'N/A'}")
    
    print("\n==================================================")
    print("CHECK 5: KNOWN-GOOD TEST")
    print("==================================================")
    # Check if Czechoslovakia, Japan, or India dataset images are present
    rdd_files = [p for p in Path("ai/test_images").glob("*.jpg") if "India" in p.name or "Japan" in p.name or "Czech" in p.name]
    if len(rdd_files) > 0:
        print(f"Found RDD2022 test image locally: {rdd_files[0].name}")
    else:
        print("KNOWN-GOOD RDD2022 TEST IMAGE: NOT AVAILABLE")

if __name__ == "__main__":
    main()
