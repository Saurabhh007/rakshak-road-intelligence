import torch
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

# Apply patch before loading the model
_patch_yolo12_attention()

def main():
    # Load model
    model = YOLO("ai/model/yolo12s_RDD2022_best.pt")
    print("MODEL LOADED")
    print(model.names)
    
    # Run inference with default conf (filtered)
    results_filtered = model("ai/test_images/sample_road4.jpg", verbose=False)
    
    # Run inference with conf=0.0 (raw/every detection)
    results_raw = model("ai/test_images/sample_road4.jpg", conf=0.0, verbose=False)
    print("IMAGE INFERENCE SUCCESSFUL")
    
    # Process raw detections
    raw_detections = []
    max_conf = {name: None for name in model.names.values()}
    
    for r in results_raw:
        boxes = r.boxes
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            class_name = model.names[cls_idx]
            confidence = float(box.conf[0].item())
            bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            
            raw_detections.append({
                "class_id": cls_idx,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })
            
            # Update max confidence
            if max_conf[class_name] is None or confidence > max_conf[class_name]:
                max_conf[class_name] = confidence

    # Process filtered detections
    filtered_detections = []
    for r in results_filtered:
        boxes = r.boxes
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            class_name = model.names[cls_idx]
            confidence = float(box.conf[0].item())
            bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            
            filtered_detections.append({
                "class_id": cls_idx,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox
            })

    # Print every detection details
    print("\n--- EVERY RAW DETECTION ---")
    for idx, det in enumerate(raw_detections, 1):
        print(f"Detection {idx:03d}: ID={det['class_id']}, Name={det['class_name']}, Conf={det['confidence']:.6f}, Box={det['bbox']}")

    print("\n--- SUMMARY OF CLASS MAX CONFIDENCES ---")
    for name in ['D00', 'D10', 'D20', 'D40', 'Repair']:
        val = max_conf.get(name)
        val_str = f"{val}" if val is not None else "NONE"
        print(f"{name}: {val_str}")

    # Check if D40 exists as a candidate
    d40_exists = "YES" if max_conf.get("D40") is not None else "NO"
    d40_max_conf = max_conf.get("D40") if max_conf.get("D40") is not None else 0.0
    d20_max_conf = max_conf.get("D20") if max_conf.get("D20") is not None else 0.0
    
    model_test_status = "PASS" if len(raw_detections) > 0 else "FAIL"

    print("\n--- DIAGNOSTIC REPORT ---")
    print(f"MODEL TEST: {model_test_status}")
    print(f"D40 CANDIDATE: {d40_exists}")
    print(f"D40 MAX CONFIDENCE: {d40_max_conf}")
    print(f"D20 MAX CONFIDENCE: {d20_max_conf}")
    print(f"TOTAL RAW DETECTIONS: {len(raw_detections)}")
    print(f"TOTAL FILTERED DETECTIONS: {len(filtered_detections)}")

if __name__ == "__main__":
    main()
