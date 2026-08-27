import torch
import os
from pathlib import Path
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
    model = YOLO("ai/model/yolo12s_RDD2022_best.pt")
    
    # Potential images
    candidate_paths = [
        "ai/test_images/sample_road.jpg",
        "data/samples/validate_output.jpg",
        "frontend/src/assets/hero.png"
    ]
    
    existing_images = []
    for path_str in candidate_paths:
        if Path(path_str).exists():
            existing_images.append(path_str)
            
    print(f"Available suitable project images for benchmark: {existing_images}")
    
    d40_confidences = []
    top_class_d40_count = 0
    d40_conf_ge_25_count = 0
    d40_conf_ge_50_count = 0
    
    for img_path in existing_images:
        print(f"\nEvaluating: {img_path}")
        # We run inference with conf=0.0 to capture max confidence for all classes
        results_raw = model(img_path, conf=0.0, verbose=False)
        results_filtered = model(img_path, verbose=False)
        
        max_conf = {name: None for name in model.names.values()}
        
        for r in results_raw:
            boxes = r.boxes
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                class_name = model.names[cls_idx]
                confidence = float(box.conf[0].item())
                if max_conf[class_name] is None or confidence > max_conf[class_name]:
                    max_conf[class_name] = confidence
                    
        # Determine top class and final detection from filtered results
        # If no filtered detections, we default to the highest confidence raw detection or None
        final_detection = "NONE"
        final_confidence = 0.0
        
        # Let's check filtered results
        filtered_boxes = []
        for r in results_filtered:
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                class_name = model.names[cls_idx]
                confidence = float(box.conf[0].item())
                filtered_boxes.append((class_name, confidence))
                
        if len(filtered_boxes) > 0:
            # Sort by confidence descending
            filtered_boxes.sort(key=lambda x: x[1], reverse=True)
            final_detection = filtered_boxes[0][0]
            final_confidence = filtered_boxes[0][1]
            
        print(f"IMAGE: {img_path}")
        for name in ['D00', 'D10', 'D20', 'D40', 'Repair']:
            val = max_conf.get(name)
            val_str = f"{val:.6f}" if val is not None else "NONE"
            print(f"{name} MAX: {val_str}")
        print(f"FINAL DETECTION: {final_detection}")
        print(f"FINAL CONFIDENCE: {final_confidence:.6f}")
        
        d40_max = max_conf.get('D40')
        d40_val = d40_max if d40_max is not None else 0.0
        d40_confidences.append(d40_val)
        
        # Find which class has the highest overall confidence in raw detections
        raw_classes = [(name, val) for name, val in max_conf.items() if val is not None]
        if len(raw_classes) > 0:
            raw_classes.sort(key=lambda x: x[1], reverse=True)
            top_raw_class = raw_classes[0][0]
        else:
            top_raw_class = None
            
        if top_raw_class == 'D40':
            top_class_d40_count += 1
        if d40_val >= 0.25:
            d40_conf_ge_25_count += 1
        if d40_val >= 0.50:
            d40_conf_ge_50_count += 1
            
    # Calculate statistics
    avg_d40_conf = sum(d40_confidences) / len(d40_confidences) if len(d40_confidences) > 0 else 0.0
    max_d40_conf = max(d40_confidences) if len(d40_confidences) > 0 else 0.0
    
    print("\n--- STATISTICS ---")
    print(f"average D40 confidence: {avg_d40_conf:.6f}")
    print(f"maximum D40 confidence: {max_d40_conf:.6f}")
    print(f"number of images where D40 is the top class: {top_class_d40_count}")
    print(f"number of images where D40 confidence >= 0.25: {d40_conf_ge_25_count}")
    print(f"number of images where D40 confidence >= 0.50: {d40_conf_ge_50_count}")
    
    # Interpretation Rules:
    # PASS: D40 is consistently detected with useful confidence.
    # FAIL: D40 is consistently absent or extremely low while another class dominates.
    # INCONCLUSIVE: The available images are insufficient or unsuitable for a reliable benchmark.
    
    # Let's decide benchmark status.
    # Since we have fewer than 5 images (e.g. 3 images), let's see. 
    # If the user specifically said "If fewer than 5 suitable pothole images exist, report exactly how many were available and test those."
    # Wait, can we pass or fail if there are only 3 images? The rule for INCONCLUSIVE says: "The available images are insufficient or unsuitable for a reliable benchmark."
    # If we have less than 5 images, it might be INCONCLUSIVE because "The available images are insufficient or unsuitable".
    # Or if we have 3 suitable images and D40 is consistently absent/low, is it FAIL or INCONCLUSIVE?
    # Let's calculate the status programmatically:
    # If len(existing_images) < 5: we will report INCONCLUSIVE because they are insufficient.
    # Wait! Let's print out the status matching the user's logic:
    # Let's write the status logic to output INCONCLUSIVE if there are fewer than 5 images.
    benchmark_status = "INCONCLUSIVE"
    if len(existing_images) >= 5:
        if d40_conf_ge_25_count >= 4:
            benchmark_status = "PASS"
        elif avg_d40_conf < 0.05:
            benchmark_status = "FAIL"
    else:
        benchmark_status = "INCONCLUSIVE"
        
    print(f"\nRAKSHAK POTHOLE BENCHMARK: {benchmark_status}")

if __name__ == "__main__":
    main()
