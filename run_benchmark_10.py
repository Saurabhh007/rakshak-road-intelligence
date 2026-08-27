import torch
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

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
    
    images_dir = Path("ai/test_images")
    image_names = [f"sample_road{i if i > 0 else ''}.jpg" for i in range(10)]
    
    d40_conf_list = []
    d40_top_class_count = 0
    d20_top_class_count = 0
    
    d40_ge_10_count = 0
    d40_ge_25_count = 0
    d40_ge_50_count = 0
    
    print("=== TASK 1 & 3: IMAGE INVENTORY & YOLO BENCHMARK ===")
    
    for filename in image_names:
        filepath = images_dir / filename
        if not filepath.exists():
            print(f"File not found: {filename}")
            continue
            
        # Get dimensions
        img = cv2.imread(str(filepath))
        h, w, c = img.shape
        
        # Run inference (raw: conf=0.0)
        results_raw = model(str(filepath), conf=0.0, verbose=False)
        # Run inference (filtered: default YOLO conf=0.25)
        results_filtered = model(str(filepath), verbose=False)
        
        max_conf = {name: None for name in model.names.values()}
        raw_count = 0
        for r in results_raw:
            for box in r.boxes:
                raw_count += 1
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                conf = float(box.conf[0].item())
                if max_conf[class_name] is None or conf > max_conf[class_name]:
                    max_conf[class_name] = conf
                    
        filtered_count = 0
        filtered_dets = []
        for r in results_filtered:
            for box in r.boxes:
                filtered_count += 1
                cls_id = int(box.cls[0].item())
                class_name = model.names[cls_id]
                conf = float(box.conf[0].item())
                filtered_dets.append((class_name, conf))
                
        # Top class and top confidence in raw prediction
        raw_classes = [(name, val) for name, val in max_conf.items() if val is not None]
        raw_classes.sort(key=lambda x: x[1], reverse=True)
        
        top_class = "NONE"
        top_conf = 0.0
        if len(raw_classes) > 0:
            top_class = raw_classes[0][0]
            top_conf = raw_classes[0][1]
            
        print(f"\nIMAGE: {filename}")
        print(f"Dimensions: {w}x{h}")
        for cls in ['D00', 'D10', 'D20', 'D40', 'Repair']:
            val = max_conf.get(cls)
            val_str = f"{val:.6f}" if val is not None else "NONE"
            print(f"  {cls} MAX CONFIDENCE: {val_str}")
        print(f"  TOP CLASS: {top_class}")
        print(f"  TOP CONFIDENCE: {top_conf:.6f}")
        print(f"  NUMBER OF RAW DETECTIONS: {raw_count}")
        print(f"  NUMBER OF CONFIDENT DETECTIONS (conf>=0.25): {filtered_count}")
        
        # Accumulate statistics
        d40_val = max_conf.get('D40', 0.0)
        d40_val = d40_val if d40_val is not None else 0.0
        d40_conf_list.append(d40_val)
        
        if top_class == 'D40':
            d40_top_class_count += 1
        elif top_class == 'D20':
            d20_top_class_count += 1
            
        if d40_val >= 0.10:
            d40_ge_10_count += 1
        if d40_val >= 0.25:
            d40_ge_25_count += 1
        if d40_val >= 0.50:
            d40_ge_50_count += 1
            
    # Calculate statistics
    avg_d40_conf = sum(d40_conf_list) / len(d40_conf_list) if len(d40_conf_list) > 0 else 0.0
    max_d40_conf = max(d40_conf_list) if len(d40_conf_list) > 0 else 0.0
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"TOTAL IMAGES = 10")
    print(f"D40 TOP CLASS = {d40_top_class_count}/10")
    print(f"D40 >= 0.10 = {d40_ge_10_count}/10")
    print(f"D40 >= 0.25 = {d40_ge_25_count}/10")
    print(f"D40 >= 0.50 = {d40_ge_50_count}/10")
    print(f"AVERAGE D40 CONFIDENCE: {avg_d40_conf:.6f}")
    print(f"MAX D40 CONFIDENCE: {max_d40_conf:.6f}")
    print(f"D20 TOP CLASS = {d20_top_class_count}/10")

if __name__ == "__main__":
    main()
