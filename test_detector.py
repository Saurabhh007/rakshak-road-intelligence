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
    
    # Run inference
    results = model("ai/test_images/sample_road.jpg", verbose=False)
    print("IMAGE INFERENCE SUCCESSFUL")
    
    # Print detections
    num_detections = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_idx = int(box.cls[0].item())
            class_name = model.names[cls_idx]
            confidence = float(box.conf[0].item())
            bbox = [round(float(x), 2) for x in box.xyxy[0].tolist()]
            
            print(f"Class: {class_name}")
            print(f"Confidence: {confidence}")
            print(f"Bounding Box: {bbox}")
            num_detections += 1
            
    print(f"Total detections: {num_detections}")

if __name__ == "__main__":
    main()
