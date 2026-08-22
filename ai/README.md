# RAKSHAK AI perception module

The module accepts BGR OpenCV frames and returns `Detection` objects containing
`class_name`, `confidence`, and a pixel bounding box (`[x1, y1, x2, y2]`).

## Model weights

No trained pothole model is included or claimed by this repository. Place
validated pothole-capable Ultralytics YOLO weights outside source control (for
example in `ai/model/`) and configure their location at runtime:

```powershell
$env:RAKSHAK_MODEL_PATH = "D:\models\your-pothole-model.pt"
$env:RAKSHAK_CONFIDENCE_THRESHOLD = "0.60"
```

If the path is absent, missing, or cannot be loaded, `PotholeDetector` uses the
deterministic `MockPotholeDetector`. It is only a development/demo fallback and
does not analyse road damage.

The detector emits the configured primary class only. Set
`RAKSHAK_POTHOLE_CLASS_NAME` if the supplied model uses a different label for
potholes.
