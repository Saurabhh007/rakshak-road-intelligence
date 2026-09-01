"""
RAKSHAK — Phase 10D: Baseline YOLOv12 Training Pipeline
Script: training/scripts/train_baseline.py

This script trains a baseline YOLOv12 model on the clean RAKSHAK MVP dataset.
It provides complete experiment isolation, evaluation metrics, and test inference visualization.
The production model (ai/model/yolo12s_RDD2022_best.pt) is NEVER modified or referenced.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RAKSHAK-Train")

def get_project_root() -> Path:
    """Resolve project root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent

def validate_dataset(data_yaml_path: Path) -> dict:
    """Validate data.yaml and associated images/labels."""
    logger.info(f"Validating dataset config: {data_yaml_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    base_dir = data_yaml_path.parent
    stats = {}

    for split in ["train", "val", "test"]:
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split

        if not img_dir.exists():
            raise FileNotFoundError(f"Missing images directory: {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing labels directory: {lbl_dir}")

        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        lbls = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')])

        potholes = 0
        cracks = 0
        backgrounds = 0
        annotated = 0

        for lbl_name in lbls:
            with open(lbl_dir / lbl_name, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if not lines:
                backgrounds += 1
            else:
                annotated += 1
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        if cls_id == 0:
                            potholes += 1
                        elif cls_id == 1:
                            cracks += 1

        stats[split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "annotated_images": annotated,
            "background_images": backgrounds,
            "potholes": potholes,
            "cracks": cracks
        }
        logger.info(f"Split [{split.upper()}]: {len(imgs)} images, {annotated} annotated, {backgrounds} background, {potholes} potholes, {cracks} cracks")

    return stats

def main():
    root_dir = get_project_root()
    logger.info(f"Project root: {root_dir}")

    # Safety check on production model
    prod_model_path = root_dir / "ai" / "model" / "yolo12s_RDD2022_best.pt"
    if prod_model_path.exists():
        initial_mtime = prod_model_path.stat().st_mtime
        logger.info(f"Production model exists at {prod_model_path} (mtime: {initial_mtime}). Safety lock engaged.")
    else:
        initial_mtime = None

    # Paths
    dataset_dir = root_dir / "training" / "datasets" / "rakshak_mvp"
    data_yaml_path = dataset_dir / "data.yaml"
    experiments_dir = root_dir / "training" / "experiments"
    results_dir = root_dir / "training" / "results" / "baseline_v1"
    predictions_dir = results_dir / "predictions"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Validation
    dataset_stats = validate_dataset(data_yaml_path)

    # 2. Model Initialization
    pretrained_model = "yolo12n.pt"
    logger.info(f"Loading pretrained baseline checkpoint: {pretrained_model}")
    model = YOLO(pretrained_model)

    # 3. Training Execution
    logger.info("Starting baseline YOLOv12 training experiment [baseline_v1]...")
    epochs = 50
    imgsz = 640
    batch_size = 4

    train_results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        project=str(experiments_dir.resolve()),
        name="baseline_v1",
        exist_ok=True,
        seed=42,
        deterministic=True,
        plots=True,
        save=True,
        val=True,
        verbose=True
    )

    exp_dir = experiments_dir / "baseline_v1"
    weights_best = exp_dir / "weights" / "best.pt"
    weights_last = exp_dir / "weights" / "last.pt"

    logger.info(f"Training completed. Best weights: {weights_best}")

    # Copy key training plots and weights to results directory
    for item in exp_dir.glob("*.*"):
        if item.is_file():
            shutil.copy2(item, results_dir / item.name)

    if weights_best.exists():
        shutil.copy2(weights_best, results_dir / "best.pt")
    if weights_last.exists():
        shutil.copy2(weights_last, results_dir / "last.pt")

    # 4. Evaluation on Validation Split
    logger.info("Evaluating best model on Validation split...")
    best_model = YOLO(str(weights_best if weights_best.exists() else weights_last))
    val_metrics = best_model.val(
        data=str(data_yaml_path.resolve()),
        split="val",
        imgsz=imgsz,
        project=str(experiments_dir.resolve()),
        name="val_eval",
        exist_ok=True
    )

    # 5. Evaluation on Test Split
    logger.info("Evaluating best model on Test split...")
    test_metrics = best_model.val(
        data=str(data_yaml_path.resolve()),
        split="test",
        imgsz=imgsz,
        project=str(experiments_dir.resolve()),
        name="test_eval",
        exist_ok=True
    )

    # 6. Test Set Inference & Visualizations
    logger.info(f"Running visual inference on test images into {predictions_dir}...")
    test_img_dir = dataset_dir / "images" / "test"
    test_images = sorted(list(test_img_dir.glob("*.*")))

    prediction_records = []
    for img_path in test_images:
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        results = best_model.predict(
            source=str(img_path),
            conf=0.25,
            save=False,
            imgsz=imgsz,
            verbose=False
        )
        r = results[0]
        # Save plotted image
        out_img_path = predictions_dir / f"pred_{img_path.name}"
        plotted = r.plot()
        import cv2
        cv2.imwrite(str(out_img_path), plotted)

        dets = []
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = r.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            dets.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(v, 2) for v in xyxy]
            })

        prediction_records.append({
            "image": img_path.name,
            "prediction_file": out_img_path.name,
            "detection_count": len(dets),
            "detections": dets
        })

    # 7. Extract & Structure Summary Metrics
    def extract_metrics(m):
        names = m.names
        p = float(m.results_dict.get("metrics/precision(B)", 0.0))
        r = float(m.results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(m.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(m.results_dict.get("metrics/mAP50-95(B)", 0.0))
        
        per_class = {}
        if hasattr(m, "maps") and len(m.maps) > 0:
            for cls_idx, cls_name in names.items():
                if cls_idx < len(m.maps):
                    per_class[cls_name] = {
                        "mAP50_95": round(float(m.maps[cls_idx]), 4),
                        "mAP50": round(float(m.box.ap50[cls_idx]), 4) if hasattr(m.box, 'ap50') and cls_idx < len(m.box.ap50) else None,
                        "precision": round(float(m.box.p[cls_idx]), 4) if hasattr(m.box, 'p') and cls_idx < len(m.box.p) else None,
                        "recall": round(float(m.box.r[cls_idx]), 4) if hasattr(m.box, 'r') and cls_idx < len(m.box.r) else None,
                    }
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "per_class": per_class
        }

    val_summary = extract_metrics(val_metrics)
    test_summary = extract_metrics(test_metrics)

    summary_report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "10D - Baseline YOLOv12 Training",
        "pretrained_checkpoint": pretrained_model,
        "epochs_completed": epochs,
        "image_size": imgsz,
        "batch_size": batch_size,
        "dataset_statistics": dataset_stats,
        "validation_metrics": val_summary,
        "test_metrics": test_summary,
        "artifacts": {
            "best_model": str(weights_best.resolve()),
            "last_model": str(weights_last.resolve()),
            "results_dir": str(results_dir.resolve()),
            "predictions_dir": str(predictions_dir.resolve()),
            "total_test_images_inferred": len(prediction_records)
        },
        "sample_predictions": prediction_records[:10]
    }

    report_path = results_dir / "baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"Saved evaluation report to {report_path}")

    # Production safety verification
    if prod_model_path.exists():
        final_mtime = prod_model_path.stat().st_mtime
        if initial_mtime != final_mtime:
            logger.error("CRITICAL ERROR: Production model timestamp changed!")
            raise RuntimeError("Production model was modified!")
        else:
            logger.info("Production safety verified: ai/model/yolo12s_RDD2022_best.pt is UNTOUCHED.")

    logger.info("Baseline training and evaluation pipeline completed successfully!")

if __name__ == "__main__":
    main()
