"""
Dedicated YOLOv5 Object Detection Fine-Tuning & Evaluation Script.
Trains YOLOv5s on the verified 5-class behavior dataset and evaluates metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import yaml
from ultralytics import YOLO
from src.config import RAW_PUBLIC_DATASET_DIR, WEIGHTS_DIR, YOLOV5_DATA_DIR
from src.models.yolov5_detector import YOLOv5Detector
from src.utils.logger import get_logger

logger = get_logger("train_yolov5")


def run_yolov5_training(
    epochs: int = 3,
    imgsz: int = 224,
    batch_size: int = 32,
    device: str = "cpu"
):
    """
    Fine-tunes YOLOv5s on the 5-class exam behavior dataset, saves weights/yolov5_best.pt,
    and returns genuine validation metrics.
    """
    logger.info("=" * 60)
    logger.info("STARTING YOLOV5 FINE-TUNING & EVALUATION")
    logger.info("=" * 60)

    # 1. Generate/verify data.yaml
    data_yaml_path = YOLOv5Detector.generate_data_yaml(
        output_path=YOLOV5_DATA_DIR / "data.yaml",
        dataset_dir=RAW_PUBLIC_DATASET_DIR
    )
    logger.info(f"Using dataset config at {data_yaml_path}")

    # 2. Build model with yolov5s pretrained backbone
    model = YOLO("yolov5s.pt")
    logger.info("Loaded YOLOv5s pretrained architecture backbone.")

    # 3. Fine-tune model
    logger.info(f"Starting training: {epochs} epochs, imgsz={imgsz}, batch={batch_size}, device={device}")
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        device=device,
        project=str(WEIGHTS_DIR / "yolov5_runs"),
        name="yolov5s_behavior",
        exist_ok=True,
        verbose=True
    )

    # 4. Copy best checkpoint to weights/yolov5_best.pt
    runs_dir = WEIGHTS_DIR / "yolov5_runs" / "yolov5s_behavior"
    best_pt = runs_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = runs_dir / "weights" / "last.pt"

    target_best_pt = WEIGHTS_DIR / "yolov5_best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, target_best_pt)
        logger.info(f"Saved trained checkpoint to {target_best_pt} (Size: {target_best_pt.stat().st_size / 1e6:.2f} MB)")
    else:
        logger.warning(f"Checkpoint file not found at {best_pt}")

    # 5. Evaluate trained checkpoint on valid split
    logger.info("Running validation evaluation on valid split...")
    trained_model = YOLO(str(target_best_pt))
    val_results = trained_model.val(data=str(data_yaml_path.resolve()), imgsz=imgsz, batch=batch_size, device=device)

    metrics = {
        "precision": float(val_results.results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(val_results.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "checkpoint_path": str(target_best_pt),
        "checkpoint_size_mb": round(target_best_pt.stat().st_size / 1e6, 2) if target_best_pt.exists() else 0.0
    }

    logger.info("=" * 60)
    logger.info("GENUINE YOLOV5 EVALUATION METRICS")
    logger.info(f"Precision  : {metrics['precision']:.4f}")
    logger.info(f"Recall     : {metrics['recall']:.4f}")
    logger.info(f"mAP@0.5    : {metrics['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    run_yolov5_training()
