"""
YOLOv5 Training Test Script.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import torch
from ultralytics import YOLO
from src.config import RAW_PUBLIC_DATASET_DIR, WEIGHTS_DIR, YOLOV5_DATA_DIR
from src.models.yolov5_detector import YOLOv5Detector

def main():
    print("=" * 60)
    print("STARTING YOLOV5 FINE-TUNING & EVALUATION")
    print("=" * 60)

    data_yaml_path = YOLOv5Detector.generate_data_yaml(
        output_path=YOLOV5_DATA_DIR / "data.yaml",
        dataset_dir=RAW_PUBLIC_DATASET_DIR
    )
    print(f"Generated data.yaml at {data_yaml_path}")

    # Load YOLOv5s backbone
    model = YOLO("yolov5s.pt")
    print("Loaded YOLOv5s pretrained backbone.")

    # Train for 3 epochs on CPU with imgsz=224, batch=32 for fast CPU execution
    print("Starting CPU training: 3 epochs, imgsz=224, batch=32...")
    results = model.train(
        data=str(data_yaml_path.resolve()),
        epochs=3,
        imgsz=224,
        batch=32,
        workers=2,
        device="cpu",
        project=str(WEIGHTS_DIR / "yolov5_runs"),
        name="yolov5s_behavior",
        exist_ok=True,
        verbose=True
    )

    print("\n--- Training Finished! ---")
    
    # Locate best trained weights
    runs_dir = WEIGHTS_DIR / "yolov5_runs" / "yolov5s_behavior"
    best_pt = runs_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = runs_dir / "weights" / "last.pt"

    target_best_pt = WEIGHTS_DIR / "yolov5_best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, target_best_pt)
        print(f"Successfully copied trained checkpoint to {target_best_pt} (Size: {target_best_pt.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"Warning: Checkpoint file not found at {best_pt}")

    # Validate trained model on valid split
    print("\nRunning Validation Evaluation on valid split...")
    trained_model = YOLO(str(target_best_pt))
    val_results = trained_model.val(data=str(data_yaml_path.resolve()), imgsz=224, batch=32, device="cpu")

    print("\n=======================================================")
    print("GENUINE YOLOV5 VALIDATION METRICS")
    print("=======================================================")
    print(f"  • Precision (P)    : {val_results.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"  • Recall (R)       : {val_results.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print(f"  • mAP@0.5          : {val_results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"  • mAP@0.5:0.95     : {val_results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")

    print("\n--- PER-CLASS RESULTS ---")
    class_names = ['eye_movement', 'hand_move', 'mobile_use', 'side_watching', 'mouth_open']
    if hasattr(val_results, 'maps') and val_results.maps is not None:
        for idx, cls_name in enumerate(class_names):
            if idx < len(val_results.maps):
                print(f"  • {cls_name:<20}: mAP50-95 = {val_results.maps[idx]:.4f}")

if __name__ == "__main__":
    main()
