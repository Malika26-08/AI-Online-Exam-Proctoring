"""
YOLOv5 Object Detection & Fine-Tuning Branch Wrapper.
Kept as a separate detection module branch per project methodology.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import yaml
from src.config import CLASS_NAMES, NUM_CLASSES, YOLOV5_DATA_DIR, RAW_PUBLIC_DATASET_DIR, WEIGHTS_DIR
from src.utils.logger import get_logger

logger = get_logger("yolov5_detector")


class YOLOv5Detector:
    """
    Dedicated YOLOv5 Object Detection & Abnormal Activity Localizer.
    Maintains custom data.yaml generation and Ultralytics YOLOv5 interface.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        conf_threshold: float = 0.25
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.model_name = "yolov5"
        self.weights_path = Path(weights_path) if weights_path else None
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Initializes Ultralytics YOLOv5 architecture backbone."""
        try:
            from ultralytics import YOLO
            best_pt = WEIGHTS_DIR / "yolov5_best.pt"
            target_pt = self.weights_path if (self.weights_path and self.weights_path.exists()) else (best_pt if best_pt.exists() else None)

            if target_pt and target_pt.exists():
                self.model = YOLO(str(target_pt))
                self.weights_path = target_pt
                logger.info(f"Loaded YOLOv5 custom checkpoint from {target_pt}")
            else:
                self.model = YOLO("yolov5s.pt")
                logger.info("Initialized default YOLOv5s architecture backbone.")
        except Exception as e:
            logger.warning(f"Ultralytics YOLOv5 initialization warning: {e}. Running in structural simulation mode.")
            self.model = None

    @staticmethod
    def generate_data_yaml(
        output_path: Path = YOLOV5_DATA_DIR / "data.yaml",
        dataset_dir: Path = RAW_PUBLIC_DATASET_DIR
    ) -> Path:
        """
        Generates custom data.yaml required for training YOLOv5 on the 5 target classes.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data_config = {
            "path": str(dataset_dir.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": NUM_CLASSES,
            "names": {idx: name for idx, name in enumerate(CLASS_NAMES)}
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        logger.info(f"Generated custom YOLOv5 dataset config at {output_path}")
        return output_path

    def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a key-frame tensor/image matrix.
        Returns list of detections: [{"class_id": int, "class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}]
        """
        if self.model is None or frame is None:
            return []

        # Convert OpenCV BGR array to RGB for Ultralytics YOLO model
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
            import cv2
            input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            input_frame = frame

        results = self.model(input_frame, verbose=False, conf=self.conf_threshold)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": xyxy
                })

        return detections

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Alias for predict_frame."""
        return self.predict_frame(frame)

    def get_summary(self) -> Dict[str, Any]:
        """Returns metadata summary for YOLOv5 model branch."""
        return {
            "model_name": "yolov5",
            "branch_type": "object_detection_and_localization",
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "weights_path": str(self.weights_path) if self.weights_path else "yolov5s.pt (default)"
        }
