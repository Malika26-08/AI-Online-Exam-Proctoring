"""
Model Evaluator for Online Exam Abnormal Activity Classification.
Evaluates CNN models exclusively on the labeled validation dataset split.
Calculates Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and exports reports & figures.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    BENCHMARK_MODELS, CLASS_NAMES, NUM_CLASSES, DEFAULT_IMAGE_SIZE,
    INCEPTION_IMAGE_SIZE, WEIGHTS_DIR, RESULTS_DIR
)
from src.models.model_factory import build_model
from src.preprocessing.dataset_loader import ExamImageDataset
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """
    Evaluates a trained benchmark model on a specified dataset split (validation set).
    Computes true model predictions, calculates metrics, saves JSON reports, and plots confusion matrices.
    """

    def __init__(
        self,
        model_name: str,
        weights_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        weights_dir: Path = WEIGHTS_DIR,
        results_dir: Path = RESULTS_DIR
    ):
        self.model_name = model_name.lower().strip()
        if self.model_name not in BENCHMARK_MODELS or self.model_name == "yolov5":
            raise ValueError(f"ModelEvaluator supports benchmark CNN classifiers: {[m for m in BENCHMARK_MODELS if m != 'yolov5']}")

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = Path(weights_dir)
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        if weights_path is None:
            self.weights_path = self.weights_dir / f"{self.model_name}_best.pt"
        else:
            self.weights_path = Path(weights_path)

        if "inception" in self.model_name:
            self.target_size = INCEPTION_IMAGE_SIZE
        else:
            self.target_size = DEFAULT_IMAGE_SIZE

        # Build PyTorch model
        self.model = build_model(
            model_name=self.model_name,
            pretrained=False,
            num_classes=NUM_CLASSES,
            device=self.device
        )

        self._load_weights()
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"Initialized ModelEvaluator for '{self.model_name}' on device='{self.device}'.")

    def _load_weights(self):
        """Loads model weights from saved PyTorch checkpoint if available."""
        if not self.weights_path.exists():
            logger.warning(f"Weights file not found at {self.weights_path}. Evaluating uninitialized model.")
            return

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and not any(k.startswith("model") for k in checkpoint.keys()):
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"Successfully loaded weights for '{self.model_name}' from {self.weights_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint from {self.weights_path}: {e}")
            raise RuntimeError(f"Failed to load weights for {self.model_name}: {e}")

    def evaluate(
        self,
        val_loader: DataLoader,
        split_name: str = "valid"
    ) -> Dict[str, Any]:
        """
        Executes model inference over the validation DataLoader and calculates classification metrics.

        Returns:
            Dict containing detailed evaluation metrics, confusion matrix, and classification report.
        """
        self.model.eval()
        all_targets: List[int] = []
        all_predictions: List[int] = []
        all_probabilities: List[List[float]] = []
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_targets.extend(targets.cpu().numpy().tolist())
                all_predictions.extend(preds.cpu().numpy().tolist())
                all_probabilities.extend(probs.cpu().numpy().tolist())
                total_samples += targets.size(0)

        val_loss = running_loss / total_samples if total_samples > 0 else 0.0

        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)

        # Calculate classification metrics directly from model predictions
        accuracy = float(accuracy_score(y_true, y_pred)) if total_samples > 0 else 0.0
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
        cls_report = classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            "model_name": self.model_name,
            "eval_split": split_name,
            "data_partition_note": "Validation split evaluation only (labeled ground truth)",
            "total_samples": total_samples,
            "val_loss": float(val_loss),
            "accuracy": accuracy,
            "macro_precision": macro_prec,
            "weighted_precision": weighted_prec,
            "macro_recall": macro_rec,
            "weighted_recall": weighted_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "class_names": CLASS_NAMES,
            "confusion_matrix": cm,
            "classification_report": cls_report
        }

        # Save evaluation JSON
        json_path = self.results_dir / f"eval_metrics_{self.model_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics for '{self.model_name}' to {json_path}")

        # Generate and save confusion matrix figure
        self.plot_confusion_matrix(cm, save_name=f"confusion_matrix_{self.model_name}.png")

        return metrics

    def plot_confusion_matrix(
        self,
        cm: Union[List[List[int]], np.ndarray],
        save_name: str = "confusion_matrix.png"
    ) -> Path:
        """Generates and saves a publication-quality confusion matrix heatmap."""
        cm_array = np.array(cm)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar=True
        )
        plt.title(f"Validation Confusion Matrix: {self.model_name.upper()}", fontsize=14, pad=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        fig_path = self.figures_dir / save_name
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix figure for '{self.model_name}' to {fig_path}")
        return fig_path
