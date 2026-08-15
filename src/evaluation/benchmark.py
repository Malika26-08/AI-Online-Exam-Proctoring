"""
Model Benchmarking and History Visualization Utility.
Evaluates all 4 CNN benchmark models on the labeled validation split,
plots training history curves, and generates comparative metrics tables in CSV and JSON formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    BENCHMARK_MODELS, CLASS_NAMES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE, RAW_PUBLIC_DATASET_DIR, RESULTS_DIR
)
from src.models.model_factory import build_model
from src.preprocessing.dataset_loader import ExamImageDataset
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("benchmark")

CNN_BENCHMARK_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]


class ModelBenchmarker:
    """
    Orchestrates validation evaluation and training history visualization across all CNN benchmark models.
    Generates comparison tables (CSV & JSON) and loss/accuracy curve figures.
    """

    def __init__(
        self,
        dataset_dir: Path = RAW_PUBLIC_DATASET_DIR,
        results_dir: Path = RESULTS_DIR,
        device: Optional[str] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def plot_training_history(self, model_name: str) -> Optional[Path]:
        """Plots training loss and accuracy curves from saved JSON history file."""
        history_path = self.results_dir / f"training_history_{model_name}.json"
        if not history_path.exists():
            logger.warning(f"Training history file not found for {model_name} at {history_path}")
            return None

        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])

        epochs = list(range(1, len(train_loss) + 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(epochs, train_loss, 'b-o', label="Train Loss", linewidth=2)
        ax1.plot(epochs, val_loss, 'r--s', label="Val Loss", linewidth=2)
        ax1.set_title(f"{model_name.upper()} — Loss Curve", fontsize=12)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Loss", fontsize=10)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Accuracy plot
        ax2.plot(epochs, train_acc, 'b-o', label="Train Acc", linewidth=2)
        ax2.plot(epochs, val_acc, 'r--s', label="Val Acc", linewidth=2)
        ax2.set_title(f"{model_name.upper()} — Accuracy Curve", fontsize=12)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Accuracy", fontsize=10)
        ax2.legend(loc="lower right")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle(f"Training History: {model_name.upper()} (Validation Split Benchmark)", fontsize=14, y=1.02)
        plt.tight_layout()

        fig_path = self.figures_dir / f"loss_acc_curve_{model_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved training history curves for '{model_name}' to {fig_path}")
        return fig_path

    def plot_combined_history_comparison(self) -> Optional[Path]:
        """Plots combined loss and accuracy curves for all benchmark models."""
        histories = {}
        for model_name in CNN_BENCHMARK_MODELS:
            history_path = self.results_dir / f"training_history_{model_name}.json"
            if history_path.exists():
                with open(history_path, "r", encoding="utf-8") as f:
                    histories[model_name] = json.load(f)

        if not histories:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = {"custom_cnn": "teal", "densenet121": "orange", "inception_v3": "purple", "inception_resnet_v2": "crimson"}

        for model_name, hist in histories.items():
            v_loss = hist.get("val_loss", [])
            v_acc = hist.get("val_acc", [])
            epochs = list(range(1, len(v_loss) + 1))
            c = colors.get(model_name, "blue")

            ax1.plot(epochs, v_loss, label=f"{model_name} (Val Loss)", color=c, linewidth=2, marker="o")
            ax2.plot(epochs, v_acc, label=f"{model_name} (Val Acc)", color=c, linewidth=2, marker="s")

        ax1.set_title("Validation Loss Comparison Across Models", fontsize=12)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Validation Loss", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.set_title("Validation Accuracy Comparison Across Models", fontsize=12)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Validation Accuracy", fontsize=10)
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("CNN Models Training History Benchmark (Validation Set)", fontsize=14, y=1.02)
        plt.tight_layout()

        fig_path = self.figures_dir / "benchmark_loss_acc_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved combined history comparison figure to {fig_path}")
        return fig_path

    def run_benchmark(self, batch_size: int = DEFAULT_BATCH_SIZE) -> pd.DataFrame:
        """
        Executes validation split evaluation across all 4 CNN benchmark models.
        Collects metrics, computes parameter counts, and exports CSV & JSON comparative tables.
        """
        logger.info("Starting CNN benchmark evaluation on validation dataset split...")

        results_list = []
        comparison_dict = {}

        # Plot individual and combined training histories
        for model_name in CNN_BENCHMARK_MODELS:
            self.plot_training_history(model_name)
        self.plot_combined_history_comparison()

        for model_name in CNN_BENCHMARK_MODELS:
            logger.info(f"--- Benchmarking Model: {model_name} ---")

            # Determine target image size
            target_size = INCEPTION_IMAGE_SIZE if "inception" in model_name else DEFAULT_IMAGE_SIZE

            # Build validation DataLoader
            val_dataset = ExamImageDataset(
                dataset_dir=self.dataset_dir,
                split="valid",
                target_size=target_size
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )

            # Get parameter count
            try:
                m = build_model(model_name, pretrained=False)
                param_info = m.get_parameter_counts() if hasattr(m, "get_parameter_counts") else {}
                total_params = param_info.get("total_params", 0)
                trainable_params = param_info.get("trainable_params", 0)
            except Exception as e:
                logger.warning(f"Could not get parameter count for {model_name}: {e}")
                total_params = 0
                trainable_params = 0

            # Evaluate model using ModelEvaluator
            evaluator = ModelEvaluator(
                model_name=model_name,
                device=self.device,
                results_dir=self.results_dir
            )
            metrics = evaluator.evaluate(val_loader, split_name="valid")

            row = {
                "Model": model_name,
                "Evaluation Split": "Validation Split Only",
                "Total Samples": metrics["total_samples"],
                "Val Loss": round(metrics["val_loss"], 4),
                "Accuracy": round(metrics["accuracy"], 4),
                "Macro Precision": round(metrics["macro_precision"], 4),
                "Weighted Precision": round(metrics["weighted_precision"], 4),
                "Macro Recall": round(metrics["macro_recall"], 4),
                "Weighted Recall": round(metrics["weighted_recall"], 4),
                "Macro F1-Score": round(metrics["macro_f1"], 4),
                "Weighted F1-Score": round(metrics["weighted_f1"], 4),
                "Total Parameters": total_params,
                "Trainable Parameters": trainable_params
            }

            results_list.append(row)
            comparison_dict[model_name] = {
                "evaluation_split": "Validation Split Only",
                "metrics": metrics,
                "total_params": total_params,
                "trainable_params": trainable_params
            }

        # Convert to DataFrame
        df = pd.DataFrame(results_list)

        # Export CSV comparison
        csv_path = self.results_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparative benchmark CSV to {csv_path}")

        # Export JSON comparison
        json_path = self.results_dir / "model_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark_partition": "Validation split evaluation only (labeled ground truth)",
                "models": comparison_dict,
                "summary_table": results_list
            }, f, indent=2)
        logger.info(f"Saved comparative benchmark JSON to {json_path}")

        return df
