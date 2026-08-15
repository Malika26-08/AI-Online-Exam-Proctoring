"""
Unit Tests for Phase 4 Model Evaluation and Benchmarking Module.
Validates metric computation logic, confusion matrix heatmap generation,
history curve plotting, ModelEvaluator execution, and CSV/JSON report exports.
"""

import json
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import CLASS_NAMES, NUM_CLASSES
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.benchmark import ModelBenchmarker
from src.models.custom_cnn import CustomCNN


def test_metric_calculation_logic():
    """Validates exact calculation of classification metrics using synthetic predictions."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 0])  # Last one misclassified as 0

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    assert acc == 0.9
    assert prec > 0.0
    assert rec > 0.0
    assert f1 > 0.0
    assert cm.shape == (5, 5)


def test_evaluator_initialization_and_plot(tmp_path):
    """Tests ModelEvaluator initialization, confusion matrix plotting, and JSON export structure."""
    results_dir = tmp_path / "results"
    weights_dir = tmp_path / "weights"
    results_dir.mkdir()
    weights_dir.mkdir()

    # Create dummy weights for custom_cnn
    dummy_model = CustomCNN()
    weights_path = weights_dir / "custom_cnn_best.pt"
    torch.save({"model_state_dict": dummy_model.state_dict()}, weights_path)

    evaluator = ModelEvaluator(
        model_name="custom_cnn",
        weights_path=weights_path,
        device="cpu",
        weights_dir=weights_dir,
        results_dir=results_dir
    )

    # Create dummy dataset & dataloader (4 samples, 3x224x224)
    dummy_imgs = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    dummy_loader = DataLoader(TensorDataset(dummy_imgs, dummy_labels), batch_size=2)

    metrics = evaluator.evaluate(dummy_loader, split_name="valid")

    assert metrics["model_name"] == "custom_cnn"
    assert metrics["eval_split"] == "valid"
    assert metrics["total_samples"] == 4
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert len(metrics["confusion_matrix"]) == 5

    # Check generated files
    assert (results_dir / "eval_metrics_custom_cnn.json").exists()
    assert (results_dir / "figures" / "confusion_matrix_custom_cnn.png").exists()


def test_benchmarker_history_plotting(tmp_path):
    """Tests ModelBenchmarker history curve generation from dummy JSON history files."""
    results_dir = tmp_path / "results"
    dataset_dir = tmp_path / "data"
    results_dir.mkdir()
    dataset_dir.mkdir()

    # Create dummy training history JSON
    history = {
        "model_name": "custom_cnn",
        "device": "cpu",
        "train_loss": [1.5, 1.2, 0.9],
        "val_loss": [1.6, 1.3, 1.0],
        "train_acc": [0.3, 0.5, 0.7],
        "val_acc": [0.2, 0.4, 0.6]
    }
    history_path = results_dir / "training_history_custom_cnn.json"
    with open(history_path, "w") as f:
        json.dump(history, f)

    benchmarker = ModelBenchmarker(dataset_dir=dataset_dir, results_dir=results_dir, device="cpu")
    fig_path = benchmarker.plot_training_history("custom_cnn")

    assert fig_path is not None
    assert fig_path.exists()
    assert fig_path.name == "loss_acc_curve_custom_cnn.png"
