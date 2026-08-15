"""
Unit Tests for Model Training Pipeline & Preprocessing Dataset Loader.
Validates dominant YOLO label assignment rule, PyTorch Dataset tensor output shapes,
data augmentation, reproducible seeding, and training loop smoke tests.
"""

import pytest
import json
from pathlib import Path
import torch
from torchvision import transforms

from src.config import CLASS_NAMES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, RAW_PUBLIC_DATASET_DIR
from src.preprocessing.dataset_loader import extract_dominant_class_label, ExamImageDataset, get_dataloaders
from src.training.trainer import ExamModelTrainer, set_seed
from src.training.train import run_training_pipeline


def test_dominant_class_label_extraction(tmp_path):
    """Test that extract_dominant_class_label selects class_id of bounding box with largest area."""
    lbl_file = tmp_path / "sample_label.txt"
    # Box 1: class 0, area = 0.1 * 0.1 = 0.01
    # Box 2: class 2, area = 0.4 * 0.5 = 0.20 (Dominant)
    # Box 3: class 1, area = 0.2 * 0.2 = 0.04
    content = (
        "0 0.5 0.5 0.1 0.1\n"
        "2 0.3 0.3 0.4 0.5\n"
        "1 0.2 0.2 0.2 0.2\n"
    )
    lbl_file.write_text(content)

    dominant_cls = extract_dominant_class_label(lbl_file)
    assert dominant_cls == 2, f"Expected dominant class ID 2, got {dominant_cls}"


def test_exam_image_dataset_sample_pairing_and_tensor_shapes(tmp_path):
    """Test ExamImageDataset initialization, sample discovery, and tensor shape transformation."""
    if not RAW_PUBLIC_DATASET_DIR.exists():
        pytest.skip("Public dataset directory not found in workspace.")

    dataset = ExamImageDataset(
        dataset_dir=RAW_PUBLIC_DATASET_DIR,
        split="train",
        target_size=DEFAULT_IMAGE_SIZE
    )

    assert len(dataset) > 0, "Dataset should discover valid image samples"
    image_tensor, label = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < 5


def test_inception_target_size_dataset(tmp_path):
    """Test ExamImageDataset with 299x299 Inception target spatial dimensions."""
    if not RAW_PUBLIC_DATASET_DIR.exists():
        pytest.skip("Public dataset directory not found in workspace.")

    dataset = ExamImageDataset(
        dataset_dir=RAW_PUBLIC_DATASET_DIR,
        split="valid",
        target_size=INCEPTION_IMAGE_SIZE
    )

    image_tensor, label = dataset[0]
    assert image_tensor.shape == (3, 299, 299)


def test_reproducible_seeding():
    """Test global seed initialization function."""
    set_seed(42)
    rand1 = torch.randn(5)
    set_seed(42)
    rand2 = torch.randn(5)
    assert torch.equal(rand1, rand2)


def test_trainer_smoke_test_custom_cnn(tmp_path):
    """Test ExamModelTrainer smoke test execution and checkpoint generation on Custom CNN."""
    if not RAW_PUBLIC_DATASET_DIR.exists():
        pytest.skip("Public dataset directory not found in workspace.")

    train_loader, val_loader = get_dataloaders(
        dataset_dir=RAW_PUBLIC_DATASET_DIR,
        target_size=DEFAULT_IMAGE_SIZE,
        batch_size=4,
        max_samples_per_split=8
    )

    weights_dir = tmp_path / "weights"
    results_dir = tmp_path / "results"

    trainer = ExamModelTrainer(
        model_name="custom_cnn",
        pretrained=False,
        weights_dir=weights_dir,
        results_dir=results_dir
    )

    history = trainer.train(train_loader, val_loader, num_epochs=1, smoke_test=True)

    assert history["model_name"] == "custom_cnn"
    assert history["smoke_test"] is True
    assert (weights_dir / "custom_cnn_smoke_best.pt").exists()
    assert (results_dir / "training_history_custom_cnn_smoke.json").exists()


def test_training_pipeline_cli_smoke_test(tmp_path):
    """Test run_training_pipeline entrypoint function in smoke_test mode."""
    if not RAW_PUBLIC_DATASET_DIR.exists():
        pytest.skip("Public dataset directory not found in workspace.")

    results = run_training_pipeline(
        model_name="custom_cnn",
        epochs=1,
        batch_size=4,
        smoke_test=True,
        dataset_dir=RAW_PUBLIC_DATASET_DIR,
        weights_dir=tmp_path / "weights",
        results_dir=tmp_path / "results"
    )

    assert "custom_cnn" in results
    assert results["custom_cnn"]["smoke_test"] is True
