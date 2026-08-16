"""
Unit Tests for Phase 3 Model Architecture Definition and Factory.
Validates model instantiation, input tensor shapes (224x224 vs 299x299), output logit dimensions (B, 5),
parameter counting, model factory selection, and YOLOv5 branch initialization.
"""

import pytest
import torch
from src.config import BENCHMARK_MODELS, NUM_CLASSES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
from src.models.model_factory import build_model, get_all_benchmark_summaries
from src.models.custom_cnn import CustomCNN
from src.models.pretrained_cnns import (
    DenseNet121Classifier,
    InceptionV3Classifier,
    InceptionResNetV2Classifier
)
from src.models.yolov5_detector import YOLOv5Detector


def test_custom_cnn_instantiation_and_forward():
    """Test Custom CNN baseline output shape (Batch, 5) with 224x224 input tensor."""
    model = CustomCNN(input_size=DEFAULT_IMAGE_SIZE, num_classes=5)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)
    assert not torch.isnan(logits).any()


def test_densenet121_instantiation_and_forward():
    """Test DenseNet121 classifier output shape (Batch, 5) with 224x224 input tensor."""
    model = DenseNet121Classifier(pretrained=False, input_size=DEFAULT_IMAGE_SIZE, num_classes=5)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)


def test_inception_v3_instantiation_and_forward():
    """Test InceptionV3 classifier output shape (Batch, 5) with 299x299 input tensor."""
    model = InceptionV3Classifier(pretrained=False, input_size=INCEPTION_IMAGE_SIZE, num_classes=5)
    model.eval()

    dummy_input = torch.randn(2, 3, 299, 299)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)


def test_inception_resnet_v2_instantiation_and_forward():
    """Test Inception-ResNet-v2 classifier output shape (Batch, 5) with 299x299 input tensor."""
    model = InceptionResNetV2Classifier(pretrained=False, input_size=INCEPTION_IMAGE_SIZE, num_classes=5)
    model.eval()

    dummy_input = torch.randn(2, 3, 299, 299)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (2, 5)


def test_model_factory_selection():
    """Test build_model instantiates all 5 benchmark models by name."""
    for model_name in BENCHMARK_MODELS:
        model = build_model(model_name, pretrained=False)
        assert model is not None
        if hasattr(model, "model_name"):
            assert model.model_name == model_name

    # Invalid name should raise ValueError
    with pytest.raises(ValueError):
        build_model("invalid_model_name")


def test_parameter_counts_and_summary():
    """Test parameter calculation utility for base classifier models."""
    model = CustomCNN()
    params = model.get_parameter_counts()
    summary = model.get_summary()

    assert "total_params" in params
    assert "trainable_params" in params
    assert params["total_params"] > 0
    assert summary["num_classes"] == 5
    assert summary["model_name"] == "custom_cnn"


def test_yolov5_branch_and_data_yaml(tmp_path):
    """Test YOLOv5 detector branch initialization and data.yaml generation."""
    detector = YOLOv5Detector()
    summary = detector.get_summary()

    assert summary["model_name"] == "yolov5"
    assert summary["branch_type"] == "object_detection_and_localization"

    yaml_path = tmp_path / "data.yaml"
    generated_path = YOLOv5Detector.generate_data_yaml(yaml_path)

    assert generated_path.exists()
    assert generated_path.is_file()


def test_yolov5_prediction_interface():
    """Test YOLOv5Detector predict_frame with a dummy frame matrix."""
    import numpy as np
    detector = YOLOv5Detector()
    dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    detections = detector.predict_frame(dummy_frame)
    assert isinstance(detections, list)
