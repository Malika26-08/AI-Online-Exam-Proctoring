"""
Model Factory Utility for Online Exam Proctoring Pipeline.
Instantiates and configures any of the 5 benchmark models by name identifier.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Union, Tuple, Dict, Any
from src.config import BENCHMARK_MODELS, NUM_CLASSES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
from src.models.base_model import BaseExamClassifier
from src.models.custom_cnn import CustomCNN
from src.models.pretrained_cnns import (
    DenseNet121Classifier,
    InceptionV3Classifier,
    InceptionResNetV2Classifier
)
from src.models.yolov5_detector import YOLOv5Detector
from src.utils.logger import get_logger

logger = get_logger("model_factory")


def build_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
    device: str = "cpu"
) -> Union[BaseExamClassifier, YOLOv5Detector]:
    """
    Factory function to build and initialize any of the 5 benchmark models.

    Supported model names:
      - 'densenet121'           (DenseNet121 ImageNet pretrained + 5-class head, 224x224)
      - 'inception_v3'          (InceptionV3 ImageNet pretrained + 5-class head, 299x299)
      - 'inception_resnet_v2'   (Inception-ResNet-v2 pretrained + 5-class head, 299x299)
      - 'custom_cnn'            (2-Block baseline CNN trained from scratch, 224x224)
      - 'yolov5'                (YOLOv5 object detection & localization branch)
    """
    name_clean = model_name.lower().strip()

    if name_clean not in BENCHMARK_MODELS:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Allowed benchmark models: {BENCHMARK_MODELS}"
        )

    logger.info(f"Building benchmark model '{name_clean}' (Pretrained={pretrained}, Num Classes={num_classes})")

    if name_clean == "densenet121":
        model = DenseNet121Classifier(pretrained=pretrained, input_size=DEFAULT_IMAGE_SIZE, num_classes=num_classes)
    elif name_clean == "inception_v3":
        model = InceptionV3Classifier(pretrained=pretrained, input_size=INCEPTION_IMAGE_SIZE, num_classes=num_classes)
    elif name_clean == "inception_resnet_v2":
        model = InceptionResNetV2Classifier(pretrained=pretrained, input_size=INCEPTION_IMAGE_SIZE, num_classes=num_classes)
    elif name_clean == "custom_cnn":
        # Custom CNN is trained from scratch by definition
        model = CustomCNN(input_size=DEFAULT_IMAGE_SIZE, num_classes=num_classes)
    elif name_clean == "yolov5":
        model = YOLOv5Detector(device=device)
        return model

    if hasattr(model, "to"):
        model.to(device)

    return model


def get_all_benchmark_summaries() -> Dict[str, Dict[str, Any]]:
    """Builds and returns parameter counts and architecture summaries for all 5 models."""
    summaries = {}
    for name in BENCHMARK_MODELS:
        try:
            m = build_model(name, pretrained=False)
            summaries[name] = m.get_summary()
        except Exception as e:
            summaries[name] = {"error": str(e)}
    return summaries
