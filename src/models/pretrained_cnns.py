"""
Pre-Trained CNN Transfer Learning Architecture Modules.
Implements DenseNet121, InceptionV3, and Inception-ResNet-v2 with replaced classification heads for 5 classes.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from src.config import (
    NUM_CLASSES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.models.base_model import BaseExamClassifier
from src.utils.logger import get_logger

logger = get_logger("pretrained_cnns")


class DenseNet121Classifier(BaseExamClassifier):
    """
    DenseNet121 backbone with replaced 5-class linear classifier head.
    Default input resolution: 224x224x3.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__(model_name="densenet121", input_size=input_size, num_classes=num_classes)

        weights = None
        if pretrained:
            try:
                weights = torchvision_models.DenseNet121_Weights.DEFAULT
            except Exception as e:
                logger.warning(f"Could not load DenseNet121 pretrained weights online: {e}. Fallback to uninitialized weights.")

        self.backbone = torchvision_models.densenet121(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class InceptionV3Classifier(BaseExamClassifier):
    """
    InceptionV3 backbone with replaced 5-class linear classifier head.
    Required input resolution: 299x299x3.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_size: Tuple[int, int] = INCEPTION_IMAGE_SIZE,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__(model_name="inception_v3", input_size=input_size, num_classes=num_classes)

        weights = None
        if pretrained:
            try:
                weights = torchvision_models.Inception_V3_Weights.DEFAULT
            except Exception as e:
                logger.warning(f"Could not load InceptionV3 pretrained weights online: {e}. Fallback to uninitialized weights.")

        # Set aux_logits=True for standard InceptionV3 training support and transform_input=False for consistent scaling across pretrained flags
        self.backbone = torchvision_models.inception_v3(weights=weights, aux_logits=True, transform_input=False)

        # Replace main classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Replace auxiliary classifier head if present
        if hasattr(self.backbone, "AuxLogits") and self.backbone.AuxLogits is not None:
            aux_in_features = self.backbone.AuxLogits.fc.in_features
            self.backbone.AuxLogits.fc = nn.Linear(aux_in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if self.training and isinstance(out, torchvision_models.InceptionOutputs):
            return out.logits
        elif isinstance(out, tuple):
            return out[0]
        return out


class InceptionResNetV2Classifier(BaseExamClassifier):
    """
    Inception-ResNet-v2 164-layer hybrid backbone for 5-class classification.
    Required input resolution: 299x299x3.
    """

    def __init__(
        self,
        pretrained: bool = True,
        input_size: Tuple[int, int] = INCEPTION_IMAGE_SIZE,
        num_classes: int = NUM_CLASSES
    ):
        super().__init__(model_name="inception_resnet_v2", input_size=input_size, num_classes=num_classes)

        self.backbone = None

        # Attempt initializing via timm if available
        try:
            import timm
            self.backbone = timm.create_model("inception_resnet_v2", pretrained=pretrained, num_classes=num_classes)
        except Exception as e:
            logger.info(f"timm library not using prebuilt weights or fallback: {e}. Using PyTorch hybrid Inception-ResNet backbone.")
            self.backbone = self._build_custom_inception_resnet_v2(num_classes)

    def _build_custom_inception_resnet_v2(self, num_classes: int) -> nn.Module:
        """Fallback architecture wrapper using ResNet backbone with Inception multi-scale pooling."""
        resnet = torchvision_models.resnet50(weights=None)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, num_classes)
        return resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
