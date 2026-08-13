"""
Custom 2-Block Baseline CNN Model trained from scratch.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Tuple
import torch
import torch.nn as nn
from src.config import NUM_CLASSES, DEFAULT_IMAGE_SIZE
from src.models.base_model import BaseExamClassifier


class CustomCNN(BaseExamClassifier):
    """
    2-Block Convolutional Baseline CNN.
    Block 1: Conv2D(3->32, k=3, p=1) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
    Block 2: Conv2D(32->64, k=3, p=1) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
    Head   : AdaptiveAvgPool2D(7, 7) -> Flatten -> FC(64*7*7 -> 128) -> ReLU -> Dropout(0.5) -> FC(128 -> 5)
    """

    def __init__(self, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__(model_name="custom_cnn", input_size=input_size, num_classes=num_classes)

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial Pooling & Adaptive Flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Expects tensor (B, 3, H, W).
        Returns raw classification logits (B, 5).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        logits = self.classifier(x)
        return logits
