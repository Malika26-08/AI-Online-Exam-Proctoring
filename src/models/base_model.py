"""
Base Model Interface for Online Exam Proctoring Classifiers.
Defines common execution, device handling, and summary parameter counting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from src.config import NUM_CLASSES, DEFAULT_IMAGE_SIZE, CLASS_NAMES


class BaseExamClassifier(nn.Module, ABC):
    """Abstract base class for all CNN classification backbones in the proctoring pipeline."""

    def __init__(self, model_name: str, input_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting tensor shape (B, 3, H, W). Returns logits (B, 5)."""
        pass

    def get_parameter_counts(self) -> Dict[str, int]:
        """Calculates total, trainable, and non-trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params
        }

    def get_summary(self) -> Dict[str, Any]:
        """Returns structural architecture metadata."""
        params = self.get_parameter_counts()
        return {
            "model_name": self.model_name,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "class_names": CLASS_NAMES,
            "total_parameters": f"{params['total_params']:,}",
            "trainable_parameters": f"{params['trainable_params']:,}",
            "non_trainable_parameters": f"{params['non_trainable_params']:,}"
        }
