"""
Model Trainer for Online Exam Abnormal Activity Classification.
Handles 4 CNN benchmark models: DenseNet121, InceptionV3, Inception-ResNet-v2, and Custom CNN.
Includes reproducible seed initialization, GPU/CPU execution, AMP mixed precision,
checkpoint saving, early stopping, and training history logging.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    BENCHMARK_MODELS, NUM_CLASSES, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE,
    RANDOM_SEED, WEIGHTS_DIR, RESULTS_DIR, DEFAULT_LEARNING_RATE, EARLY_STOPPING_PATIENCE
)
from src.models.model_factory import build_model
from src.utils.logger import get_logger

logger = get_logger("trainer")


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seed across Python, NumPy, PyTorch CPU & CUDA for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Initialized global random seed = {seed}")


class ExamModelTrainer:
    """Manages model build, optimizer setup, training/validation loops, checkpoints, and early stopping."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        device: Optional[str] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = 1e-4,
        weights_dir: Path = WEIGHTS_DIR,
        results_dir: Path = RESULTS_DIR,
        seed: int = RANDOM_SEED
    ):
        self.model_name = model_name.lower().strip()
        if self.model_name == "yolov5":
            raise ValueError("YOLOv5 uses object detection fine-tuning. Use CNN classification model_name.")
        if self.model_name not in BENCHMARK_MODELS:
            raise ValueError(f"Unknown model name '{model_name}'. Allowed: {BENCHMARK_MODELS}")

        self.pretrained = pretrained
        self.seed = seed
        set_seed(self.seed)

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.weights_dir = Path(weights_dir)
        self.results_dir = Path(results_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Select target input size based on architecture requirement
        if "inception" in self.model_name:
            self.target_size = INCEPTION_IMAGE_SIZE
        else:
            self.target_size = DEFAULT_IMAGE_SIZE

        # Build PyTorch model
        self.model = build_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=NUM_CLASSES,
            device=self.device
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))

        logger.info(
            f"Initialized ExamModelTrainer for '{self.model_name}' "
            f"on device='{self.device}', lr={self.learning_rate}, target_size={self.target_size}"
        )

    def train_epoch(self, train_loader: DataLoader, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """Runs a single training epoch with optional batch limit for smoke testing."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                outputs = self.model(inputs)
                # InceptionV3 in training mode may return InceptionOutputs tuple/namedtuple
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data).item()
            total += targets.size(0)

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """Runs a single validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(inputs)
                    if hasattr(outputs, "logits"):
                        outputs = outputs.logits
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == targets.data).item()
                total += targets.size(0)

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        patience: int = EARLY_STOPPING_PATIENCE,
        smoke_test: bool = False,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete training loop with checkpoint saving, early stopping, and history recording.
        If smoke_test=True, limits execution to 1 epoch and 2 batches.
        """
        max_batches = 2 if smoke_test else None
        epochs_to_run = 1 if smoke_test else num_epochs

        best_val_loss = float("inf")
        patience_counter = 0
        start_epoch = 1

        history = {
            "model_name": self.model_name,
            "device": self.device,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "best_val_acc": 0.0,
            "smoke_test": smoke_test
        }

        if smoke_test:
            best_weights_path = self.weights_dir / f"{self.model_name}_smoke_best.pt"
            latest_weights_path = self.weights_dir / f"{self.model_name}_smoke_latest.pt"
            history_path = self.results_dir / f"training_history_{self.model_name}_smoke.json"
        else:
            best_weights_path = self.weights_dir / f"{self.model_name}_best.pt"
            latest_weights_path = self.weights_dir / f"{self.model_name}_latest.pt"
            history_path = self.results_dir / f"training_history_{self.model_name}.json"

        # Resume logic
        if resume and not smoke_test and latest_weights_path.exists():
            try:
                ckpt = torch.load(latest_weights_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    start_epoch = ckpt.get("epoch", 0) + 1
                    best_val_loss = ckpt.get("val_loss", float("inf"))
                    logger.info(f"Resuming training for {self.model_name} from epoch {start_epoch}...")
            except Exception as e:
                logger.warning(f"Could not load resume checkpoint: {e}. Starting fresh.")

        logger.info(f"Starting training loop for {self.model_name} (Epochs {start_epoch} to {epochs_to_run})...")

        for epoch in range(start_epoch, epochs_to_run + 1):
            t_loss, t_acc = self.train_epoch(train_loader, max_batches=max_batches)
            v_loss, v_acc = self.validate_epoch(val_loader, max_batches=max_batches)

            self.scheduler.step(v_loss)

            history["train_loss"].append(t_loss)
            history["train_acc"].append(t_acc)
            history["val_loss"].append(v_loss)
            history["val_acc"].append(v_acc)

            logger.info(
                f"Epoch [{epoch}/{epochs_to_run}] — "
                f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | "
                f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}"
            )

            # Checkpoint saving for best model
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                history["best_epoch"] = epoch
                history["best_val_loss"] = v_loss
                history["best_val_acc"] = v_acc

                checkpoint = {
                    "epoch": epoch,
                    "model_name": self.model_name,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc
                }
                torch.save(checkpoint, best_weights_path)
                logger.info(f"Saved best model checkpoint to {best_weights_path}")
            else:
                patience_counter += 1

            # Save latest checkpoint
            checkpoint_latest = {
                "epoch": epoch,
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": v_loss,
                "val_acc": v_acc
            }
            torch.save(checkpoint_latest, latest_weights_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if patience_counter >= patience and not smoke_test:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

        # Save training history JSON
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved training history to {history_path}")
        return history
