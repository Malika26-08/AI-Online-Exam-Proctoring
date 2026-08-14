"""
Dataset Loader for Online Exam Abnormal Activity Classification.
Loads image files and parses corresponding YOLO bounding box annotations (.txt)
using the dominant bounding-box area rule for image-level label assignment.
Applies Gaussian filtering, histogram equalization, resizing, and data augmentation.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    CLASS_NAMES, NUM_CLASSES, RAW_PUBLIC_DATASET_DIR,
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, DEFAULT_BATCH_SIZE
)
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.utils.logger import get_logger

logger = get_logger("dataset_loader")


def extract_dominant_class_label(lbl_path: Path) -> int:
    """
    Dominant Bounding Box Area Rule for Image-Level Label Assignment:
    Parses YOLO format annotation lines: <class_id> <x_center> <y_center> <width> <height>.
    Calculates area = width * height for each bounding box.
    Returns the class_id corresponding to the bounding box with the largest area.
    If no valid bounding box is present, raises ValueError.
    """
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label file not found: {lbl_path}")

    boxes = []
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                    area = w * h
                    if 0 <= cls_id < NUM_CLASSES:
                        boxes.append((area, cls_id))
                except ValueError:
                    continue

    if not boxes:
        raise ValueError(f"No valid bounding box annotations found in label file: {lbl_path}")

    # Sort by area descending (largest bounding box first)
    boxes.sort(key=lambda x: x[0], reverse=True)
    dominant_class_id = boxes[0][1]
    return dominant_class_id


class ExamImageDataset(Dataset):
    """
    PyTorch Dataset for exam activity images with YOLO annotation labels.
    Combines FramePreprocessor (denoising, contrast equalization, resizing)
    with PyTorch vision transforms and data augmentation.
    """

    def __init__(
        self,
        dataset_dir: Path = RAW_PUBLIC_DATASET_DIR,
        split: str = "train",
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        apply_preprocessing: bool = True,
        transform: Optional[Any] = None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = "valid" if split.lower() in ["val", "valid"] else split.lower()
        self.target_size = target_size
        self.apply_preprocessing = apply_preprocessing

        split_dir = self.dataset_dir / self.split
        self.img_dir = split_dir / "images"
        self.lbl_dir = split_dir / "labels"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        self.preprocessor = FramePreprocessor(target_size=self.target_size)
        self.transform = transform if transform is not None else self._get_default_transform()

        # Discover and pair images with labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

        logger.info(
            f"Initialized ExamImageDataset for split='{self.split}' "
            f"({len(self.samples)} valid samples, target_size={self.target_size})"
        )

    def _get_default_transform(self) -> transforms.Compose:
        """Constructs default ImageNet-normalized torchvision transform pipeline."""
        if self.split == "train":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _load_samples(self):
        """Scans image directory and pairs each image with its dominant YOLO class label."""
        img_files = sorted(list(self.img_dir.glob("*.*")))

        for img_path in img_files:
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                logger.warning(f"Skipping unannotated image: {img_path.name}")
                continue

            try:
                class_id = extract_dominant_class_label(lbl_path)
                self.samples.append((img_path, class_id))
            except Exception as e:
                logger.warning(f"Error loading label for {img_path.name}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_id = self.samples[idx]

        # Read BGR image frame
        frame = cv2.imread(str(img_path))
        if frame is None:
            raise ValueError(f"Failed to read image file: {img_path}")

        # Convert BGR to RGB for standard CNN backbones
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply preprocessing pipeline (Gaussian filter -> Histogram equalization -> Resize)
        if self.apply_preprocessing:
            processed = self.preprocessor.preprocess(frame_rgb, target_size=self.target_size)
        else:
            processed = self.preprocessor.resize(frame_rgb, target_size=self.target_size)

        # Apply torchvision transforms (augmentation / normalization)
        image_tensor = self.transform(processed)

        return image_tensor, class_id

    def get_class_counts(self) -> Dict[str, int]:
        """Returns sample counts per target class in this dataset split."""
        counts = {cls_name: 0 for cls_name in CLASS_NAMES}
        for _, class_id in self.samples:
            if 0 <= class_id < NUM_CLASSES:
                counts[CLASS_NAMES[class_id]] += 1
        return counts


def get_dataloaders(
    dataset_dir: Path = RAW_PUBLIC_DATASET_DIR,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    max_samples_per_split: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to construct PyTorch DataLoaders for train and validation splits.
    Supports max_samples_per_split for fast smoke testing.
    """
    train_dataset = ExamImageDataset(
        dataset_dir=dataset_dir,
        split="train",
        target_size=target_size
    )

    val_dataset = ExamImageDataset(
        dataset_dir=dataset_dir,
        split="valid",
        target_size=target_size
    )

    if max_samples_per_split is not None:
        train_dataset.samples = train_dataset.samples[:max_samples_per_split]
        val_dataset.samples = val_dataset.samples[:max_samples_per_split]
        logger.info(f"Subsampled datasets to {max_samples_per_split} samples for smoke test.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
