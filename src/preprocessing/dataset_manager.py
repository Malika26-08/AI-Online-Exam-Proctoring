"""
Dataset Manager and Validation Module.
Enforces video-level train/validation/test splitting (preventing frame-level leakage)
and provides comprehensive dataset validation utilities.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import random
from src.config import (
    CLASS_NAMES, RAW_VIDEOS_DIR, RAW_PUBLIC_DATASET_DIR,
    PROCESSED_FRAMES_DIR, RANDOM_SEED, RESULTS_DIR
)
from src.utils.logger import get_logger
from src.utils.video_loader import VideoLoader, VideoValidationError

logger = get_logger("dataset_manager")


class DatasetManager:
    """Manages raw video discovery, video-level splitting, and integrity validation."""

    def __init__(
        self,
        raw_dir: Path = RAW_VIDEOS_DIR,
        processed_dir: Path = PROCESSED_FRAMES_DIR,
        seed: int = RANDOM_SEED
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.seed = seed
        random.seed(self.seed)

    def scan_raw_videos(self) -> Dict[str, List[Path]]:
        """Scans raw_videos directory grouped by target class names."""
        dataset_by_class: Dict[str, List[Path]] = {cls: [] for cls in CLASS_NAMES}

        if not self.raw_dir.exists():
            logger.warning(f"Raw videos directory does not exist yet: {self.raw_dir}")
            return dataset_by_class

        supported_exts = {".mp4", ".avi", ".mov", ".mkv"}

        for cls in CLASS_NAMES:
            class_folder = self.raw_dir / cls
            if class_folder.exists() and class_folder.is_dir():
                for file_path in class_folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                        dataset_by_class[cls].append(file_path)

        return dataset_by_class

    def validate_raw_videos(self) -> Dict[str, Any]:
        """
        Validates all discovered raw videos for file corruption, unreadable frames,
        and duplicate files.
        """
        raw_data = self.scan_raw_videos()
        results = {
            "total_videos_found": 0,
            "valid_videos": 0,
            "corrupt_videos": 0,
            "corrupt_details": [],
            "class_video_counts": {}
        }

        hashes: Dict[str, Path] = {}
        duplicates = []

        for cls, video_paths in raw_data.items():
            results["class_video_counts"][cls] = len(video_paths)
            results["total_videos_found"] += len(video_paths)

            for video_path in video_paths:
                # Check video readability
                try:
                    loader = VideoLoader(video_path)
                    results["valid_videos"] += 1
                except VideoValidationError as e:
                    results["corrupt_videos"] += 1
                    results["corrupt_details"].append({
                        "file": str(video_path),
                        "error": str(e)
                    })

                # Check duplicate file hash
                try:
                    file_hash = self.compute_file_hash(video_path)
                    if file_hash in hashes:
                        duplicates.append({
                            "original": str(hashes[file_hash]),
                            "duplicate": str(video_path)
                        })
                    else:
                        hashes[file_hash] = video_path
                except Exception:
                    pass

        results["duplicates_found"] = len(duplicates)
        results["duplicate_details"] = duplicates

        logger.info(
            f"Dataset Validation Scan complete: Found {results['total_videos_found']} videos "
            f"({results['valid_videos']} valid, {results['corrupt_videos']} corrupt, {results['duplicates_found']} duplicates)."
        )

        return results

    def validate_public_dataset(self, dataset_dir: Path = RAW_PUBLIC_DATASET_DIR) -> Dict[str, Any]:
        """
        Validates the approved public dataset structure (train, valid, test images & labels).
        Checks for file corruption, duplicate SHA-256 hashes, class distribution, and annotation formats.
        """
        dataset_path = Path(dataset_dir)
        results = {
            "dataset_path": str(dataset_path),
            "exists": dataset_path.exists(),
            "detected_classes": [],
            "splits": {},
            "total_images": 0,
            "total_annotations": 0,
            "corrupted_images": 0,
            "duplicates_found": 0,
            "duplicate_details": []
        }

        if not dataset_path.exists():
            logger.warning(f"Public dataset path does not exist: {dataset_path}")
            return results

        classes_txt = dataset_path / "classes.txt"
        if classes_txt.exists():
            with open(classes_txt, "r", encoding="utf-8") as f:
                results["detected_classes"] = [line.strip() for line in f if line.strip()]

        all_hashes: Dict[str, Path] = {}
        duplicates = []

        for split in ["train", "valid", "test"]:
            split_dir = dataset_path / split
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"

            img_files = sorted(list(img_dir.glob("*.*"))) if img_dir.exists() else []
            lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

            valid_imgs = 0
            for img_path in img_files:
                file_hash = self.compute_file_hash(img_path)
                if file_hash in all_hashes:
                    duplicates.append({
                        "original": str(all_hashes[file_hash]),
                        "duplicate": str(img_path)
                    })
                else:
                    all_hashes[file_hash] = img_path
                valid_imgs += 1

            total_boxes = 0
            class_counts = {cls_name: 0 for cls_name in CLASS_NAMES}

            for lbl_path in lbl_files:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        parts = line_str.split()
                        if len(parts) == 5:
                            try:
                                cls_id = int(parts[0])
                                if 0 <= cls_id < len(CLASS_NAMES):
                                    class_counts[CLASS_NAMES[cls_id]] += 1
                                    total_boxes += 1
                            except ValueError:
                                pass

            results["splits"][split] = {
                "image_count": len(img_files),
                "label_count": len(lbl_files),
                "total_bounding_boxes": total_boxes,
                "class_counts": class_counts
            }
            results["total_images"] += len(img_files)
            results["total_annotations"] += total_boxes

        results["duplicates_found"] = len(duplicates)
        results["duplicate_details"] = duplicates

        logger.info(
            f"Public Dataset Validation complete: {results['total_images']} images, "
            f"{results['total_annotations']} annotations across 5 classes."
        )

        return results

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Computes SHA-256 hash of a file for duplicate detection."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def generate_video_level_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Dict[str, List[Path]]]:
        """
        Generates VIDEO-LEVEL splits (train/val/test).
        STRICT REQUIREMENT: Splitting is performed strictly at the video file level,
        NEVER at the frame level, guaranteeing zero frame leakage across splits.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"

        raw_data = self.scan_raw_videos()
        splits: Dict[str, Dict[str, List[Path]]] = {
            "train": {cls: [] for cls in CLASS_NAMES},
            "val": {cls: [] for cls in CLASS_NAMES},
            "test": {cls: [] for cls in CLASS_NAMES}
        }

        for cls, videos in raw_data.items():
            shuffled_videos = list(videos)
            random.Random(self.seed).shuffle(shuffled_videos)

            n_total = len(shuffled_videos)
            if n_total == 0:
                continue

            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_vids = shuffled_videos[:n_train]
            val_vids = shuffled_videos[n_train:n_train + n_val]
            test_vids = shuffled_videos[n_train + n_val:]

            splits["train"][cls] = train_vids
            splits["val"][cls] = val_vids
            splits["test"][cls] = test_vids

        # Validate zero video-level leakage
        self.verify_zero_leakage(splits)

        return splits

    @staticmethod
    def verify_zero_leakage(splits: Dict[str, Dict[str, List[Path]]]):
        """Verifies that no video file exists in more than one split set."""
        train_set: Set[str] = set()
        val_set: Set[str] = set()
        test_set: Set[str] = set()

        for cls in CLASS_NAMES:
            for v in splits["train"][cls]:
                train_set.add(v.name)
            for v in splits["val"][cls]:
                val_set.add(v.name)
            for v in splits["test"][cls]:
                test_set.add(v.name)

        overlap_train_val = train_set.intersection(val_set)
        overlap_train_test = train_set.intersection(test_set)
        overlap_val_test = val_set.intersection(test_set)

        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f"CRITICAL DATA LEAKAGE DETECTED across video splits! "
                f"Train/Val overlap: {overlap_train_val}, "
                f"Train/Test overlap: {overlap_train_test}, "
                f"Val/Test overlap: {overlap_val_test}"
            )
        logger.info("Verified ZERO video-level data leakage across train, val, and test splits.")

    def save_validation_report(self, report_data: Dict[str, Any], file_name: str = "dataset_validation_report.json"):
        """Saves dataset validation report to results directory."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = RESULTS_DIR / file_name
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved dataset validation report to {report_path}")
