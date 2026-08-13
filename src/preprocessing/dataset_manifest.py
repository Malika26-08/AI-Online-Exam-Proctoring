"""
Dataset Manifest Generator for Online Exam Proctoring Pipeline.
Creates data/dataset_manifest.json with complete video metadata, SHA-256 hashes,
and video-level split assignments for zero data leakage training.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import (
    CLASS_NAMES, DATA_DIR, MOTION_THRESHOLD, FRAME_SKIP,
    RAW_VIDEOS_DIR, RAW_PUBLIC_DATASET_DIR, RESULTS_DIR
)
from src.preprocessing.dataset_manager import DatasetManager
from src.utils.logger import get_logger
from src.utils.video_loader import VideoLoader

logger = get_logger("dataset_manifest")

MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"


class DatasetManifestBuilder:
    """Builds and validates structured JSON manifest for the exam dataset."""

    def __init__(self, manager: Optional[DatasetManager] = None):
        self.manager = manager if manager is not None else DatasetManager()

    def build_manifest(self, output_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
        """
        Scans dataset directory, generates train/val/test split entries,
        and constructs the complete JSON dataset manifest.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_name": "Students' Abnormal Behavior in Online Exam Dataset",
                "reference_paper_dataset": "S_OCA",
                "motion_threshold": MOTION_THRESHOLD,
                "frame_skip": FRAME_SKIP,
                "target_classes": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES)
            },
            "summary": {
                "total_items": 0,
                "split_counts": {"train": 0, "val": 0, "test": 0},
                "class_counts": {cls: 0 for cls in CLASS_NAMES}
            },
            "items": []
        }

        # Check public dataset
        if RAW_PUBLIC_DATASET_DIR.exists():
            val_info = self.manager.validate_public_dataset(RAW_PUBLIC_DATASET_DIR)
            manifest_data["summary"]["total_items"] = val_info["total_images"]
            manifest_data["summary"]["total_annotations"] = val_info["total_annotations"]
            manifest_data["summary"]["duplicates_found"] = val_info["duplicates_found"]
            for split_key in ["train", "valid", "test"]:
                m_key = "val" if split_key == "valid" else split_key
                if split_key in val_info["splits"]:
                    s_data = val_info["splits"][split_key]
                    manifest_data["summary"]["split_counts"][m_key] = s_data["image_count"]
                    for c_name, c_cnt in s_data["class_counts"].items():
                        manifest_data["summary"]["class_counts"][c_name] += c_cnt
        else:
            splits = self.manager.generate_video_level_splits()
            for split_name in ["train", "val", "test"]:
                for cls in CLASS_NAMES:
                    video_paths = splits[split_name][cls]
                    manifest_data["summary"]["split_counts"][split_name] += len(video_paths)
                    manifest_data["summary"]["class_counts"][cls] += len(video_paths)
                    manifest_data["summary"]["total_items"] += len(video_paths)

                for path in video_paths:
                    video_entry = self._extract_video_manifest_entry(path, cls, split_name)
                    manifest_data["videos"].append(video_entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logger.info(f"Successfully generated dataset manifest with {manifest_data['summary']['total_items']} items at {output_path}")
        return manifest_data

    def _extract_video_manifest_entry(self, video_path: Path, class_name: str, split_name: str) -> Dict[str, Any]:
        """Extracts metadata entry for a single video file."""
        file_hash = DatasetManager.compute_file_hash(video_path)

        try:
            loader = VideoLoader(video_path)
            info = loader.metadata
        except Exception as e:
            logger.warning(f"Could not read metadata for {video_path.name}: {e}")
            info = {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0, "duration_sec": 0.0}

        try:
            rel_path = str(video_path.relative_to(DATA_DIR.parent))
        except ValueError:
            rel_path = str(video_path)

        return {
            "video_id": video_path.stem,
            "class_name": class_name,
            "split": split_name,
            "file_name": video_path.name,
            "relative_path": rel_path,
            "sha256_hash": file_hash,
            "duration_sec": info.get("duration_sec", 0.0),
            "total_frames": info.get("frame_count", 0),
            "fps": info.get("fps", 0.0),
            "width": info.get("width", 0),
            "height": info.get("height", 0)
        }


if __name__ == "__main__":
    builder = DatasetManifestBuilder()
    builder.build_manifest()
