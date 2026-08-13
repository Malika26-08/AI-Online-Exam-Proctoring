"""
Unit Tests for Dataset Manager and Leakage Verification.
Verifies video-level train/val/test splitting and zero leakage checks.
"""

import pytest
from pathlib import Path
from src.preprocessing.dataset_manager import DatasetManager
from src.config import CLASS_NAMES


def test_video_level_splitting_no_leakage(tmp_path):
    """Test video-level splitting guarantees zero leakage across train, val, and test splits."""
    raw_dir = tmp_path / "raw_videos"
    processed_dir = tmp_path / "processed_frames"

    # Create dummy raw video files across 5 classes
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(10):  # 10 mock videos per class
            (cls_dir / f"{cls}_vid_{i:02d}.mp4").write_text("dummy_content")

    manager = DatasetManager(raw_dir=raw_dir, processed_dir=processed_dir, seed=42)
    splits = manager.generate_video_level_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_files = set()
    val_files = set()
    test_files = set()

    for cls in CLASS_NAMES:
        for p in splits["train"][cls]:
            train_files.add(p.name)
        for p in splits["val"][cls]:
            val_files.add(p.name)
        for p in splits["test"][cls]:
            test_files.add(p.name)

    # Verify zero intersection (zero video leakage across splits)
    assert len(train_files.intersection(val_files)) == 0
    assert len(train_files.intersection(test_files)) == 0
    assert len(val_files.intersection(test_files)) == 0

    # Total unique files should equal 5 classes * 10 videos = 50 videos
    assert len(train_files) + len(val_files) + len(test_files) == 50


def test_file_hash_duplicate_detection(tmp_path):
    """Test SHA-256 duplicate file detection."""
    test_file1 = tmp_path / "video1.mp4"
    test_file2 = tmp_path / "video2.mp4"

    content = b"sample video stream content 12345"
    test_file1.write_bytes(content)
    test_file2.write_bytes(content)

    hash1 = DatasetManager.compute_file_hash(test_file1)
    hash2 = DatasetManager.compute_file_hash(test_file2)

    assert hash1 == hash2


def test_validate_public_dataset_structure(tmp_path):
    """Test validate_public_dataset method on structured public dataset folders."""
    dataset_dir = tmp_path / "public_dataset"
    train_imgs = dataset_dir / "train" / "images"
    train_lbls = dataset_dir / "train" / "labels"
    train_imgs.mkdir(parents=True, exist_ok=True)
    train_lbls.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "classes.txt").write_text("\n".join(CLASS_NAMES))
    (train_imgs / "img1.jpg").write_bytes(b"dummy_img_data")
    (train_lbls / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n")

    manager = DatasetManager()
    val_info = manager.validate_public_dataset(dataset_dir)

    assert val_info["exists"] is True
    assert val_info["total_images"] == 1
    assert val_info["total_annotations"] == 2
    assert val_info["detected_classes"] == CLASS_NAMES
    assert val_info["splits"]["train"]["class_counts"]["eye_movement"] == 1
    assert val_info["splits"]["train"]["class_counts"]["hand_move"] == 1

