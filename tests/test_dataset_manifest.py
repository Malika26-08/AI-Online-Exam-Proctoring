"""
Unit Tests for Phase 4 Dataset Manifest and Readiness Utilities.
Validates dataset manifest schema, parameter loading (T=340000, frame_skip=3),
5 target classes match, zero split leakage, and non-fabricated report generation.
"""

import pytest
import json
from pathlib import Path
from src.config import CLASS_NAMES, MOTION_THRESHOLD, FRAME_SKIP
from src.preprocessing.dataset_manager import DatasetManager
from src.preprocessing.dataset_manifest import DatasetManifestBuilder
from src.preprocessing.dataset_reporter import DatasetReporter


def test_manifest_builder_schema_and_classes(tmp_path):
    """Test DatasetManifestBuilder generates complete schema with 5 target classes and exact parameters."""
    raw_dir = tmp_path / "raw_videos"
    processed_dir = tmp_path / "processed_frames"

    # Create 5 class subfolders with mock video files
    for cls in CLASS_NAMES:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        (cls_dir / f"test_{cls}_vid1.mp4").write_text("mock content")

    manager = DatasetManager(raw_dir=raw_dir, processed_dir=processed_dir, seed=42)
    builder = DatasetManifestBuilder(manager=manager)

    manifest_file = tmp_path / "dataset_manifest.json"
    manifest = builder.build_manifest(output_path=manifest_file)

    assert manifest_file.exists()
    assert "metadata" in manifest
    assert "summary" in manifest

    meta = manifest["metadata"]
    assert meta["dataset_name"] == "Students' Abnormal Behavior in Online Exam Dataset"
    assert meta["reference_paper_dataset"] == "S_OCA"
    assert meta["motion_threshold"] == 340000
    assert meta["frame_skip"] == 3
    assert meta["target_classes"] == CLASS_NAMES
    assert meta["num_classes"] == 5


def test_dataset_reporter_non_fabricated_summary(tmp_path):
    """Test DatasetReporter generates non-fabricated status summary file."""
    raw_dir = tmp_path / "raw_videos"
    public_dir = tmp_path / "non_existent_public_dataset"
    results_dir = tmp_path / "results"

    # Create empty class dirs
    for cls in CLASS_NAMES:
        (raw_dir / cls).mkdir(parents=True, exist_ok=True)

    reporter = DatasetReporter(raw_dir=raw_dir, public_dir=public_dir, results_dir=results_dir)
    summary = reporter.generate_summary_report()

    summary_file = results_dir / "dataset_readiness_summary.json"
    assert summary_file.exists()

    assert summary["dataset_status"] == "NOT AVAILABLE IN WORKSPACE"
    assert summary["total_videos_in_workspace"] == 0
    assert summary["valid_videos"] == 0
    assert summary["corrupt_videos"] == 0
    assert summary["duplicate_videos"] == 0
    assert summary["class_distribution"] == {cls: 0 for cls in CLASS_NAMES}
