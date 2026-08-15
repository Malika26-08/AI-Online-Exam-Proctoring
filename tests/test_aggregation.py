"""
Unit Tests for Phase 5 Temporal Aggregation & Timeline Reporting.
Validates sliding window parameter binding, voting thresholds, normal behavior filtering,
alert interval merging, and JSON/CSV session report export.
"""

import pytest
import json
import csv
from src.config import SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD, CLASS_NAMES
from src.aggregation.sliding_window import (
    SlidingWindowAggregator, FramePrediction, FlaggedSegmentAlert
)
from src.aggregation.report_generator import ReportGenerator


def test_sliding_window_parameter_binding():
    """Verify aggregator defaults to config constants (window=3.0s, threshold=0.65)."""
    aggregator = SlidingWindowAggregator()
    assert aggregator.window_seconds == 3.0
    assert aggregator.confidence_threshold == 0.65
    assert SLIDING_WINDOW_SECONDS == 3.0
    assert CONFIDENCE_THRESHOLD == 0.65


def test_normal_behavior_filtering():
    """Normal behavior key-frame predictions should never raise abnormal alerts."""
    aggregator = SlidingWindowAggregator()

    for i in range(10):
        pred = FramePrediction(
            frame_idx=i * 3,
            timestamp_sec=i * 0.1,
            predicted_class="normal",
            confidence=0.95
        )
        alert = aggregator.add_prediction(pred)
        assert alert is None

    assert len(aggregator.get_merged_alerts()) == 0


def test_low_confidence_filtering():
    """Abnormal predictions below threshold (e.g. 0.40 < 0.65) should not raise alerts."""
    aggregator = SlidingWindowAggregator(confidence_threshold=0.65)

    pred = FramePrediction(
        frame_idx=3,
        timestamp_sec=0.1,
        predicted_class="external_device",
        confidence=0.40
    )
    alert = aggregator.add_prediction(pred)
    assert alert is None


def test_abnormal_activity_alert_generation():
    """High confidence abnormal prediction should raise a FlaggedSegmentAlert."""
    aggregator = SlidingWindowAggregator(confidence_threshold=0.65)

    pred1 = FramePrediction(
        frame_idx=3,
        timestamp_sec=0.1,
        predicted_class="external_device",
        confidence=0.85
    )
    alert = aggregator.add_prediction(pred1)
    assert alert is not None
    assert alert.predicted_class == "external_device"
    assert alert.peak_confidence == 0.85


def test_contiguous_alert_merging():
    """Contiguous alerts of the same class should be merged into a single timeline interval."""
    aggregator = SlidingWindowAggregator(window_seconds=3.0, confidence_threshold=0.65)

    # Sequence of 5 key-frames flagged as 'head_movement'
    for i in range(5):
        pred = FramePrediction(
            frame_idx=i * 3,
            timestamp_sec=i * 0.5,
            predicted_class="head_movement",
            confidence=0.80 + (i * 0.02)
        )
        aggregator.add_prediction(pred)

    merged = aggregator.get_merged_alerts()
    assert len(merged) == 1
    assert merged[0].predicted_class == "head_movement"
    assert merged[0].start_time_sec == 0.0
    assert merged[0].end_time_sec == 2.0
    assert merged[0].duration_sec == 2.0


def test_report_generator_json_and_csv_export(tmp_path):
    """Test ReportGenerator generates valid JSON and CSV reports with correct summary metrics."""
    generator = ReportGenerator(output_dir=tmp_path)

    sample_alert = FlaggedSegmentAlert(
        start_time_sec=10.0,
        end_time_sec=15.0,
        duration_sec=5.0,
        predicted_class="external_device",
        peak_confidence=0.92,
        average_confidence=0.88,
        key_frame_count=5
    )

    report_data = generator.generate_session_report(
        video_name="test_exam.mp4",
        total_duration_sec=60.0,
        total_frames=1800,
        key_frames_analyzed=100,
        model_name="yolov5",
        alerts=[sample_alert],
        json_filename="test_report.json",
        csv_filename="test_report.csv"
    )

    json_file = tmp_path / "test_report.json"
    csv_file = tmp_path / "test_report.csv"

    assert json_file.exists()
    assert csv_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert loaded_json["session_metadata"]["video_name"] == "test_exam.mp4"
    assert loaded_json["summary_statistics"]["total_flagged_segments"] == 1
    assert loaded_json["summary_statistics"]["class_wise_counts"]["external_device"] == 1

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["predicted_class"] == "external_device"
