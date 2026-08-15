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


def test_complete_video_prediction_accumulation_and_five_class_distribution(tmp_path):
    """Test accumulating frame predictions across an entire video recording and generating 5-class report."""
    aggregator = SlidingWindowAggregator(window_seconds=3.0, confidence_threshold=0.65)

    # Simulate 50 key-frames over a 15-second recording
    for i in range(50):
        timestamp = i * 0.3
        # First 20 frames: side_watching (conf=0.75)
        # Remaining 30 frames: eye_movement (conf=0.50 below threshold)
        if i < 20:
            p_cls = "side_watching"
            conf = 0.75
        else:
            p_cls = "eye_movement"
            conf = 0.50

        aggregator.add_prediction(FramePrediction(
            frame_idx=i * 3,
            timestamp_sec=timestamp,
            predicted_class=p_cls,
            confidence=conf
        ))

    merged_alerts = aggregator.get_merged_alerts()
    assert len(merged_alerts) == 1
    assert merged_alerts[0].predicted_class == "side_watching"

    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="complete_video_session.mp4",
        total_duration_sec=15.0,
        total_frames=450,
        key_frames_analyzed=50,
        model_name="densenet121",
        alerts=merged_alerts
    )

    class_counts = report["summary_statistics"]["class_wise_counts"]
    # Ensure all 5 target classes exist in class_counts
    for cls in CLASS_NAMES:
        assert cls in class_counts

    assert class_counts["side_watching"] == 1
    assert class_counts["eye_movement"] == 0
    assert report["summary_statistics"]["total_flagged_segments"] == 1


def test_clean_session_zero_alerts_handling(tmp_path):
    """Test that a clean video session (all predictions < 65% threshold) produces zero alerts and clean report."""
    aggregator = SlidingWindowAggregator(confidence_threshold=0.65)

    for i in range(30):
        aggregator.add_prediction(FramePrediction(
            frame_idx=i * 3,
            timestamp_sec=i * 0.2,
            predicted_class="eye_movement",
            confidence=0.45
        ))

    merged_alerts = aggregator.get_merged_alerts()
    assert len(merged_alerts) == 0

    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="clean_exam_session.mp4",
        total_duration_sec=6.0,
        total_frames=180,
        key_frames_analyzed=30,
        model_name="densenet121",
        alerts=merged_alerts
    )

    assert report["summary_statistics"]["total_flagged_segments"] == 0
    assert report["timeline"] == []
    for cls in CLASS_NAMES:
        assert report["summary_statistics"]["class_wise_counts"][cls] == 0


def test_multimodel_consensus_and_overlapping_merging(tmp_path):
    """Test merging overlapping detections across multiple CNN models into single consensus events."""
    from src.aggregation.sliding_window import FlaggedSegmentAlert, merge_multimodel_alerts

    model_alerts = {
        "Custom CNN": [
            FlaggedSegmentAlert(
                start_time_sec=0.0, end_time_sec=33.2, duration_sec=33.2,
                predicted_class="side_watching", peak_confidence=0.803,
                average_confidence=0.786, key_frame_count=167
            )
        ],
        "DenseNet121": [
            FlaggedSegmentAlert(
                start_time_sec=20.8, end_time_sec=27.4, duration_sec=6.6,
                predicted_class="side_watching", peak_confidence=0.744,
                average_confidence=0.682, key_frame_count=34
            )
        ],
        "InceptionV3": [
            FlaggedSegmentAlert(
                start_time_sec=0.0, end_time_sec=33.2, duration_sec=33.2,
                predicted_class="hand_move", peak_confidence=0.985,
                average_confidence=0.848, key_frame_count=167
            )
        ]
    }

    consensus = merge_multimodel_alerts(model_alerts, gap_tolerance_sec=3.0)

    # 2 distinct classes: side_watching and hand_move
    assert len(consensus) == 2

    side_watch_alert = next(a for a in consensus if a.predicted_class == "side_watching")
    assert side_watch_alert.num_agreeing_models == 2
    assert "Custom CNN" in side_watch_alert.agreeing_models
    assert "DenseNet121" in side_watch_alert.agreeing_models
    assert side_watch_alert.start_time_sec == 0.0
    assert side_watch_alert.end_time_sec == 33.2

    hand_move_alert = next(a for a in consensus if a.predicted_class == "hand_move")
    assert hand_move_alert.num_agreeing_models == 1
    assert hand_move_alert.agreeing_models == ["InceptionV3"]

    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="consensus_test_video.mp4",
        total_duration_sec=33.2,
        total_frames=499,
        key_frames_analyzed=167,
        model_name="Multi-CNN Consensus",
        alerts=consensus
    )

    assert report["summary_statistics"]["total_flagged_segments"] == 2
    assert report["timeline"][0]["agreeing_models"] is not None
    assert report["timeline"][0]["num_agreeing_models"] >= 1


