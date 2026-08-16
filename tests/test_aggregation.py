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
    """Test merging overlapping detections across multiple CNN models requiring at least 2 models to agree."""
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
        ],
        "InceptionResNetV2": [
            FlaggedSegmentAlert(
                start_time_sec=0.0, end_time_sec=33.2, duration_sec=33.2,
                predicted_class="hand_move", peak_confidence=0.920,
                average_confidence=0.810, key_frame_count=167
            )
        ]
    }

    consensus = merge_multimodel_alerts(model_alerts, gap_tolerance_sec=3.0)

    # 2 distinct classes reach 2-of-4 consensus: side_watching (2 models) and hand_move (2 models)
    assert len(consensus) == 2

    side_watch_alert = next(a for a in consensus if a.predicted_class == "side_watching")
    assert side_watch_alert.num_agreeing_models == 2
    assert "Custom CNN" in side_watch_alert.agreeing_models
    assert "DenseNet121" in side_watch_alert.agreeing_models

    hand_move_alert = next(a for a in consensus if a.predicted_class == "hand_move")
    assert hand_move_alert.num_agreeing_models == 2
    assert "InceptionV3" in hand_move_alert.agreeing_models
    assert "InceptionResNetV2" in hand_move_alert.agreeing_models

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
    assert report["timeline"][0]["num_agreeing_models"] >= 2


def test_normal_intervals_break_alert_and_separated_events(tmp_path):
    """Verify normal intervals break active alerts and separated abnormal episodes stay distinct."""
    aggregator = SlidingWindowAggregator(window_seconds=3.0, confidence_threshold=0.65)

    # Episode 1: 0.0s to 1.5s side_watching (high confidence)
    for i in range(6):
        aggregator.add_prediction(FramePrediction(
            frame_idx=i * 3, timestamp_sec=i * 0.3,
            predicted_class="side_watching", confidence=0.85
        ))

    # Normal gap: 1.8s to 4.8s (normal/low confidence predictions)
    for i in range(6, 17):
        aggregator.add_prediction(FramePrediction(
            frame_idx=i * 3, timestamp_sec=i * 0.3,
            predicted_class="eye_movement", confidence=0.40
        ))

    # Episode 2: 5.1s to 8.7s side_watching (high confidence)
    for i in range(17, 30):
        aggregator.add_prediction(FramePrediction(
            frame_idx=i * 3, timestamp_sec=i * 0.3,
            predicted_class="side_watching", confidence=0.88
        ))

    merged = aggregator.get_merged_alerts(max_gap_sec=1.0)
    assert len(merged) == 2
    assert merged[0].predicted_class == "side_watching"
    assert merged[0].start_time_sec == 0.0
    assert merged[1].predicted_class == "side_watching"
    assert merged[1].start_time_sec >= 4.0

    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="separated_events_test.mp4",
        total_duration_sec=9.0,
        total_frames=270,
        key_frames_analyzed=30,
        model_name="densenet121",
        alerts=merged
    )

    assert report["summary_statistics"]["total_flagged_segments"] == 2
    assert report["summary_statistics"]["class_wise_counts"]["side_watching"] == 2


def test_requirement_9_keyframe_and_timeline_verifications(tmp_path):
    """
    Specifically tests Requirement #9 rules:
    - 499 frames with frame_skip=3 gives approx 167 key-frames
    - 4 models do NOT turn 167 key-frames into 668 key-frames
    - sliding-window observations do NOT increase key-frame count
    - separated abnormal periods remain separate
    - normal periods break alerts
    - timestamps are preserved
    - overlapping model detections merge correctly
    - class distribution equals final consensus timeline
    """
    # 1. 499 frames with frame_skip=3 gives ~167 key-frames
    frame_indices = list(range(0, 499, 3))
    key_frame_count = len(frame_indices)
    assert key_frame_count == 167

    # 2. Process 167 key-frames through 4 models
    CNN_MODELS = ["Custom CNN", "DenseNet121", "InceptionV3", "InceptionResNetV2"]
    model_predictions = {}
    aggregators = {}

    for m in CNN_MODELS:
        aggregators[m] = SlidingWindowAggregator(confidence_threshold=0.65)
        model_predictions[m] = []

        for f_idx in frame_indices:
            ts = round(f_idx * 0.0667, 2)  # ~15 FPS
            # Episode A: 0 to 10s (f_idx 0 to 150): side_watching conf 0.80
            # Normal Gap: 10s to 20s (f_idx 153 to 300): normal conf 0.90
            # Episode B: 20s to 30s (f_idx 303 to 450): side_watching conf 0.85
            if ts <= 10.0:
                cls = "side_watching"
                conf = 0.80
            elif ts <= 20.0:
                cls = "normal"
                conf = 0.90
            else:
                cls = "side_watching"
                conf = 0.85

            pred = FramePrediction(
                frame_idx=f_idx,
                timestamp_sec=ts,
                predicted_class=cls,
                confidence=conf
            )
            model_predictions[m].append(pred)
            aggregators[m].add_prediction(pred)

    # Verify per-model prediction counts = 167 (NOT 668)
    for m in CNN_MODELS:
        assert len(model_predictions[m]) == 167

    # The total key-frames analyzed for the video remains 167
    video_key_frames = len(frame_indices)
    assert video_key_frames == 167

    # 3. Separated abnormal periods remain separate; normal periods break alerts
    model_alerts_dict = {}
    for m in CNN_MODELS:
        merged = aggregators[m].get_merged_alerts(max_gap_sec=1.0)
        # Should produce 2 separate alerts due to normal period between 10s and 20s
        assert len(merged) == 2
        assert merged[0].predicted_class == "side_watching"
        assert merged[0].end_time_sec <= 10.5
        assert merged[1].predicted_class == "side_watching"
        assert merged[1].start_time_sec >= 19.5
        # Verify key_frame_count in each alert does NOT sum sliding windows (e.g., 2400)
        assert merged[0].key_frame_count <= 167
        assert merged[1].key_frame_count <= 167
        model_alerts_dict[m] = merged

    # 4. Overlapping model detections merge correctly into consensus
    from src.aggregation.sliding_window import merge_multimodel_alerts
    consensus = merge_multimodel_alerts(model_alerts_dict, all_predictions_map=model_predictions, gap_tolerance_sec=1.0)
    assert len(consensus) == 2
    for c in consensus:
        assert c.num_agreeing_models == 4
        assert c.key_frame_count <= 167

    # 5. Class distribution equals final consensus timeline alert counts
    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="test_video_499.mp4",
        total_duration_sec=33.25,
        total_frames=499,
        key_frames_analyzed=video_key_frames,
        model_name="Multi-CNN Consensus",
        alerts=consensus
    )

    class_counts = report["summary_statistics"]["class_wise_counts"]
    assert class_counts["side_watching"] == len(consensus)  # 2
    assert report["summary_statistics"]["total_flagged_segments"] == 2
    assert report["session_metadata"]["key_frames_analyzed"] == 167


def test_time_aligned_multimodel_consensus_specifications(tmp_path):
    """
    Validates Time-Aligned Multi-Model Consensus logic per Requirement #16 & Refinement:
    - 2-of-4 consensus
    - single isolated consensus key-frame excluded (min 2 consecutive consensus key-frames required)
    - 2 consecutive consensus key-frames produce a valid alert
    - single-model prediction not becoming consensus
    - different classes at same timestamp
    - confidence threshold (<0.65 excluded)
    - temporal separation of events (>1.0s gap breaks event)
    - class-wise counts and duration
    - unique key-frame counting
    - no-consensus clean session
    """
    from src.aggregation.sliding_window import merge_multimodel_alerts, FramePrediction

    models = ["Custom CNN", "DenseNet121", "InceptionV3", "InceptionResNetV2"]
    predictions_map = {m: [] for m in models}

    # Scenario A (t=0.0s & t=0.2s): Custom CNN & DenseNet121 predict 'side_watching' (conf 0.70 & 0.80) -> 2 CONSECUTIVE CONSENSUS FRAMES -> VALID ALERT
    for f_idx, ts in [(0, 0.0), (1, 0.2)]:
        predictions_map["Custom CNN"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="side_watching", confidence=0.70))
        predictions_map["DenseNet121"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="side_watching", confidence=0.80))
        predictions_map["InceptionV3"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="hand_move", confidence=0.90))
        predictions_map["InceptionResNetV2"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="side_watching", confidence=0.40))

    # Scenario B (t=5.0s, f_idx=25): Isolated single consensus frame for 'mouth_open' (Custom CNN & DenseNet121 conf 0.85) -> ISOLATED SINGLE FRAME -> NO ALERT
    predictions_map["Custom CNN"].append(FramePrediction(frame_idx=25, timestamp_sec=5.0, predicted_class="mouth_open", confidence=0.85))
    predictions_map["DenseNet121"].append(FramePrediction(frame_idx=25, timestamp_sec=5.0, predicted_class="mouth_open", confidence=0.85))
    predictions_map["InceptionV3"].append(FramePrediction(frame_idx=25, timestamp_sec=5.0, predicted_class="normal", confidence=0.90))
    predictions_map["InceptionResNetV2"].append(FramePrediction(frame_idx=25, timestamp_sec=5.0, predicted_class="normal", confidence=0.90))

    # Scenario C (t=10.0s f_idx=50, t=10.2s f_idx=51): Custom CNN & InceptionV3 predict 'hand_move' (conf 0.85 & 0.88) -> 2 CONSECUTIVE CONSENSUS FRAMES -> VALID ALERT
    for f_idx, ts in [(50, 10.0), (51, 10.2)]:
        predictions_map["Custom CNN"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="hand_move", confidence=0.85))
        predictions_map["DenseNet121"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="normal", confidence=0.90))
        predictions_map["InceptionV3"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="hand_move", confidence=0.88))
        predictions_map["InceptionResNetV2"].append(FramePrediction(frame_idx=f_idx, timestamp_sec=ts, predicted_class="normal", confidence=0.90))

    model_alerts_dict = {m: [] for m in models}
    consensus = merge_multimodel_alerts(model_alerts_dict, all_predictions_map=predictions_map, gap_tolerance_sec=1.0, min_consensus_frames=2)

    # Should produce 2 valid consensus events (isolated single frame mouth_open at t=5.0s excluded):
    # 1. side_watching from 0.0s to 0.2s (agreeing: Custom CNN, DenseNet121)
    # 2. hand_move from 10.0s to 10.2s (agreeing: Custom CNN, InceptionV3)
    assert len(consensus) == 2

    c1 = consensus[0]
    assert c1.predicted_class == "side_watching"
    assert c1.start_time_sec == 0.0
    assert c1.end_time_sec == 0.2
    assert c1.key_frame_count == 2
    assert c1.num_agreeing_models == 2
    assert set(c1.agreeing_models) == {"Custom CNN", "DenseNet121"}

    c2 = consensus[1]
    assert c2.predicted_class == "hand_move"
    assert c2.start_time_sec == 10.0
    assert c2.end_time_sec == 10.2
    assert c2.duration_sec == 0.2
    assert c2.num_agreeing_models == 2
    assert c2.key_frame_count == 2

    # Verify isolated single consensus frame (mouth_open at t=5.0s) was EXCLUDED
    assert not any(c.predicted_class == "mouth_open" for c in consensus)

    # Verify report generator computes both alert counts and class-wise duration
    generator = ReportGenerator(output_dir=tmp_path)
    report = generator.generate_session_report(
        video_name="time_aligned_test.mp4",
        total_duration_sec=15.0,
        total_frames=300,
        key_frames_analyzed=100,
        model_name="Multi-CNN Consensus",
        alerts=consensus
    )

    stats = report["summary_statistics"]
    assert stats["class_wise_counts"]["side_watching"] == 1
    assert stats["class_wise_counts"]["hand_move"] == 1
    assert stats["class_wise_counts"]["mouth_open"] == 0
    assert stats["class_wise_duration_sec"]["hand_move"] == 0.2

    # Scenario D: Clean Session (No 2-model consensus)
    clean_preds_map = {
        "Custom CNN": [FramePrediction(frame_idx=0, timestamp_sec=0.0, predicted_class="side_watching", confidence=0.90)],
        "DenseNet121": [FramePrediction(frame_idx=0, timestamp_sec=0.0, predicted_class="hand_move", confidence=0.90)],
        "InceptionV3": [FramePrediction(frame_idx=0, timestamp_sec=0.0, predicted_class="mobile_use", confidence=0.90)],
        "InceptionResNetV2": [FramePrediction(frame_idx=0, timestamp_sec=0.0, predicted_class="normal", confidence=0.90)]
    }
    clean_consensus = merge_multimodel_alerts({}, all_predictions_map=clean_preds_map)
    assert len(clean_consensus) == 0






