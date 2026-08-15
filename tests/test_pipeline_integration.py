"""
End-to-End Pipeline Integration Test.
Connects Video Ingestion -> Preprocessing -> Motion Key-Frame Extraction -> Model Inference -> Sliding Window Aggregation -> Report Export.
Uses synthetic video clips generated with OpenCV VideoWriter to test the entire system without external dependencies.
"""

import pytest
from pathlib import Path
import cv2
import numpy as np
import json
import torch

from src.config import (
    MOTION_THRESHOLD, FRAME_SKIP, SLIDING_WINDOW_SECONDS,
    CONFIDENCE_THRESHOLD, BENCHMARK_MODELS, CLASS_NAMES
)
from src.utils.video_loader import VideoLoader
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model
from src.aggregation.sliding_window import SlidingWindowAggregator, FramePrediction
from src.aggregation.report_generator import ReportGenerator


def create_synthetic_video(output_path: Path, num_frames: int = 30, fps: float = 30.0) -> Path:
    """Creates a temporary synthetic MP4 video clip with alternating static and motion frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (640, 480))

    for i in range(num_frames):
        # Create motion by shifting image intensity every 6 frames
        intensity = 200 if (i // 6) % 2 == 1 else 50
        frame = np.ones((480, 640, 3), dtype=np.uint8) * intensity
        writer.write(frame)

    writer.release()
    return output_path


def test_end_to_end_pipeline_integration(tmp_path):
    """
    Tests full pipeline flow:
    Video -> VideoLoader -> Preprocessor -> KeyFrameExtractor -> Model -> Aggregator -> ReportGenerator
    """
    # 1. Generate Synthetic Video Clip
    video_path = tmp_path / "test_exam_session.mp4"
    create_synthetic_video(video_path, num_frames=30, fps=30.0)
    assert video_path.exists()

    # 2. Ingest Video
    loader = VideoLoader(video_path)
    meta = loader.metadata
    assert meta["frame_count"] == 30
    assert meta["fps"] == 30.0

    # 3. Test for each of the 5 benchmark models
    for model_name in BENCHMARK_MODELS:
        preprocessor = FramePreprocessor(target_size=(224, 224))
        keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)
        model = build_model(model_name, pretrained=False)
        if hasattr(model, "eval"):
            model.eval()

        aggregator = SlidingWindowAggregator(
            window_seconds=SLIDING_WINDOW_SECONDS,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        key_frames_count = 0

        for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
            processed_frame = preprocessor.preprocess(raw_frame)
            is_key_frame, sum_diff, _ = keyframe_extractor.process_frame(
                raw_frame, frame_idx, timestamp_sec
            )

            if is_key_frame:
                key_frames_count += 1

                if model_name == "yolov5" and hasattr(model, "predict_frame"):
                    detections = model.predict_frame(raw_frame)
                    pred_class = "normal"
                    conf = 0.90
                else:
                    tensor_img = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    with torch.no_grad():
                        logits = model(tensor_img)
                        probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()

                    top_idx = int(np.argmax(probabilities))
                    pred_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "normal"
                    conf = float(probabilities[top_idx])

                frame_pred = FramePrediction(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    predicted_class=pred_class,
                    confidence=conf
                )
                aggregator.add_prediction(frame_pred)

        assert key_frames_count > 0, f"Expected key frames extracted for {model_name}"

        # 4. Generate Final Session Summary Report
        merged_alerts = aggregator.get_merged_alerts()
        report_gen = ReportGenerator(output_dir=tmp_path)
        report = report_gen.generate_session_report(
            video_name=video_path.name,
            total_duration_sec=meta["duration_sec"],
            total_frames=meta["frame_count"],
            key_frames_analyzed=key_frames_count,
            model_name=model_name,
            alerts=merged_alerts,
            json_filename=f"{model_name}_report.json",
            csv_filename=f"{model_name}_report.csv"
        )

        assert report["session_metadata"]["model_name"] == model_name
        assert report["session_metadata"]["total_frames"] == 30
        assert (tmp_path / f"{model_name}_report.json").exists()
        assert (tmp_path / f"{model_name}_report.csv").exists()
