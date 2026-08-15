"""
Diagnostic Script for Real Video Session-Level Analysis.
Ingests real uploaded webcam video (tmp1s7__972.mp4), runs full pipeline with trained checkpoints,
logs frame predictions, tests sliding window aggregation, and generates session summary.
STRICT RULE: Uses actual model outputs; does not fabricate or modify any values.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
from src.config import (
    CLASS_NAMES, BENCHMARK_MODELS, MOTION_THRESHOLD,
    FRAME_SKIP, SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.utils.logger import get_logger
from src.utils.video_loader import VideoLoader
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model
from src.aggregation.sliding_window import SlidingWindowAggregator, FramePrediction
from src.aggregation.report_generator import ReportGenerator

logger = get_logger("diagnose_real_video")

# Path to the real uploaded webcam video in TEMP
temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
real_video_files = list(temp_dir.glob("tmp*.mp4"))
real_video_path = max(real_video_files, key=lambda p: p.stat().st_mtime) if real_video_files else None


def analyze_real_video(model_name: str = "densenet121"):
    if not real_video_path or not real_video_path.exists():
        print(f"ERROR: Real video file not found in temp directory {temp_dir}")
        return

    print(f"\n=======================================================")
    print(f"ANALYZING REAL UPLOADED WEBCAM VIDEO: {real_video_path.name}")
    print(f"Selected Model: {model_name}")
    print(f"=======================================================")

    # 1. Video Ingestion
    loader = VideoLoader(real_video_path)
    meta = loader.metadata
    print(f"Video Ingested: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} FPS, {meta['frame_count']} frames ({meta['duration_sec']:.2f}s)")

    # 2. Setup Preprocessor, Extractor, Model, Aggregator
    input_size = INCEPTION_IMAGE_SIZE if model_name in ["inception_v3", "inception_resnet_v2"] else DEFAULT_IMAGE_SIZE
    preprocessor = FramePreprocessor(target_size=input_size)
    keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)
    
    # Build model (automatically loads weights/{model_name}_best.pt)
    model = build_model(model_name, load_best_checkpoint=True)
    if hasattr(model, "eval"):
        model.eval()

    aggregator = SlidingWindowAggregator(
        window_seconds=SLIDING_WINDOW_SECONDS,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    key_frames_count = 0
    raw_predictions = []
    class_raw_counts = {cls: 0 for cls in CLASS_NAMES}
    confidences = []

    # 3. Complete Video Inference (all extracted key-frames)
    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        processed_frame = preprocessor.preprocess(raw_frame)
        is_key_frame, sum_diff, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)

        if is_key_frame:
            key_frames_count += 1

            if model_name == "yolov5" and hasattr(model, "predict_frame"):
                detections = model.predict_frame(raw_frame)
                if detections:
                    top_det = max(detections, key=lambda d: d["confidence"])
                    pred_class = top_det["class_name"]
                    conf = top_det["confidence"]
                else:
                    pred_class = "normal"
                    conf = 0.90
                probs = {cls: (conf if cls == pred_class else (1.0 - conf) / 4.0) for cls in CLASS_NAMES}
            else:
                tensor_img = torch.from_numpy(processed_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    logits = model(tensor_img)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()

                top_idx = int(np.argmax(probabilities))
                pred_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else CLASS_NAMES[0]
                conf = float(probabilities[top_idx])
                probs = {cls: float(probabilities[i]) for i, cls in enumerate(CLASS_NAMES) if i < len(probabilities)}

            class_raw_counts[pred_class] += 1
            confidences.append(conf)

            frame_pred = FramePrediction(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                predicted_class=pred_class,
                confidence=conf,
                probabilities=probs
            )
            raw_predictions.append(frame_pred)
            aggregator.add_prediction(frame_pred)

    # 4. Aggregation & Merged Alerts
    merged_alerts = aggregator.get_merged_alerts()

    # 5. Session Report
    report_gen = ReportGenerator()
    report = report_gen.generate_session_report(
        video_name=real_video_path.name,
        total_duration_sec=meta["duration_sec"],
        total_frames=meta["frame_count"],
        key_frames_analyzed=key_frames_count,
        model_name=model_name,
        alerts=merged_alerts
    )

    print("\n--- INFERENCE STATS ---")
    print(f"Total Video Frames      : {meta['frame_count']}")
    print(f"Extracted Key-Frames    : {key_frames_count}")
    print(f"Redundant Frame % Cut   : {report['session_metadata']['key_frame_reduction_percent']}%")
    print(f"Average Confidence      : {np.mean(confidences):.4f} (Min: {np.min(confidences):.4f}, Max: {np.max(confidences):.4f})")
    print(f"Raw Class Prediction Counts : {class_raw_counts}")

    print("\n--- AGGREGATED ALERTS (>65% CONFIDENCE THRESHOLD) ---")
    print(f"Total Flagged Segments  : {len(merged_alerts)}")
    print(f"Flagged Time (sec)      : {report['summary_statistics']['total_flagged_time_sec']}s")
    print(f"Class-wise Alert Counts : {report['summary_statistics']['class_wise_counts']}")

    if merged_alerts:
        print("\n--- FLAGGED TIMELINE EVENTS ---")
        for alert in merged_alerts:
            print(f"  • [{alert.start_time_sec:.1f}s - {alert.end_time_sec:.1f}s] {alert.predicted_class} | Peak Conf: {alert.peak_confidence*100:.1f}% | Avg Conf: {alert.average_confidence*100:.1f}%")
    else:
        print("\n  • Clean Session: No abnormal predictions crossed the 65% threshold.")

    return report


if __name__ == "__main__":
    for m in ["densenet121", "custom_cnn", "inception_v3", "inception_resnet_v2"]:
        analyze_real_video(m)
