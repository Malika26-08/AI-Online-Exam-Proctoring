"""
Diagnostic Script to print required debug information for Requirement #8.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.config import (
    CLASS_NAMES, MOTION_THRESHOLD, FRAME_SKIP,
    SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.utils.video_loader import VideoLoader
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model
from src.aggregation.sliding_window import (
    SlidingWindowAggregator, FramePrediction, merge_multimodel_alerts
)

def run_debug():
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    video_files = list(temp_dir.glob("tmp*.mp4"))
    if not video_files:
        print("No video file found in temp.")
        return
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)
    print(f"=======================================================")
    print(f"DEBUGGING EXISTING PIPELINE FOR VIDEO: {video_path.name}")
    print(f"=======================================================")

    loader = VideoLoader(video_path)
    meta = loader.metadata
    total_video_frames = meta["frame_count"]
    print(f"1. Total video frames: {total_video_frames}")

    CNN_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]
    keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)

    preprocessors = {}
    models = {}
    aggregators = {}
    model_predictions = {m: [] for m in CNN_MODELS}

    for m in CNN_MODELS:
        target_size = INCEPTION_IMAGE_SIZE if "inception" in m else DEFAULT_IMAGE_SIZE
        preprocessors[m] = FramePreprocessor(target_size=target_size)
        models[m] = build_model(m, load_best_checkpoint=True)
        if hasattr(models[m], "eval"):
            models[m].eval()
        aggregators[m] = SlidingWindowAggregator(
            window_seconds=SLIDING_WINDOW_SECONDS,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

    extracted_key_frames = 0
    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        is_key_frame, sum_diff, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
        if is_key_frame:
            extracted_key_frames += 1
            for m in CNN_MODELS:
                proc_frame = preprocessors[m].preprocess(raw_frame)
                tensor_img = torch.from_numpy(proc_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    logits = models[m](tensor_img)
                    probs_tensor = torch.softmax(logits, dim=1).squeeze(0).numpy()

                top_idx = int(np.argmax(probs_tensor))
                pred_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "eye_movement"
                conf = float(probs_tensor[top_idx])
                probs = {cls: float(probs_tensor[i]) for i, cls in enumerate(CLASS_NAMES) if i < len(probs_tensor)}

                pred = FramePrediction(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    predicted_class=pred_class,
                    confidence=conf,
                    probabilities=probs
                )
                model_predictions[m].append(pred)
                aggregators[m].add_prediction(pred)

    print(f"2. Extracted key-frame count: {extracted_key_frames}")
    print("\n3. Number of predictions & timestamps per model:")
    for m in CNN_MODELS:
        preds = model_predictions[m]
        print(f"\n--- Model: {m} ---")
        print(f"Prediction count: {len(preds)}")
        first_10_ts = [round(p.timestamp_sec, 2) for p in preds[:10]]
        last_10_ts = [round(p.timestamp_sec, 2) for p in preds[-10:]]
        print(f"First 10 prediction timestamps: {first_10_ts}")
        print(f"Last 10 prediction timestamps:  {last_10_ts}")

        raw_alerts = aggregators[m].raw_window_alerts
        merged_alerts = aggregators[m].get_merged_alerts()
        print(f"Number of raw alerts: {len(raw_alerts)}")
        print(f"Number of merged alerts: {len(merged_alerts)}")
        for ma in merged_alerts:
            print(f"  Merged alert: [{ma.start_time_sec:.2f}s - {ma.end_time_sec:.2f}s] class='{ma.predicted_class}' peak_conf={ma.peak_confidence} avg_conf={ma.average_confidence} key_frame_count={ma.key_frame_count}")

    model_alerts_dict = {m.replace("_", " ").title(): aggregators[m].get_merged_alerts() for m in CNN_MODELS}
    consensus_alerts = merge_multimodel_alerts(model_alerts_dict)
    print("\n4. Final Merged Consensus Alert Intervals:")
    for ca in consensus_alerts:
        print(f"  Consensus Alert: [{ca.start_time_sec:.2f}s - {ca.end_time_sec:.2f}s] duration={ca.duration_sec}s class='{ca.predicted_class}' agreeing={ca.agreeing_models} ({ca.num_agreeing_models}) peak_conf={ca.peak_confidence} avg_conf={ca.average_confidence} key_frame_count={ca.key_frame_count}")

if __name__ == "__main__":
    run_debug()
