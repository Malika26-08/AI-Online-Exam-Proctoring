"""
Final verification script to execute the real 33-second webcam recording through the updated pipeline
and report all required metrics.
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
from src.aggregation.report_generator import ReportGenerator

def run_real_video_analysis():
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    video_files = list(temp_dir.glob("tmp*.mp4"))
    if not video_files:
        print("No temp video found.")
        return
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)

    loader = VideoLoader(video_path)
    meta = loader.metadata
    total_frames = meta["frame_count"]

    CNN_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]
    keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)

    preprocessors = {}
    models = {}
    aggregators = {}
    predictions_map = {}

    for m in CNN_MODELS:
        target_size = INCEPTION_IMAGE_SIZE if "inception" in m else DEFAULT_IMAGE_SIZE
        preprocessors[m] = FramePreprocessor(target_size=target_size)
        models[m] = build_model(m, load_best_checkpoint=True)
        if hasattr(models[m], "eval"):
            models[m].eval()
        aggregators[m] = SlidingWindowAggregator(confidence_threshold=CONFIDENCE_THRESHOLD)
        predictions_map[m] = []

    extracted_key_frames = 0
    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        is_kf, sum_diff, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
        if is_kf:
            extracted_key_frames += 1
            for m in CNN_MODELS:
                proc = preprocessors[m].preprocess(raw_frame)
                tensor_img = torch.from_numpy(proc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    logits = models[m](tensor_img)
                    probs_tensor = torch.softmax(logits, dim=1).squeeze(0).numpy()

                top_idx = int(np.argmax(probs_tensor))
                pred_class = CLASS_NAMES[top_idx]
                conf = float(probs_tensor[top_idx])

                pred = FramePrediction(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    predicted_class=pred_class,
                    confidence=conf
                )
                predictions_map[m].append(pred)
                aggregators[m].add_prediction(pred)

    # Compile Per-Model Reports & Alerts
    report_gen = ReportGenerator()
    model_alerts_dict = {}
    per_model_reports = {}

    for m in CNN_MODELS:
        m_display = m.replace("_", " ").title()
        alerts = aggregators[m].get_merged_alerts()
        model_alerts_dict[m_display] = alerts
        per_model_reports[m_display] = report_gen.generate_session_report(
            video_name=video_path.name,
            total_duration_sec=meta["duration_sec"],
            total_frames=total_frames,
            key_frames_analyzed=extracted_key_frames,
            model_name=m,
            alerts=alerts,
            json_filename=f"{m}_session_report.json",
            csv_filename=f"{m}_session_report.csv"
        )

    # Multi-Model Consensus Alert Merging
    disp_predictions_map = {m.replace("_", " ").title(): predictions_map[m] for m in CNN_MODELS}
    consensus_alerts = merge_multimodel_alerts(model_alerts_dict, all_predictions_map=disp_predictions_map)

    consensus_report = report_gen.generate_session_report(
        video_name=video_path.name,
        total_duration_sec=meta["duration_sec"],
        total_frames=total_frames,
        key_frames_analyzed=extracted_key_frames,
        model_name="Multi-CNN Consensus",
        alerts=consensus_alerts,
        json_filename="consensus_session_report.json",
        csv_filename="consensus_session_report.csv"
    )

    print("\n=======================================================")
    print("REAL WEBCAM VIDEO PROCTORING ANALYSIS REPORT")
    print("=======================================================")
    print(f"Video File           : {video_path.name}")
    print(f"Video Duration       : {meta['duration_sec']:.2f} seconds ({consensus_report['session_metadata']['total_duration_formatted']})")
    print(f"Total Video Frames   : {total_frames}")
    print(f"Extracted Key-Frames : {extracted_key_frames} (frame_skip={FRAME_SKIP})")
    print(f"Redundant Cut %      : {consensus_report['session_metadata']['key_frame_reduction_percent']}%")

    print("\n--- PREDICTION COUNT PER MODEL ---")
    for m in CNN_MODELS:
        m_disp = m.replace('_', ' ').title()
        print(f"  • {m_disp:<20}: {len(predictions_map[m])} key-frame predictions (raw alerts: {len(aggregators[m].raw_window_alerts)}, merged: {len(model_alerts_dict[m_disp])})")

    print("\n--- PER-MODEL MERGED ALERT INTERVALS ---")
    for m in CNN_MODELS:
        m_disp = m.replace('_', ' ').title()
        print(f"\nModel: {m_disp}")
        alerts = model_alerts_dict[m_disp]
        if not alerts:
            print("  • Clean Session")
        for a in alerts:
            print(f"  • [{a.start_time_sec:5.2f}s - {a.end_time_sec:5.2f}s] ({a.duration_sec:4.1f}s) {a.predicted_class:<15} | Peak Conf: {a.peak_confidence*100:.1f}% | Avg Conf: {a.average_confidence*100:.1f}% | Key-Frames: {a.key_frame_count}")

    print("\n=======================================================")
    print("FINAL CONSENSUS TIMELINE & ALERT INTERVALS")
    print("=======================================================")
    for idx, c in enumerate(consensus_alerts, 1):
        print(f"{idx:2d}. [{c.start_time_sec:5.2f}s - {c.end_time_sec:5.2f}s] ({c.duration_sec:4.1f}s) Activity: {c.predicted_class:<15} | Models: {c.agreeing_models} ({c.num_agreeing_models}) | Peak: {c.peak_confidence*100:.1f}% | Avg: {c.average_confidence*100:.1f}% | Key-Frames: {c.key_frame_count}")

    print("\n--- FINAL CONSENSUS CLASS DISTRIBUTION & DURATION ---")
    class_counts = consensus_report["summary_statistics"]["class_wise_counts"]
    class_durations = consensus_report["summary_statistics"].get("class_wise_duration_sec", {})
    for cls in CLASS_NAMES:
        cnt = class_counts.get(cls, 0)
        dur = class_durations.get(cls, 0.0)
        print(f"  • {cls.replace('_', ' ').title():<25}: {cnt} alerts | {dur:.2f}s total abnormal duration")

if __name__ == "__main__":
    run_real_video_analysis()
