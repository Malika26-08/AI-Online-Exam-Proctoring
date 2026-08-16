"""
Scratch script to prototype and verify the new SlidingWindowAggregator logic.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.config import (
    CLASS_NAMES, MOTION_THRESHOLD, FRAME_SKIP,
    SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.utils.video_loader import VideoLoader
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model
from src.aggregation.sliding_window import FramePrediction, FlaggedSegmentAlert, ConsensusSegmentAlert

class NewSlidingWindowAggregator:
    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_gap_sec: float = 1.0
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.max_gap_sec = float(max_gap_sec)
        self.reset()

    def reset(self):
        self.predictions: List[FramePrediction] = []

    def add_prediction(self, prediction: FramePrediction):
        self.predictions.append(prediction)

    def get_merged_alerts(self) -> List[FlaggedSegmentAlert]:
        if not self.predictions:
            return []

        # 1. Identify contiguous runs of abnormal predictions (conf >= 0.65, class != 'normal')
        raw_runs: List[List[FramePrediction]] = []
        current_run: List[FramePrediction] = []
        current_class: Optional[str] = None

        for p in self.predictions:
            is_abnormal = (p.predicted_class != "normal") and (p.confidence >= self.confidence_threshold)

            if is_abnormal:
                if not current_run:
                    current_run = [p]
                    current_class = p.predicted_class
                elif p.predicted_class == current_class:
                    current_run.append(p)
                else:
                    raw_runs.append(current_run)
                    current_run = [p]
                    current_class = p.predicted_class
            else:
                if current_run:
                    raw_runs.append(current_run)
                    current_run = []
                    current_class = None

        if current_run:
            raw_runs.append(current_run)

        if not raw_runs:
            return []

        # Convert raw runs to initial segment alerts
        initial_alerts: List[FlaggedSegmentAlert] = []
        for run in raw_runs:
            start_ts = run[0].timestamp_sec
            end_ts = run[-1].timestamp_sec
            dur = max(0.0, end_ts - start_ts)
            cls = run[0].predicted_class
            peak_c = max(p.confidence for p in run)
            avg_c = float(np.mean([p.confidence for p in run]))
            kf_cnt = len(set(p.frame_idx for p in run))

            initial_alerts.append(
                FlaggedSegmentAlert(
                    start_time_sec=round(start_ts, 2),
                    end_time_sec=round(end_ts, 2),
                    duration_sec=round(dur, 2),
                    predicted_class=cls,
                    peak_confidence=round(peak_c, 4),
                    average_confidence=round(avg_c, 4),
                    key_frame_count=kf_cnt
                )
            )

        # 2. Merge contiguous or close alerts of SAME class within max_gap_sec
        merged: List[FlaggedSegmentAlert] = []
        curr = initial_alerts[0]

        # Helper to count unique key-frames in timestamp range for this aggregator
        def count_keyframes_in_range(t_start: float, t_end: float) -> int:
            return len(set(p.frame_idx for p in self.predictions if t_start <= p.timestamp_sec <= t_end))

        for nxt in initial_alerts[1:]:
            if (nxt.predicted_class == curr.predicted_class and
                nxt.start_time_sec <= curr.end_time_sec + self.max_gap_sec):

                new_start = min(curr.start_time_sec, nxt.start_time_sec)
                new_end = max(curr.end_time_sec, nxt.end_time_sec)
                new_dur = new_end - new_start
                new_peak = max(curr.peak_confidence, nxt.peak_confidence)
                new_avg = (curr.average_confidence + nxt.average_confidence) / 2.0
                new_kf_cnt = count_keyframes_in_range(new_start, new_end)

                curr = FlaggedSegmentAlert(
                    start_time_sec=round(new_start, 2),
                    end_time_sec=round(new_end, 2),
                    duration_sec=round(new_dur, 2),
                    predicted_class=curr.predicted_class,
                    peak_confidence=round(new_peak, 4),
                    average_confidence=round(new_avg, 4),
                    key_frame_count=new_kf_cnt
                )
            else:
                merged.append(curr)
                curr = nxt

        merged.append(curr)
        return merged

def new_merge_multimodel_alerts(
    model_alerts_dict: Dict[str, List[FlaggedSegmentAlert]],
    all_predictions_map: Dict[str, List[FramePrediction]],
    gap_tolerance_sec: float = 1.0
) -> List[ConsensusSegmentAlert]:
    if not model_alerts_dict:
        return []

    class_groups: Dict[str, List[tuple]] = {}
    for model_name, alerts in model_alerts_dict.items():
        for alert in alerts:
            cls = alert.predicted_class
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append((model_name, alert))

    consensus_alerts: List[ConsensusSegmentAlert] = []

    for cls, items in class_groups.items():
        items_sorted = sorted(items, key=lambda x: x[1].start_time_sec)
        clusters: List[List[tuple]] = []

        for model_name, alert in items_sorted:
            if not clusters:
                clusters.append([(model_name, alert)])
            else:
                last_cluster = clusters[-1]
                max_end = max(a.end_time_sec for _, a in last_cluster)
                if alert.start_time_sec <= max_end + gap_tolerance_sec:
                    last_cluster.append((model_name, alert))
                else:
                    clusters.append([(model_name, alert)])

        for cluster in clusters:
            start_t = min(a.start_time_sec for _, a in cluster)
            end_t = max(a.end_time_sec for _, a in cluster)
            dur = max(0.0, end_t - start_t)

            agreeing_models = list(dict.fromkeys(m for m, _ in cluster))
            num_agreeing = len(agreeing_models)

            peak_conf = max(a.peak_confidence for _, a in cluster)
            avg_conf = float(np.mean([a.average_confidence for _, a in cluster]))

            # Unique key-frames count across all models in this timestamp interval
            unique_kf_indices = set()
            for m in agreeing_models:
                preds = all_predictions_map.get(m, [])
                for p in preds:
                    if start_t <= p.timestamp_sec <= end_t:
                        unique_kf_indices.add(p.frame_idx)
            key_frame_cnt = len(unique_kf_indices)

            consensus_alerts.append(
                ConsensusSegmentAlert(
                    start_time_sec=round(start_t, 2),
                    end_time_sec=round(end_t, 2),
                    duration_sec=round(dur, 2),
                    predicted_class=cls,
                    agreeing_models=agreeing_models,
                    num_agreeing_models=num_agreeing,
                    peak_confidence=round(peak_conf, 4),
                    average_confidence=round(avg_conf, 4),
                    key_frame_count=key_frame_cnt
                )
            )

    consensus_alerts.sort(key=lambda x: x.start_time_sec)
    return consensus_alerts


def test_real_video():
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    video_files = list(temp_dir.glob("tmp*.mp4"))
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)

    loader = VideoLoader(video_path)
    meta = loader.metadata
    print(f"Testing real video: {video_path.name} ({meta['frame_count']} frames, {meta['duration_sec']:.2f}s)")

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
        aggregators[m] = NewSlidingWindowAggregator(confidence_threshold=CONFIDENCE_THRESHOLD)
        predictions_map[m] = []

    extracted_kf = 0
    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        is_kf, sum_diff, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
        if is_kf:
            extracted_kf += 1
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

    print(f"Extracted Key-Frames: {extracted_kf}")

    model_alerts_dict = {}
    for m in CNN_MODELS:
        m_display = m.replace("_", " ").title()
        alerts = aggregators[m].get_merged_alerts()
        model_alerts_dict[m_display] = alerts
        print(f"\n--- Model {m_display} Merged Alerts ({len(alerts)}) ---")
        for a in alerts:
            print(f"  [{a.start_time_sec}s - {a.end_time_sec}s] {a.predicted_class} | Peak: {a.peak_confidence*100:.1f}% | Avg: {a.average_confidence*100:.1f}% | Key-Frames: {a.key_frame_count}")

    disp_predictions_map = {m.replace("_", " ").title(): predictions_map[m] for m in CNN_MODELS}
    consensus = new_merge_multimodel_alerts(model_alerts_dict, disp_predictions_map)

    print(f"\n=======================================================")
    print(f"FINAL CONSENSUS TIMELINE ({len(consensus)} alerts)")
    print(f"=======================================================")
    for c in consensus:
        print(f"  [{c.start_time_sec:5.2f}s - {c.end_time_sec:5.2f}s] ({c.duration_sec:4.1f}s) {c.predicted_class:<15} | Models: {c.agreeing_models} ({c.num_agreeing_models}) | Peak: {c.peak_confidence*100:.1f}% | Avg: {c.average_confidence*100:.1f}% | Key-Frames: {c.key_frame_count}")

if __name__ == "__main__":
    test_real_video()
