"""
Sliding Window Aggregation Module for Online Exam Proctoring Pipeline.
Aggregates per-key-frame predictions over a temporal sliding window to reduce false positives
and produce unified timestamped abnormal activity alerts.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from src.config import (
    CLASS_NAMES, SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD
)
from src.utils.logger import get_logger

logger = get_logger("sliding_window")


@dataclass
class FramePrediction:
    """Represents prediction output for a single key-frame."""
    frame_idx: int
    timestamp_sec: float
    predicted_class: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


@dataclass
class FlaggedSegmentAlert:
    """Represents a unified temporal alert for a flagged abnormal activity segment."""
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    predicted_class: str
    peak_confidence: float
    average_confidence: float
    key_frame_count: int


@dataclass
class ConsensusSegmentAlert:
    """Represents a merged consensus abnormal segment alert across multiple CNN models."""
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    predicted_class: str
    agreeing_models: List[str]
    num_agreeing_models: int
    peak_confidence: float
    average_confidence: float
    key_frame_count: int


class SlidingWindowAggregator:
    """
    Maintains a temporal log over key-frame predictions.
    Computes per-model temporal alert segments where normal or low-confidence frames break alerts,
    and merges contiguous alerts within max_gap_sec.
    """

    def __init__(
        self,
        window_seconds: float = SLIDING_WINDOW_SECONDS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.window_seconds = float(window_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.reset()

        logger.info(
            f"Initialized SlidingWindowAggregator (window={self.window_seconds}s, threshold={self.confidence_threshold})"
        )

    def reset(self):
        """Resets aggregator state between video sessions."""
        self.predictions: List[FramePrediction] = []
        self.raw_window_alerts: List[FlaggedSegmentAlert] = []

    def add_prediction(self, prediction: FramePrediction) -> Optional[FlaggedSegmentAlert]:
        """
        Adds a key-frame prediction to the aggregator buffer.
        """
        self.predictions.append(prediction)
        
        # If this frame is abnormal eligible, create a single-frame candidate alert
        is_abnormal = (prediction.predicted_class != "normal") and (prediction.confidence >= self.confidence_threshold)
        if is_abnormal:
            alert = FlaggedSegmentAlert(
                start_time_sec=round(prediction.timestamp_sec, 2),
                end_time_sec=round(prediction.timestamp_sec, 2),
                duration_sec=0.0,
                predicted_class=prediction.predicted_class,
                peak_confidence=round(prediction.confidence, 4),
                average_confidence=round(prediction.confidence, 4),
                key_frame_count=1
            )
            self.raw_window_alerts.append(alert)
            return alert
        return None

    def get_merged_alerts(self, max_gap_sec: float = 1.0) -> List[FlaggedSegmentAlert]:
        """
        Merges contiguous or overlapping abnormal predictions of the same class into unified intervals.
        Normal frames (predicted_class == 'normal') or low confidence frames (< confidence_threshold)
        break alerts. If normal gap exceeds max_gap_sec, separate alerts are returned.
        """
        if not self.predictions:
            return []

        # 1. Group consecutive abnormal predictions (conf >= threshold and class != 'normal')
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
                # Normal or low-confidence frame breaks the active run
                if current_run:
                    raw_runs.append(current_run)
                    current_run = []
                    current_class = None

        if current_run:
            raw_runs.append(current_run)

        if not raw_runs:
            return []

        # Convert raw runs into initial segment alerts
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

        # 2. Merge contiguous alerts of the SAME class within max_gap_sec
        merged: List[FlaggedSegmentAlert] = []
        curr = initial_alerts[0]

        def count_keyframes_in_range(t_start: float, t_end: float) -> int:
            return len(set(p.frame_idx for p in self.predictions if t_start <= p.timestamp_sec <= t_end))

        for nxt in initial_alerts[1:]:
            if (nxt.predicted_class == curr.predicted_class and
                nxt.start_time_sec <= curr.end_time_sec + max_gap_sec):

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


def merge_multimodel_alerts(
    model_alerts_dict: Dict[str, List[FlaggedSegmentAlert]],
    all_predictions_map: Optional[Dict[str, List[FramePrediction]]] = None,
    gap_tolerance_sec: float = 1.0,
    min_agreeing_models: int = 2,
    min_consensus_frames: int = 2
) -> List[ConsensusSegmentAlert]:
    """
    Computes time-aligned multi-model consensus alerts across 4 CNN models.
    
    A consensus flag requires at least min_agreeing_models (default 2) to agree on the SAME
    abnormal class at approximately the same timestamp with confidence >= 65%, and at least
    min_consensus_frames (default 2) consecutive time-aligned key-frames. Single isolated
    key-frame consensus points do NOT become timeline alerts.

    Args:
        model_alerts_dict: Dictionary mapping model display name -> list of FlaggedSegmentAlert.
        all_predictions_map: Optional dictionary mapping model display name -> list of FramePrediction.
        gap_tolerance_sec: Time gap tolerance in seconds to merge contiguous consensus points.
        min_agreeing_models: Minimum number of agreeing models required for consensus (default 2).
        min_consensus_frames: Minimum consecutive consensus key-frames required for an alert event (default 2).

    Returns:
        List of ConsensusSegmentAlert sorted chronologically.
    """
    if not model_alerts_dict:
        return []

    # If per-keyframe frame predictions map is available, use exact time-aligned timestamp voting
    if all_predictions_map:
        # Group predictions by (frame_idx, timestamp_sec)
        ts_map: Dict[Tuple[int, float], Dict[str, FramePrediction]] = {}
        for model_name, preds in all_predictions_map.items():
            for p in preds:
                key = (p.frame_idx, round(p.timestamp_sec, 2))
                if key not in ts_map:
                    ts_map[key] = {}
                ts_map[key][model_name] = p

        # Sort timestamp keys chronologically
        sorted_keys = sorted(ts_map.keys(), key=lambda x: x[1])

        # Evaluate consensus at each key-frame timestamp
        consensus_points = []
        for frame_idx, ts_sec in sorted_keys:
            m_preds = ts_map[(frame_idx, ts_sec)]

            # Group eligible predictions (conf >= 0.65 and class != "normal") by class
            class_groups: Dict[str, List[Tuple[str, FramePrediction]]] = {}
            for m_name, pred in m_preds.items():
                if pred.predicted_class != "normal" and pred.confidence >= CONFIDENCE_THRESHOLD:
                    cls = pred.predicted_class
                    if cls not in class_groups:
                        class_groups[cls] = []
                    class_groups[cls].append((m_name, pred))

            # Retain classes where at least min_agreeing_models agree at this timestamp
            for cls, agreeing_pairs in class_groups.items():
                if len(agreeing_pairs) >= min_agreeing_models:
                    agreeing_models = [m for m, _ in agreeing_pairs]
                    confs = [p.confidence for _, p in agreeing_pairs]
                    consensus_points.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": ts_sec,
                        "predicted_class": cls,
                        "agreeing_models": agreeing_models,
                        "num_agreeing": len(agreeing_models),
                        "peak_confidence": max(confs),
                        "avg_confidence": float(np.mean(confs))
                    })

        if not consensus_points:
            return []

        # Group consecutive time-aligned consensus points into consensus alert events
        events: List[List[Dict[str, Any]]] = []
        curr_event: List[Dict[str, Any]] = []

        for pt in consensus_points:
            if not curr_event:
                curr_event = [pt]
            else:
                prev = curr_event[-1]
                if (pt["predicted_class"] == prev["predicted_class"] and
                    pt["timestamp_sec"] <= prev["timestamp_sec"] + gap_tolerance_sec):
                    curr_event.append(pt)
                else:
                    events.append(curr_event)
                    curr_event = [pt]

        if curr_event:
            events.append(curr_event)

        consensus_alerts: List[ConsensusSegmentAlert] = []
        for ev in events:
            # Require at least min_consensus_frames (default 2) key-frames per alert event
            if len(ev) < min_consensus_frames:
                continue

            start_t = ev[0]["timestamp_sec"]
            end_t = ev[-1]["timestamp_sec"]
            dur = max(0.0, end_t - start_t)
            cls = ev[0]["predicted_class"]

            agreeing_models = list(dict.fromkeys(m for pt in ev for m in pt["agreeing_models"]))
            num_agreeing = len(agreeing_models)

            peak_conf = max(pt["peak_confidence"] for pt in ev)
            avg_conf = float(np.mean([pt["avg_confidence"] for pt in ev]))
            kf_cnt = len(set(pt["frame_idx"] for pt in ev))

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
                    key_frame_count=kf_cnt
                )
            )

        consensus_alerts.sort(key=lambda x: x.start_time_sec)
        return consensus_alerts

    # Fallback when all_predictions_map is not available: interval overlap requiring min_agreeing_models
    class_groups: Dict[str, List[Tuple[str, FlaggedSegmentAlert]]] = {}
    for model_name, alerts in model_alerts_dict.items():
        for alert in alerts:
            cls = alert.predicted_class
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append((model_name, alert))

    consensus_alerts: List[ConsensusSegmentAlert] = []

    for cls, items in class_groups.items():
        items_sorted = sorted(items, key=lambda x: x[1].start_time_sec)
        merged_clusters: List[List[Tuple[str, FlaggedSegmentAlert]]] = []

        for model_name, alert in items_sorted:
            if not merged_clusters:
                merged_clusters.append([(model_name, alert)])
            else:
                last_cluster = merged_clusters[-1]
                max_end = max(a.end_time_sec for _, a in last_cluster)
                if alert.start_time_sec <= max_end + gap_tolerance_sec:
                    last_cluster.append((model_name, alert))
                else:
                    merged_clusters.append([(model_name, alert)])

        for cluster in merged_clusters:
            agreeing_models = list(dict.fromkeys(m for m, _ in cluster))
            num_agreeing = len(agreeing_models)

            # Exclude single-model predictions from consensus
            if num_agreeing < min_agreeing_models:
                continue

            start_time = min(a.start_time_sec for _, a in cluster)
            end_time = max(a.end_time_sec for _, a in cluster)
            duration = max(0.0, end_time - start_time)

            peak_conf = max(a.peak_confidence for _, a in cluster)
            avg_conf = float(np.mean([a.average_confidence for _, a in cluster]))
            key_frame_count = max(a.key_frame_count for _, a in cluster)

            consensus_alerts.append(
                ConsensusSegmentAlert(
                    start_time_sec=round(start_time, 2),
                    end_time_sec=round(end_time, 2),
                    duration_sec=round(duration, 2),
                    predicted_class=cls,
                    agreeing_models=agreeing_models,
                    num_agreeing_models=num_agreeing,
                    peak_confidence=round(peak_conf, 4),
                    average_confidence=round(avg_conf, 4),
                    key_frame_count=key_frame_count
                )
            )

    consensus_alerts.sort(key=lambda x: x.start_time_sec)
    return consensus_alerts



