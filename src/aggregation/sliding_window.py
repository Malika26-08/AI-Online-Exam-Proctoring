"""
Sliding Window Aggregation Module for Online Exam Proctoring Pipeline.
Aggregates per-key-frame predictions over a temporal sliding window to reduce false positives
and produce unified timestamped abnormal activity alerts.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Union
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
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        window_size_seconds: Optional[float] = None,
        fps: Optional[float] = None
    ):
        if window_size_seconds is not None:
            window_seconds = window_size_seconds
        self.window_seconds = float(window_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.fps = fps
        self.reset()

        logger.info(
            f"Initialized SlidingWindowAggregator (window={self.window_seconds}s, threshold={self.confidence_threshold})"
        )

    def reset(self):
        """Resets aggregator state between video sessions."""
        self.predictions: List[FramePrediction] = []
        self.raw_window_alerts: List[FlaggedSegmentAlert] = []

    @property
    def predictions_history(self) -> List[FramePrediction]:
        return self.predictions

    def add_prediction(
        self,
        frame_idx: Union[int, FramePrediction] = 0,
        timestamp_sec: float = 0.0,
        predicted_class: str = "normal",
        confidence: Optional[float] = None,
        class_index: int = 0,
        probabilities: Optional[Any] = None,
        prediction: Optional[FramePrediction] = None
    ) -> Optional[FlaggedSegmentAlert]:
        """
        Adds a key-frame prediction to the aggregator buffer.
        Supports both FramePrediction dataclass instance and individual arguments.
        """
        if isinstance(frame_idx, FramePrediction):
            pred_obj = frame_idx
        elif prediction is not None:
            pred_obj = prediction
        else:
            conf = float(confidence) if confidence is not None else (
                float(np.max(probabilities)) if probabilities is not None else 1.0
            )
            prob_dict = None
            if probabilities is not None:
                if isinstance(probabilities, dict):
                    prob_dict = probabilities
                elif isinstance(probabilities, (list, tuple, np.ndarray)):
                    prob_dict = {
                        CLASS_NAMES[i]: float(p)
                        for i, p in enumerate(probabilities)
                        if i < len(CLASS_NAMES)
                    }

            pred_obj = FramePrediction(
                frame_idx=int(frame_idx),
                timestamp_sec=float(timestamp_sec),
                predicted_class=str(predicted_class),
                confidence=conf,
                probabilities=prob_dict
            )

        self.predictions.append(pred_obj)

        is_abnormal = (pred_obj.predicted_class != "normal") and (pred_obj.confidence >= self.confidence_threshold)
        if is_abnormal:
            alert = FlaggedSegmentAlert(
                start_time_sec=round(pred_obj.timestamp_sec, 2),
                end_time_sec=round(pred_obj.timestamp_sec, 2),
                duration_sec=0.0,
                predicted_class=pred_obj.predicted_class,
                peak_confidence=round(pred_obj.confidence, 4),
                average_confidence=round(pred_obj.confidence, 4),
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

    def get_session_summary(self, model_name: str = "model") -> Dict[str, Any]:
        """Generates full session summary dictionary for report generation."""
        merged_timeline = self.get_merged_alerts()
        total_preds = len(self.predictions)

        class_counts = {c: 0 for c in CLASS_NAMES}
        for p in self.predictions:
            if p.predicted_class in class_counts:
                class_counts[p.predicted_class] += 1
            else:
                class_counts[p.predicted_class] = 1

        abnormal_count = sum(c for k, c in class_counts.items() if k != "normal")
        abnormal_pct = (abnormal_count / max(1, total_preds)) * 100.0

        return {
            "model_name": model_name,
            "timeline": [asdict(a) for a in merged_timeline],
            "raw_alerts": [asdict(a) for a in self.raw_window_alerts],
            "predictions": [asdict(p) for p in self.predictions],
            "summary_statistics": {
                "total_key_frames": total_preds,
                "abnormal_key_frames": abnormal_count,
                "abnormal_percentage": round(abnormal_pct, 2),
                "class_wise_counts": class_counts
            }
        }


def merge_multimodel_alerts(
    model_alerts_dict: Optional[Dict[str, Any]] = None,
    all_predictions_map: Optional[Dict[str, List[FramePrediction]]] = None,
    gap_tolerance_sec: float = 1.0,
    min_agreeing_models: int = 2,
    min_consensus_frames: int = 2,
    per_model_reports: Optional[Dict[str, Any]] = None,
    video_filename: Optional[str] = None,
    fps: float = 30.0,
    total_frames: int = 0,
    key_frames_analyzed: int = 0,
    key_frame_reduction_percent: float = 0.0,
    **kwargs
) -> Union[List[ConsensusSegmentAlert], Dict[str, Any]]:
    """
    Computes time-aligned multi-model consensus alerts across 4 CNN models.
    Requires at least min_agreeing_models (default 2) to agree on the same class
    over a temporal window, eliminating single-model low-confidence false positives.
    """
    source_reports = per_model_reports if per_model_reports is not None else model_alerts_dict
    if source_reports is None:
        source_reports = {}

    extracted_alerts_dict: Dict[str, List[FlaggedSegmentAlert]] = {}
    extracted_preds_map: Dict[str, List[FramePrediction]] = {}

    for m_name, item in source_reports.items():
        if isinstance(item, list):
            extracted_alerts_dict[m_name] = [
                a if isinstance(a, FlaggedSegmentAlert) else FlaggedSegmentAlert(**a) for a in item
            ]
        elif isinstance(item, dict):
            t_list = item.get("timeline", [])
            extracted_alerts_dict[m_name] = [
                a if isinstance(a, FlaggedSegmentAlert) else FlaggedSegmentAlert(**a) for a in t_list
            ]
            p_list = item.get("predictions", [])
            if p_list:
                extracted_preds_map[m_name] = [
                    p if isinstance(p, FramePrediction) else FramePrediction(**p) for p in p_list
                ]

    preds_map = all_predictions_map if all_predictions_map else (extracted_preds_map if extracted_preds_map else None)

    consensus_alerts: List[ConsensusSegmentAlert] = []

    if preds_map:
        ts_map: Dict[Tuple[int, float], Dict[str, FramePrediction]] = {}
        for model_name, preds in preds_map.items():
            for p in preds:
                key = (p.frame_idx, round(p.timestamp_sec, 2))
                if key not in ts_map:
                    ts_map[key] = {}
                ts_map[key][model_name] = p

        sorted_keys = sorted(ts_map.keys(), key=lambda x: x[1])
        consensus_points = []
        for frame_idx, ts_sec in sorted_keys:
            m_preds = ts_map[(frame_idx, ts_sec)]
            class_groups: Dict[str, List[Tuple[str, FramePrediction]]] = {}
            for m_name, pred in m_preds.items():
                if pred.predicted_class != "normal" and pred.confidence >= CONFIDENCE_THRESHOLD:
                    cls = pred.predicted_class
                    if cls not in class_groups:
                        class_groups[cls] = []
                    class_groups[cls].append((m_name, pred))

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

        if consensus_points:
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

            for ev in events:
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
    else:
        class_groups: Dict[str, List[Tuple[str, FlaggedSegmentAlert]]] = {}
        for model_name, alerts in extracted_alerts_dict.items():
            for alert in alerts:
                cls = alert.predicted_class
                if cls not in class_groups:
                    class_groups[cls] = []
                class_groups[cls].append((model_name, alert))

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

    # Return full report dict if video_filename or per_model_reports is passed
    if video_filename is not None or per_model_reports is not None:
        timeline_dicts = []
        for a in consensus_alerts:
            entry = asdict(a)
            if a.num_agreeing_models >= 3 and a.average_confidence >= 0.75:
                entry["evaluation_status"] = "High Consensus Violation"
            elif a.num_agreeing_models >= 2:
                entry["evaluation_status"] = "Potential Behavior Flag"
            else:
                entry["evaluation_status"] = "Single Model Notice"
            timeline_dicts.append(entry)

        class_counts = {c: 0 for c in CLASS_NAMES}
        for a in consensus_alerts:
            if a.predicted_class in class_counts:
                class_counts[a.predicted_class] += 1
            else:
                class_counts[a.predicted_class] = 1

        total_dur = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

        return {
            "session_metadata": {
                "video_name": video_filename or "exam_video.mp4",
                "total_duration_sec": round(total_dur, 2),
                "total_frames": total_frames,
                "key_frames_analyzed": key_frames_analyzed,
                "key_frame_reduction_percent": round(key_frame_reduction_percent, 2),
                "fps": round(fps, 2),
                "consensus_threshold_models": min_agreeing_models
            },
            "summary_statistics": {
                "total_flagged_segments": len(consensus_alerts),
                "class_wise_counts": class_counts
            },
            "timeline": timeline_dicts
        }

    return consensus_alerts
