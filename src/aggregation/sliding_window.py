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
    Maintains a temporal sliding window over key-frame predictions.
    Computes majority vote / mean probability per time window and merges contiguous alerts.
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
        self.buffer: List[FramePrediction] = []
        self.raw_window_alerts: List[FlaggedSegmentAlert] = []

    def add_prediction(self, prediction: FramePrediction) -> Optional[FlaggedSegmentAlert]:
        """
        Adds a key-frame prediction and evaluates the current sliding time window.
        Returns a FlaggedSegmentAlert if an abnormal activity threshold is exceeded.
        """
        self.buffer.append(prediction)

        # Evict predictions older than window_seconds relative to current timestamp
        current_time = prediction.timestamp_sec
        self.buffer = [
            p for p in self.buffer
            if (current_time - p.timestamp_sec) <= self.window_seconds
        ]

        if not self.buffer:
            return None

        # Calculate class-wise average confidence within current window
        class_scores: Dict[str, float] = {cls: 0.0 for cls in CLASS_NAMES}

        for p in self.buffer:
            if p.probabilities:
                for cls, prob in p.probabilities.items():
                    if cls not in class_scores:
                        class_scores[cls] = 0.0
                    class_scores[cls] += prob
            else:
                if p.predicted_class not in class_scores:
                    class_scores[p.predicted_class] = 0.0
                class_scores[p.predicted_class] += p.confidence

        window_size = len(self.buffer)
        for cls in class_scores:
            class_scores[cls] /= window_size

        # Find dominant class in window
        top_class = max(class_scores, key=class_scores.get)
        top_confidence = class_scores[top_class]

        # Ignore normal behavior class or predictions below threshold
        if top_class == "normal" or top_confidence < self.confidence_threshold:
            return None

        start_time = self.buffer[0].timestamp_sec
        end_time = self.buffer[-1].timestamp_sec
        duration = max(0.0, end_time - start_time)
        peak_conf = max(p.confidence for p in self.buffer if p.predicted_class == top_class)

        alert = FlaggedSegmentAlert(
            start_time_sec=round(start_time, 2),
            end_time_sec=round(end_time, 2),
            duration_sec=round(duration, 2),
            predicted_class=top_class,
            peak_confidence=round(peak_conf, 4),
            average_confidence=round(top_confidence, 4),
            key_frame_count=window_size
        )

        self.raw_window_alerts.append(alert)
        return alert

    def get_merged_alerts(self) -> List[FlaggedSegmentAlert]:
        """
        Merges overlapping or contiguous window alerts of the same class into unified intervals.
        Returns clean, deduplicated timeline alerts.
        """
        if not self.raw_window_alerts:
            return []

        merged: List[FlaggedSegmentAlert] = []
        current = self.raw_window_alerts[0]

        for next_alert in self.raw_window_alerts[1:]:
            # If same class and overlapping/contiguous in time (within window_seconds gap)
            if (next_alert.predicted_class == current.predicted_class and
                next_alert.start_time_sec <= current.end_time_sec + self.window_seconds):
                # Merge interval
                new_start = min(current.start_time_sec, next_alert.start_time_sec)
                new_end = max(current.end_time_sec, next_alert.end_time_sec)
                new_duration = new_end - new_start
                new_peak = max(current.peak_confidence, next_alert.peak_confidence)
                new_avg = (current.average_confidence + next_alert.average_confidence) / 2.0
                new_count = current.key_frame_count + next_alert.key_frame_count

                current = FlaggedSegmentAlert(
                    start_time_sec=round(new_start, 2),
                    end_time_sec=round(new_end, 2),
                    duration_sec=round(new_duration, 2),
                    predicted_class=current.predicted_class,
                    peak_confidence=round(new_peak, 4),
                    average_confidence=round(new_avg, 4),
                    key_frame_count=new_count
                )
            else:
                merged.append(current)
                current = next_alert

        merged.append(current)
        return merged


def merge_multimodel_alerts(
    model_alerts_dict: Dict[str, List[FlaggedSegmentAlert]],
    gap_tolerance_sec: float = 3.0
) -> List[ConsensusSegmentAlert]:
    """
    Merges per-model flagged segment alerts across multiple CNN models into unified consensus alerts.
    
    Args:
        model_alerts_dict: Dictionary mapping model display name -> list of FlaggedSegmentAlert.
        gap_tolerance_sec: Time gap tolerance in seconds to merge contiguous/overlapping intervals.

    Returns:
        List of ConsensusSegmentAlert sorted chronologically.
    """
    if not model_alerts_dict:
        return []

    # Group alerts by predicted class
    class_groups: Dict[str, List[Tuple[str, FlaggedSegmentAlert]]] = {}
    for model_name, alerts in model_alerts_dict.items():
        for alert in alerts:
            cls = alert.predicted_class
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append((model_name, alert))

    consensus_alerts: List[ConsensusSegmentAlert] = []

    for cls, items in class_groups.items():
        # Sort items by start_time_sec
        items_sorted = sorted(items, key=lambda x: x[1].start_time_sec)

        # Merge overlapping/contiguous intervals for this class
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
            start_time = min(a.start_time_sec for _, a in cluster)
            end_time = max(a.end_time_sec for _, a in cluster)
            duration = max(0.0, end_time - start_time)

            agreeing_models = list(dict.fromkeys(m for m, _ in cluster))
            num_agreeing = len(agreeing_models)

            peak_conf = max(a.peak_confidence for _, a in cluster)
            avg_conf = float(np.mean([a.average_confidence for _, a in cluster]))
            total_frames = max(a.key_frame_count for _, a in cluster)

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
                    key_frame_count=total_frames
                )
            )

    # Sort consensus alerts chronologically by start_time_sec
    consensus_alerts.sort(key=lambda x: x.start_time_sec)
    return consensus_alerts

