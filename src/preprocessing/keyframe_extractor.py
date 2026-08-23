"""
Key-Frame Extraction Module for Online Exam Proctoring Pipeline.
Implements fixed-threshold motion difference frame selection:
  SumDiff = sum(abs(F_t - F_{t-1}))
  Key-Frame Condition: SumDiff > T (where T = 340,000 and frame_skip = 3)
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Generator
import cv2
import numpy as np
from src.config import MOTION_THRESHOLD, FRAME_SKIP
from src.utils.logger import get_logger
from src.utils.video_loader import VideoLoader

logger = get_logger("keyframe_extractor")


class KeyFrameExtractor:
    """
    Evaluates frame-to-frame pixel differences against motion threshold T.
    Normalizes pixel area across resolutions to ensure invariant key-frame extraction
    for general real-world examination videos (480p, 720p, 1080p).
    """

    def __init__(
        self,
        threshold: float = MOTION_THRESHOLD,
        frame_skip: int = FRAME_SKIP,
        motion_threshold: Optional[float] = None,
        min_keyframe_interval: int = 1
    ):
        eff_threshold = motion_threshold if motion_threshold is not None else threshold
        self.threshold = float(eff_threshold)
        self.motion_threshold = float(eff_threshold)
        self.frame_skip = int(frame_skip)
        self.min_keyframe_interval = int(min_keyframe_interval)
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.last_motion_sum: float = 0.0
        self.reset()

        logger.info(f"Initialized KeyFrameExtractor (Fixed Threshold T={self.threshold}, frame_skip={self.frame_skip})")

    def reset(self):
        """Resets key-frame extractor state between videos."""
        self.prev_frame_gray = None
        self.last_motion_sum = 0.0

    def compute_frame_difference(self, current_frame: np.ndarray) -> float:
        """
        Computes absolute frame difference sum:
        SumDiff = sum(abs(F_t - F_{t-1}))
        Normalized to 640x480 resolution baseline for resolution-invariant motion detection.
        """
        if current_frame is None or current_frame.size == 0:
            raise ValueError("Invalid empty frame passed to difference calculation.")

        if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_frame

        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_gray
            self.last_motion_sum = float("inf")
            return float("inf")

        diff = cv2.absdiff(curr_gray, self.prev_frame_gray)
        raw_sum_diff = float(np.sum(diff))

        # Resolution area normalization relative to 640x480 baseline (307,200 pixels)
        h, w = curr_gray.shape[:2]
        area_scale = (w * h) / 307200.0 if (w * h > 0) else 1.0
        norm_sum_diff = raw_sum_diff / max(0.1, area_scale)

        self.last_motion_sum = norm_sum_diff
        self.prev_frame_gray = curr_gray
        return norm_sum_diff

    def should_keep_frame(self, current_frame: np.ndarray) -> bool:
        """Returns True if the frame-to-frame difference exceeds threshold T or is the initial frame."""
        sum_diff = self.compute_frame_difference(current_frame)
        return sum_diff > self.threshold

    def process_frame(
        self,
        current_frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Processes a single frame and evaluates if it qualifies as a key-frame.
        Returns Tuple[is_key_frame: bool, sum_diff: float, metadata: dict].
        """
        sum_diff = self.compute_frame_difference(current_frame)
        if sum_diff == float("inf"):
            is_key_frame = True
            reason = "initial_baseline_frame"
        elif sum_diff > self.threshold:
            is_key_frame = True
            reason = "exceeds_fixed_threshold"
        else:
            is_key_frame = False
            reason = "below_threshold_discarded"

        metadata = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "sum_diff": sum_diff,
            "threshold": self.threshold,
            "reason": reason
        }

        return is_key_frame, sum_diff, metadata

    def extract_keyframes(self, video_path: Path) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Processes video file with VideoLoader and yields (frame_idx, timestamp_sec, raw_frame)
        only for frames exceeding the motion threshold T.
        """
        loader = VideoLoader(video_path, frame_skip=self.frame_skip)
        self.reset()

        for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=self.frame_skip):
            if self.should_keep_frame(raw_frame):
                yield (frame_idx, timestamp_sec, raw_frame)
