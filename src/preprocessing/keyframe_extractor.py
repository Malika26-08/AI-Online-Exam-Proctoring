"""
Key-Frame Extraction Module for Online Exam Proctoring Pipeline.
Implements fixed-threshold motion difference frame selection:
  SumDiff = sum(abs(F_t - F_{t-1}))
  Key-Frame Condition: SumDiff > T (where T = 340,000 and frame_skip = 3)
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from src.config import MOTION_THRESHOLD, FRAME_SKIP
from src.utils.logger import get_logger

logger = get_logger("keyframe_extractor")


class KeyFrameExtractor:
    """
    Evaluates frame-to-frame pixel differences against fixed motion threshold T.
    Retains frames exceeding T to reduce redundant non-moving video frames before model inference.
    """

    def __init__(
        self,
        threshold: float = MOTION_THRESHOLD,
        frame_skip: int = FRAME_SKIP
    ):
        # Always bind threshold and frame_skip parameters (defaulting to config constants)
        self.threshold = float(threshold)
        self.frame_skip = int(frame_skip)
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.reset()

        logger.info(f"Initialized KeyFrameExtractor (Fixed Threshold T={self.threshold}, frame_skip={self.frame_skip})")

    def reset(self):
        """Resets key-frame extractor state between videos."""
        self.prev_frame_gray = None

    def compute_frame_difference(self, current_frame: np.ndarray) -> float:
        """
        Computes absolute frame difference sum:
        SumDiff = sum(abs(F_t - F_{t-1}))
        """
        if current_frame is None or current_frame.size == 0:
            raise ValueError("Invalid empty frame passed to difference calculation.")

        # Convert to single-channel grayscale for motion difference computation
        if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_frame

        if self.prev_frame_gray is None:
            # First frame in video: store reference and consider initial frame retained
            self.prev_frame_gray = curr_gray
            return float("inf")

        # Compute element-wise absolute difference and sum across all pixels
        diff_matrix = cv2.absdiff(curr_gray, self.prev_frame_gray)
        sum_diff = float(np.sum(diff_matrix))

        return sum_diff

    def process_frame(
        self,
        current_frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Processes candidate frame and determines if key-frame condition is met.
        Returns:
          is_key_frame: bool (True if frame is retained)
          sum_diff: float (Calculated pixel difference sum)
          metadata: Dict (Diagnostic information)
        """
        sum_diff = self.compute_frame_difference(current_frame)

        # First frame is always retained as baseline key-frame
        if sum_diff == float("inf"):
            is_key_frame = True
            reason = "initial_baseline_frame"
        else:
            is_key_frame = sum_diff > self.threshold
            reason = "exceeds_fixed_threshold" if is_key_frame else "below_threshold_discarded"

        if is_key_frame:
            # Update reference frame to current gray frame when a key-frame is retained
            if len(current_frame.shape) == 3:
                self.prev_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                self.prev_frame_gray = current_frame

        metadata = {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp,
            "sum_diff": sum_diff,
            "threshold": self.threshold,
            "is_key_frame": is_key_frame,
            "reason": reason
        }

        return is_key_frame, sum_diff, metadata
