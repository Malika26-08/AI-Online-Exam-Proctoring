"""
Unit Tests for Key-Frame Extractor Module.
Validates fixed threshold T=340,000 and frame_skip=3 configuration loading,
static frame rejection, and motion key-frame retention logic.
"""

import pytest
import numpy as np
from src.config import MOTION_THRESHOLD, FRAME_SKIP
from src.preprocessing.keyframe_extractor import KeyFrameExtractor


def test_config_parameter_binding():
    """Verify that KeyFrameExtractor loads T=340,000 and frame_skip=3 from central config."""
    extractor = KeyFrameExtractor()
    assert extractor.threshold == 340000.0, f"Expected threshold 340000.0, got {extractor.threshold}"
    assert extractor.frame_skip == 3, f"Expected frame_skip 3, got {extractor.frame_skip}"
    assert MOTION_THRESHOLD == 340000
    assert FRAME_SKIP == 3


def test_initial_frame_retention():
    """First frame of a video should always be retained as baseline key-frame."""
    extractor = KeyFrameExtractor()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    is_key_frame, sum_diff, metadata = extractor.process_frame(frame, frame_idx=0, timestamp=0.0)
    assert is_key_frame is True
    assert sum_diff == float("inf")
    assert metadata["reason"] == "initial_baseline_frame"


def test_static_frame_rejection():
    """
    Subsequent identical/static frames produce SumDiff = 0 <= 340,000
    and MUST be discarded (is_key_frame == False).
    """
    extractor = KeyFrameExtractor()
    frame_static = np.ones((480, 640, 3), dtype=np.uint8) * 128

    # Frame 0: Baseline
    extractor.process_frame(frame_static, frame_idx=0, timestamp=0.0)

    # Frame 3 (frame_skip=3): Identical frame -> SumDiff = 0 <= 340,000
    is_key_frame, sum_diff, metadata = extractor.process_frame(frame_static, frame_idx=3, timestamp=0.1)
    assert is_key_frame is False
    assert sum_diff == 0.0
    assert sum_diff <= MOTION_THRESHOLD
    assert metadata["reason"] == "below_threshold_discarded"


def test_motion_frame_retention():
    """
    Significant motion frame producing SumDiff > 340,000
    MUST be retained (is_key_frame == True).
    """
    extractor = KeyFrameExtractor(threshold=340000)

    # Create Frame 1 (black)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    extractor.process_frame(frame1, frame_idx=0, timestamp=0.0)

    # Create Frame 2 with pixel difference sum > 340,000
    # For a 480x640 frame (307,200 pixels), a shift of 2 in pixel intensity across image yields diff 614,400 > 340,000
    frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 5

    is_key_frame, sum_diff, metadata = extractor.process_frame(frame2, frame_idx=3, timestamp=0.1)
    assert sum_diff > 340000.0, f"Expected SumDiff > 340000, got {sum_diff}"
    assert is_key_frame is True
    assert metadata["reason"] == "exceeds_fixed_threshold"
