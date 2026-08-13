"""
Video Loader and Validation Utility for Online Exam Proctoring Pipeline.
Handles video ingestion, format validation, property extraction, and frame iteration.
"""

from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional
import cv2
import numpy as np
from src.config import FRAME_SKIP
from src.utils.logger import get_logger

logger = get_logger("video_loader")

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoValidationError(Exception):
    """Custom exception raised when video validation fails."""
    pass


class VideoLoader:
    """Wraps OpenCV VideoCapture to validate videos and yield frames with frame-skip logic."""

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.metadata = self.validate_and_get_info()

    def validate_and_get_info(self) -> Dict[str, Any]:
        """Validates video container format, file existence, and readable frames."""
        if not self.video_path.exists():
            raise VideoValidationError(f"Video file does not exist: {self.video_path}")

        if not self.video_path.is_file():
            raise VideoValidationError(f"Provided path is not a file: {self.video_path}")

        if self.video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise VideoValidationError(
                f"Unsupported video extension '{self.video_path.suffix}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            cap.release()
            raise VideoValidationError(f"Failed to open video container: {self.video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise VideoValidationError(
                f"Corrupt video metadata for {self.video_path.name}: "
                f"frames={frame_count}, width={width}, height={height}"
            )

        # Attempt reading the first frame to verify readability
        ret, first_frame = cap.read()
        cap.release()

        if not ret or first_frame is None:
            raise VideoValidationError(f"Video opened but first frame could not be read: {self.video_path}")

        duration_sec = frame_count / fps if fps > 0 else 0.0

        info = {
            "file_name": self.video_path.name,
            "file_path": str(self.video_path),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": duration_sec
        }
        logger.info(
            f"Validated video '{self.video_path.name}': "
            f"{width}x{height} @ {fps:.1f} FPS, {frame_count} frames, {duration_sec:.2f}s"
        )
        return info

    def read_frames(self, frame_skip: int = FRAME_SKIP) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_index, timestamp_seconds, frame_bgr) for every `frame_skip`-th frame.
        Guarantees exact frame-level index tracking.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise VideoValidationError(f"Cannot open video for reading: {self.video_path}")

        frame_idx = 0
        fps = self.metadata.get("fps", 30.0)
        fps = fps if fps > 0 else 30.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_skip == 0:
                    timestamp = frame_idx / fps
                    yield (frame_idx, timestamp, frame)

                frame_idx += 1
        finally:
            cap.release()
            logger.debug(f"Closed video reader for '{self.video_path.name}'. Processed {frame_idx} frames total.")
