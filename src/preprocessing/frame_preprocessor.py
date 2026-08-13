"""
Frame Pre-processing Module for Online Exam Proctoring Pipeline.
Applies Gaussian noise filtering, histogram equalization/contrast normalization, and spatial resizing.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Tuple
import cv2
import numpy as np
from src.config import DEFAULT_IMAGE_SIZE
from src.utils.logger import get_logger

logger = get_logger("frame_preprocessor")


class FramePreprocessor:
    """Applies noise reduction, histogram equalization, and image resizing to video frames."""

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        gaussian_kernel: Tuple[int, int] = (5, 5),
        gaussian_sigma: float = 1.0,
        enable_histogram_eq: bool = True
    ):
        self.target_size = target_size
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
        self.enable_histogram_eq = enable_histogram_eq

    def apply_gaussian_filter(self, frame: np.ndarray) -> np.ndarray:
        """Applies Gaussian smoothing to eliminate high-frequency video sensor noise."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid empty frame passed to Gaussian filter.")
        return cv2.GaussianBlur(frame, self.gaussian_kernel, self.gaussian_sigma)

    def apply_histogram_equalization(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies histogram equalization to normalize contrast.
        Converts BGR frame to YCrCb space, equalizes the Y (luminance) channel,
        and converts back to BGR to preserve color balance.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid empty frame passed to histogram equalization.")

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            equalized_bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            return equalized_bgr
        elif len(frame.shape) == 2:
            return cv2.equalizeHist(frame)
        else:
            return frame

    def resize(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resizes frame to downstream model input dimensions (e.g. 224x224x3 or 299x299x3)."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid empty frame passed to resize.")

        size = target_size if target_size is not None else self.target_size
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def preprocess(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Executes complete pre-processing sequence:
        Raw Frame -> Gaussian Noise Filtering -> Histogram Normalization -> Resizing.
        """
        denoised = self.apply_gaussian_filter(frame)
        if self.enable_histogram_eq:
            normalized = self.apply_histogram_equalization(denoised)
        else:
            normalized = denoised

        final_frame = self.resize(normalized, target_size=target_size)
        return final_frame
