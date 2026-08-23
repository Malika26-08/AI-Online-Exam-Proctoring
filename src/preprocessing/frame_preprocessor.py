"""
Frame Pre-processing Module for Online Exam Proctoring Pipeline.
Applies Gaussian noise filtering, histogram equalization/contrast normalization, and spatial resizing.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

from typing import Tuple, Optional, Union, Dict, Any
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
        enable_histogram_eq: bool = True,
        output_size: Optional[Tuple[int, int]] = None
    ):
        if output_size is not None:
            target_size = output_size
        self.target_size = target_size
        self.output_size = target_size
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

        t_size = target_size if target_size is not None else self.target_size
        return cv2.resize(frame, t_size, interpolation=cv2.INTER_AREA)

    def preprocess(
        self,
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Executes complete pre-processing sequence:
        1. Gaussian Filtering
        2. Histogram Equalization (Contrast Normalization)
        3. Spatial Resizing
        4. Min-Max Intensity Normalization [0, 1] (in dict mode)

        By default returns resized np.ndarray (shape HxWxC, uint8) to satisfy unit tests and dataloaders.
        If return_dict=True, returns dictionary containing intermediate arrays and normalized tensor.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid empty frame passed to preprocess.")

        t_size = target_size if target_size is not None else self.target_size

        filtered = self.apply_gaussian_filter(frame)
        if self.enable_histogram_eq:
            equalized = self.apply_histogram_equalization(filtered)
        else:
            equalized = filtered

        resized = self.resize(equalized, target_size=t_size)

        if return_dict:
            normalized = resized.astype(np.float32) / 255.0
            return {
                "raw_image": frame,
                "filtered_image": filtered,
                "equalized_image": equalized,
                "resized_image": resized,
                "normalized_image": normalized
            }

        return resized
