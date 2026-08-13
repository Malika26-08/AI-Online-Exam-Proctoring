"""
Unit Tests for Frame Preprocessing Module.
Tests Gaussian filtering, histogram equalization, and spatial resizing.
"""

import pytest
import numpy as np
import cv2
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.config import DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE


def test_gaussian_filtering():
    """Test Gaussian smoothing noise reduction on a noisy synthetic matrix."""
    preprocessor = FramePreprocessor()
    # Create synthetic frame with random noise
    np.random.seed(42)
    noisy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    filtered = preprocessor.apply_gaussian_filter(noisy_frame)
    assert filtered.shape == noisy_frame.shape
    assert filtered.dtype == np.uint8
    # Standard deviation of smoothed image should be lower than raw noisy image
    assert np.std(filtered) < np.std(noisy_frame)


def test_histogram_equalization():
    """Test Y-channel histogram contrast equalization."""
    preprocessor = FramePreprocessor()
    # Low-contrast dark image
    dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 30

    equalized = preprocessor.apply_histogram_equalization(dark_frame)
    assert equalized.shape == dark_frame.shape
    assert equalized.dtype == np.uint8


def test_image_resizing():
    """Test resizing to target CNN dimensions (224x224x3 and 299x299x3)."""
    preprocessor = FramePreprocessor(target_size=DEFAULT_IMAGE_SIZE)
    raw_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Test default 224x224
    resized_224 = preprocessor.resize(raw_frame)
    assert resized_224.shape == (224, 224, 3)

    # Test Inception 299x299
    resized_299 = preprocessor.resize(raw_frame, target_size=INCEPTION_IMAGE_SIZE)
    assert resized_299.shape == (299, 299, 3)


def test_complete_preprocessing_pipeline():
    """Test full sequence: Gaussian filter -> Histogram equalization -> Resize."""
    preprocessor = FramePreprocessor(target_size=(224, 224))
    raw_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    processed = preprocessor.preprocess(raw_frame)
    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.uint8
