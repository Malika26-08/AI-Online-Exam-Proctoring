"""
Environment Validation Utility.
Inspects Python runtime, CUDA / GPU acceleration, OpenCV, and workspace directory structure.
"""

import sys
import os
import platform
from pathlib import Path
from src.config import (
    BASE_DIR, DATA_DIR, CLASS_NAMES, BENCHMARK_MODELS,
    FRAME_SKIP, MOTION_THRESHOLD
)
from src.utils.logger import get_logger

logger = get_logger("env_check")


def check_environment():
    """Runs non-destructive environment checks and logs environment metadata."""
    info = {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "workspace": str(BASE_DIR),
        "gpu_available": False,
        "gpu_name": "None",
        "cuda_version": "None",
        "opencv_installed": False,
        "torch_installed": False
    }

    # Check PyTorch & CUDA
    try:
        import torch
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        info["torch_installed"] = False

    # Check OpenCV
    try:
        import cv2
        info["opencv_installed"] = True
        info["opencv_version"] = cv2.__version__
    except ImportError:
        info["opencv_installed"] = False

    logger.info("=== ENVIRONMENT & WORKSPACE DIAGNOSTICS ===")
    logger.info(f"Workspace Directory : {info['workspace']}")
    logger.info(f"Python Runtime     : {info['python_version']}")
    logger.info(f"Operating System   : {info['os']}")
    logger.info(f"PyTorch Installed  : {info['torch_installed']} (Version: {info.get('torch_version', 'N/A')})")
    logger.info(f"GPU / CUDA Status  : GPU Available: {info['gpu_available']} | Device: {info['gpu_name']} | CUDA: {info['cuda_version']}")
    logger.info(f"OpenCV Installed   : {info['opencv_installed']} (Version: {info.get('opencv_version', 'N/A')})")
    logger.info(f"Fixed Threshold T  : {MOTION_THRESHOLD}")
    logger.info(f"Frame Skip Factor  : {FRAME_SKIP}")
    logger.info(f"Target Classes (5) : {', '.join(CLASS_NAMES)}")
    logger.info(f"Benchmark Models(5): {', '.join(BENCHMARK_MODELS)}")
    logger.info("==========================================")

    return info


if __name__ == "__main__":
    check_environment()
