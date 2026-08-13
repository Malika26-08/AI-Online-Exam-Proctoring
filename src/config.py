"""
Central Configuration for Online Exam Abnormal Activity Detection.
Based on the methodology from Ramzan et al. (IEEE Access, 2024) and project_report.pdf.
"""

from pathlib import Path

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
RAW_PUBLIC_DATASET_DIR = DATA_DIR / "raw_public_dataset" / "Dataset_Students_Behavior_Online_Exam_Org" / "dataset_Students_Behavior_Online_Exam_org"
PROCESSED_FRAMES_DIR = DATA_DIR / "processed_frames"
DEMO_CLIPS_DIR = DATA_DIR / "demo_clips"
YOLOV5_DATA_DIR = DATA_DIR / "yolov5_data"

# Output Directories
WEIGHTS_DIR = BASE_DIR / "weights"
RESULTS_DIR = BASE_DIR / "results"

# Motion-Based Key-Frame Extraction Settings (Report / Paper Methodology)
FRAME_SKIP = 3
MOTION_THRESHOLD = 340000  # Sum of absolute pixel differences |dF|

# Image Input Dimensions
DEFAULT_IMAGE_SIZE = (224, 224)
INCEPTION_IMAGE_SIZE = (299, 299)
IMAGE_CHANNELS = 3

# Target Behavior Classes (Exact 5 classes from approved implementation public dataset)
# Reference paper (Ramzan et al., 2024) used S_OCA. Implementation uses approved Students' Behavior Online Exam dataset.
CLASS_NAMES = [
    "eye_movement",
    "hand_move",
    "mobile_use",
    "side_watching",
    "mouth_open"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Core Benchmark Model Architecture Identifiers
BENCHMARK_MODELS = [
    "densenet121",
    "inception_v3",
    "inception_resnet_v2",
    "custom_cnn",
    "yolov5"
]

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Training Default Hyperparameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 15

# Sliding Window Aggregation Settings
SLIDING_WINDOW_SECONDS = 3.0
CONFIDENCE_THRESHOLD = 0.65
