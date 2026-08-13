# Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams

An AI-Based Online Exam Proctoring System reproducing the research methodology by **Ramzan et al. (IEEE Access, 2024)** as specified in `project_report.pdf`.

---

## 📌 Project Overview

This project implements an end-to-end video analytics pipeline designed to identify suspicious candidate behaviors during online examinations using webcam video streams.

### Core Workflow
1. **Video Feed Ingestion**: Accepts webcam stream or video clip (.mp4, .avi).
2. **Pre-processing**: Applies Gaussian noise filtering, histogram equalization, and spatial resizing ($224 \times 224 \times 3$).
3. **Motion Key-Frame Extraction**: Evaluates frame-to-frame pixel difference using a fixed motion difference threshold $T = 340,000$ and a frame skip factor of $3$ to eliminate redundant static frames.
4. **Model Inference**: Swappable classification head evaluating 5 benchmark models.
5. **Temporal Aggregation**: Sliding window voting to reduce false positives.
6. **Session Report / Timeline**: Real-time alert timeline and report output.

---

## 🎯 Benchmark Target Classes (5)
- `eye_movement`: Candidate gaze shift or abnormal eye motion.
- `hand_move`: Hand gestures or movement towards unauthorized areas.
- `mobile_use`: Holding, viewing, or using a mobile phone device.
- `side_watching`: Candidate head turned sideways towards notes/person.
- `mouth_open`: Candidate speaking or opening mouth repeatedly.

> **Note on Dataset Identification**:
> - Reference Paper Dataset: **S_OCA** (Student Online Cheating Activity, Ramzan et al. 2024; reference paper dataset).
> - Our Implementation Dataset: **Students' Abnormal Behavior in Online Exam Dataset** (Integrated & Validated in `data/raw_public_dataset/`).

---

## 🧠 Benchmark Models (5)
1. **DenseNet121** (ImageNet pre-trained backbone fine-tuned for 5 classes)
2. **InceptionV3** (ImageNet pre-trained backbone fine-tuned for 5 classes)
3. **Inception-ResNet-v2** (ImageNet pre-trained backbone fine-tuned for 5 classes)
4. **Custom CNN** (2-block convolutional neural network baseline trained from scratch)
5. **YOLOv5** (Ultralytics YOLOv5 fine-tuned for abnormal activity detection)

---

## 🚀 Environment Setup

### Prerequisites
- Operating System: Windows 10/11
- Hardware: GPU (NVIDIA GTX 1650 Ti or equivalent recommended)
- Python 3.10+

### Setup Instructions

1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify Environment**:
   ```powershell
   python -m src.utils.env_check
   ```

---

## 📁 Repository Layout

```
abnormality_in_online_exam/
├── project_report.pdf          # Primary source of truth document
├── README.md                   # Setup and execution guide
├── requirements.txt            # Python dependencies list
├── .gitignore                  # Git exclude rules
├── data/                       # Storage for raw, processed, and demo data
├── src/                        # Core methodology implementation
│   ├── config.py               # Central configuration (T=340000, skip=3, classes)
│   ├── utils/                  # Environment check & logging utilities
│   ├── preprocessing/          # Denoising, equalization, key-frame extraction
│   ├── models/                 # CNN architectures & YOLOv5 wrapper
│   ├── training/               # Model training scripts
│   ├── evaluation/             # Metrics calculation & plotting utilities
│   └── aggregation/            # Temporal sliding window & report builder
├── app/                        # Streamlit web demo application
├── tests/                      # Unit and integration test suite
└── weights/                    # Saved model weights (.pt, .pth)
```

---

## 🔒 Scope & Non-Fabrication Policy
- **Scope Limit**: Excludes facial recognition, biometric identity checks, audio proctoring, exam delivery, or grading.
- **Dataset Policy**: Dataset metrics will only be computed upon real model training. Reference paper figures remain strictly separated from actual project results.
