# Implementation Dataset & Technical Readiness Specification

This document details the dataset requirements, privacy compliance, technical readiness schema, and verification report for the **Students' Abnormal Behavior in Online Exam Dataset**, alongside the reference paper **Student Online Cheating Activity (S_OCA)** dataset (*Ramzan et al., IEEE Access, 2024*).

---

## 1. Reference Paper Dataset Specification vs Implementation Dataset

- **Primary Citation**: Ramzan, M., Abid, A., Bilal, M., Aamir, K. M., Memon, S. A., & Chung, T.-S. (2024). *"Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams."* IEEE Access, Vol. 12, pp. 21503–21519.
- **Reference Paper Dataset**: S_OCA (Student Online Cheating Activity) Dataset (52 webcam videos; reference paper dataset only).
- **Implementation Dataset**: **Students' Abnormal Behavior in Online Exam Dataset** (Approved Public Dataset available in `data/raw_public_dataset/Dataset_Students_Behavior_Online_Exam_Org/dataset_Students_Behavior_Online_Exam_org`).

---

## 2. Implementation Target Behavioral Classes (5 Classes)

| Class Name | Class ID | Description in Implementation Dataset | Bounding Boxes (Train+Valid) | Percentage |
|---|---|---|---|---|
| `eye_movement` | `0` | Candidate gaze shift or abnormal eye motion. | 1,159 | 21.03% |
| `hand_move` | `1` | Hand gestures or movement towards unauthorized areas. | 1,229 | 22.30% |
| `mobile_use` | `2` | Holding, viewing, or using a mobile phone device. | 1,043 | 18.93% |
| `side_watching` | `3` | Candidate head turned sideways towards notes/person. | 1,179 | 21.39% |
| `mouth_open` | `4` | Candidate speaking or opening mouth repeatedly. | 901 | 16.35% |

---

## 3. Data State Comparison Matrix

| Aspect | Reference Paper (Ramzan et al. 2024) | Current Implementation State | Status |
|---|---|---|---|
| **Dataset Title** | S_OCA Dataset | Students' Behavior Online Exam Dataset | **INTEGRATED** |
| **Dataset Path** | N/A (Internal paper) | `data/raw_public_dataset/.../dataset_Students_Behavior_Online_Exam_org` | **AVAILABLE** |
| **Split Structure** | 70/30 Train/Test Ratio | `train` (4395), `valid` (1099), `test` (81) | **VERIFIED** |
| **Target Classes** | 5 S_OCA Classes | 5 Public Dataset Classes | **CONFIGURED** |
| **Key-Frame Selection** | Motion Threshold $T=340,000$, Skip=3 | `KeyFrameExtractor` Preserved | **VERIFIED** |
| **YOLO Annotations** | Class Bounding Box Coordinates | 5,511 YOLO Annotations Verified | **VERIFIED** |
| **Training Status** | Evaluated in paper | Training NOT started yet | **READY FOR TRAINING** |

---

## 4. Quality Control & Duplicate Documentation

### Quality Control Results
1. **Readable Image Check**: 100% of 5,575 images are valid and uncorrupted.
2. **Annotation Format Check**: 100% of 5,511 bounding boxes follow standard YOLO format (`<class_id> <x_center> <y_center> <width> <height>`) with class IDs in $[0..4]$.
3. **Duplicate Documentation**:
   - Exactly 5 duplicate images were detected in the `test/images/` split that match images present in `train/images/` (3) or `valid/images/` (2).
   - **Handling Rule**: These 5 duplicate images are explicitly flagged in `dataset_readiness_summary.json` and `dataset_manifest.json`, and will NOT be used to report independent test set performance.
