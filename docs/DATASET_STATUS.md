# DATASET STATUS & INTEGRATION SPECIFICATION

> [!NOTE]
> **IMPLEMENTATION DATASET STATUS: AVAILABLE (INTEGRATED & VALIDATED)**
> The mentor-approved public dataset **"Students' Abnormal Behavior in Online Exam Dataset"** is verified and ready in the workspace.

---

## 📄 Dataset Distinction & Status Overview

| Dataset Attribute | Reference Paper Dataset | Our Implementation Dataset |
|---|---|---|
| **Dataset Title** | S_OCA (Student Online Cheating Activity) | Students' Abnormal Behavior in Online Exam Dataset |
| **Primary Citation** | Ramzan et al. (*IEEE Access, 2024*) | Approved Public Dataset (`Dataset_Students_Behavior_Online_Exam_Org`) |
| **Workspace Location** | Reference Paper Only (Not in workspace) | `C:\abnormality_in_online_exam\data\raw_public_dataset\Dataset_Students_Behavior_Online_Exam_Org\dataset_Students_Behavior_Online_Exam_org` |
| **Target Classes (5)** | `normal`, `external_device`, `head_movement`, `multiple_persons`, `talking_to_others` | `eye_movement`, `hand_move`, `mobile_use`, `side_watching`, `mouth_open` |
| **Total Images** | N/A (52 full videos in paper) | **5,575 Images** |
| **Total Bounding Box Annotations** | N/A | **5,511 Annotations** |
| **Workspace Status** | Reference Only | **INTEGRATED & VALIDATED (READY FOR TRAINING)** |

---

## 📊 Implementation Dataset Summary (Verified)

- **Total Images**: 5,575
  - `train`: 4,395 images (4,407 bounding box annotations)
  - `valid`: 1,099 images (1,104 bounding box annotations)
  - `test`: 81 images (unannotated test inference split)
- **Class Distribution (Bounding Box Annotations)**:
  1. `eye_movement`: 1,159 (21.03%)
  2. `hand_move`: 1,229 (22.30%)
  3. `mobile_use`: 1,043 (18.93%)
  4. `side_watching`: 1,179 (21.39%)
  5. `mouth_open`: 901 (16.35%)
- **Data Integrity Verification**:
  - Corrupted Images: **0** (100% valid readable image files)
  - Malformed Annotation Lines: **0** (100% valid YOLO format)
  - Duplicate Files: **5 duplicate images** identified between test split and train/valid splits; explicitly documented and excluded from independent test performance claims.

---

## 🔒 Policy & Integrity Constraints

1. **Dataset Name Distinction**: Experiments and implementation code strictly reference **Students' Abnormal Behavior in Online Exam Dataset**. S_OCA is documented purely as the reference paper dataset.
2. **NO Synthetic Data Fabrication**: No synthetic images or fabricated annotation files were generated.
3. **NO Training / Metric Fabrication**: Model training has not been executed yet. No accuracy, precision, recall, F1, or mAP metrics are claimed before training.
4. **Preserved Preprocessing Pipeline**: The pipeline preserves motion key-frame thresholding ($T=340,000$, `frame_skip=3`), Gaussian filtering, histogram equalization, and model-specific spatial resizing ($224 \times 224$ and $299 \times 299$).
