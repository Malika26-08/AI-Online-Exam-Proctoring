import os
import sys
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np

DATASET_ROOT = Path(r"C:\abnormality_in_online_exam\data\raw_public_dataset\Dataset_Students_Behavior_Online_Exam_Org\dataset_Students_Behavior_Online_Exam_org")

EXPECTED_CLASSES = ["eye_movement", "hand_move", "mobile_use", "side_watching", "mouth_open"]

def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_validation():
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Exists: {DATASET_ROOT.exists()}")
    
    classes_file = DATASET_ROOT / "classes.txt"
    with open(classes_file, "r", encoding="utf-8") as f:
        classes_read = [line.strip() for line in f if line.strip()]
    print(f"Classes in classes.txt: {classes_read}")
    
    stats = {}
    all_hashes = {}
    duplicates = []
    corrupted_images = []
    malformed_labels = []
    
    overall_class_counts = {idx: 0 for idx in range(len(EXPECTED_CLASSES))}
    
    for split in ["train", "valid", "test"]:
        split_dir = DATASET_ROOT / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        img_files = sorted(list(img_dir.glob("*.*"))) if img_dir.exists() else []
        lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []
        
        img_stems = {f.stem: f for f in img_files}
        lbl_stems = {f.stem: f for f in lbl_files}
        
        # Check corrupted images & hashes
        valid_img_count = 0
        for img_path in img_files:
            # Check Hash
            h = compute_file_hash(img_path)
            if h in all_hashes:
                duplicates.append((str(img_path), str(all_hashes[h])))
            else:
                all_hashes[h] = img_path
                
            # Check readability
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted_images.append(str(img_path))
            else:
                valid_img_count += 1
                
        # Match image-label correspondence
        unpaired_imgs = [s for s in img_stems if s not in lbl_stems]
        unpaired_lbls = [s for s in lbl_stems if s not in img_stems]
        
        # Count annotations and class distribution
        split_class_counts = {idx: 0 for idx in range(len(EXPECTED_CLASSES))}
        total_boxes = 0
        empty_label_files = 0
        
        for lbl_path in lbl_files:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if not lines or all(not l.strip() for l in lines):
                empty_label_files += 1
                
            for line_idx, line in enumerate(lines):
                line_str = line.strip()
                if not line_str:
                    continue
                parts = line_str.split()
                if len(parts) != 5:
                    malformed_labels.append((str(lbl_path), line_idx, f"Expected 5 tokens, got {len(parts)}: '{line_str}'"))
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    if cls_id < 0 or cls_id >= len(EXPECTED_CLASSES):
                        malformed_labels.append((str(lbl_path), line_idx, f"Class ID {cls_id} out of range [0, {len(EXPECTED_CLASSES)-1}]"))
                    else:
                        split_class_counts[cls_id] += 1
                        overall_class_counts[cls_id] += 1
                        total_boxes += 1
                except Exception as e:
                    malformed_labels.append((str(lbl_path), line_idx, str(e)))
                    
        stats[split] = {
            "image_count": len(img_files),
            "label_count": len(lbl_files),
            "valid_images": valid_img_count,
            "corrupted_images": len([x for x in corrupted_images if split in x]),
            "unpaired_images_count": len(unpaired_imgs),
            "unpaired_labels_count": len(unpaired_lbls),
            "empty_label_files": empty_label_files,
            "total_bounding_boxes": total_boxes,
            "class_distribution_boxes": {EXPECTED_CLASSES[i]: split_class_counts[i] for i in range(len(EXPECTED_CLASSES))}
        }

    summary = {
        "dataset_path": str(DATASET_ROOT),
        "detected_classes": classes_read,
        "splits_summary": stats,
        "overall_class_distribution_boxes": {EXPECTED_CLASSES[i]: overall_class_counts[i] for i in range(len(EXPECTED_CLASSES))},
        "total_images_all_splits": sum(s["image_count"] for s in stats.values()),
        "total_annotations_all_splits": sum(s["total_bounding_boxes"] for s in stats.values()),
        "corrupted_images_total": len(corrupted_images),
        "corrupted_images_details": corrupted_images,
        "duplicate_files_total": len(duplicates),
        "duplicate_files_details": duplicates[:10], # sample
        "malformed_labels_total": len(malformed_labels),
        "malformed_labels_details": malformed_labels[:10]
    }
    
    print("\n================ DETAILED SUMMARY ================")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_validation()
