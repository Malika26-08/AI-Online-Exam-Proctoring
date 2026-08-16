"""
Dataset Verification Script for YOLOv5 training.
"""

from pathlib import Path
import os

DATASET_ROOT = Path(r"data/raw_public_dataset/Dataset_Students_Behavior_Online_Exam_Org/dataset_Students_Behavior_Online_Exam_org")

def verify_dataset():
    print("=" * 60)
    print("YOLOv5 DATASET VERIFICATION REPORT")
    print("=" * 60)

    splits = ["train", "valid", "test"]
    split_counts = {}

    for split in splits:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[{split.upper()}] Directory missing: img_dir={img_dir.exists()}, lbl_dir={lbl_dir.exists()}")
            continue

        images = [f for f in img_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        labels = [f for f in lbl_dir.iterdir() if f.suffix.lower() == ".txt"]

        print(f"\nSplit: {split.upper()}")
        print(f"  • Image files found : {len(images)}")
        print(f"  • Label files found : {len(labels)}")

        corrupt_labels = 0
        invalid_class_ids = set()
        box_count = 0

        for lbl_file in labels:
            try:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        corrupt_labels += 1
                        continue
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls_id < 0 or cls_id > 4:
                        invalid_class_ids.add(cls_id)
                    box_count += 1
            except Exception as e:
                corrupt_labels += 1

        print(f"  • Total Bounding Boxes: {box_count}")
        print(f"  • Corrupt label lines : {corrupt_labels}")
        print(f"  • Invalid class IDs   : {invalid_class_ids if invalid_class_ids else 'None (All in range 0-4)'}")

        split_counts[split] = {"images": len(images), "labels": len(labels), "boxes": box_count}

    print("\n--- YAML and Classes file check ---")
    yaml_file = DATASET_ROOT / "data.yaml"
    classes_file = DATASET_ROOT / "classes.txt"
    print(f"  • data.yaml exists  : {yaml_file.exists()}")
    print(f"  • classes.txt exists: {classes_file.exists()}")

if __name__ == "__main__":
    verify_dataset()
