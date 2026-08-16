"""
Generates resolved data.yaml with absolute path for Ultralytics YOLOv5.
"""

from pathlib import Path
import yaml

DATASET_ROOT = Path(r"c:\abnormality_in_online_exam\data\raw_public_dataset\Dataset_Students_Behavior_Online_Exam_Org\dataset_Students_Behavior_Online_Exam_org").resolve()

data_config = {
    "path": str(DATASET_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "nc": 5,
    "names": ['eye_movement', 'hand_move', 'mobile_use', 'side_watching', 'mouth_open']
}

output_yaml = DATASET_ROOT / "resolved_data.yaml"
with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"Created resolved_data.yaml at {output_yaml}")
with open(output_yaml, "r", encoding="utf-8") as f:
    print(f.read())
