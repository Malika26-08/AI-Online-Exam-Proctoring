"""
Diagnostic script to test trained YOLOv5 checkpoint on real webcam video.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.config import MOTION_THRESHOLD, FRAME_SKIP
from src.utils.video_loader import VideoLoader
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.yolov5_detector import YOLOv5Detector

def test_real_video_yolo():
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    video_files = list(temp_dir.glob("tmp*.mp4"))
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)

    loader = VideoLoader(video_path)
    meta = loader.metadata

    keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)
    yolo_detector = YOLOv5Detector()  # Loads weights/yolov5_best.pt automatically

    print("=" * 60)
    print("YOLOV5 REAL VIDEO INFERENCE DIAGNOSTIC")
    print("=" * 60)
    print(f"Video File: {video_path.name}")
    print(f"Duration  : {meta['duration_sec']:.2f}s ({meta['frame_count']} total frames)")
    print(f"Loaded Weights: {yolo_detector.weights_path}")

    key_frames_analyzed = 0
    frames_with_detections = 0
    total_detections_count = 0
    class_detections = {}
    all_detections_list = []

    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        is_kf, _, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
        if is_kf:
            key_frames_analyzed += 1
            dets = yolo_detector.predict_frame(raw_frame)
            if dets:
                frames_with_detections += 1
                total_detections_count += len(dets)
                for d in dets:
                    cls_name = d["class_name"]
                    class_detections[cls_name] = class_detections.get(cls_name, 0) + 1
                    all_detections_list.append({
                        "frame_idx": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "class_name": cls_name,
                        "confidence": d["confidence"],
                        "bbox": d["bbox"]
                    })

    print(f"\nExtracted Key-Frames       : {key_frames_analyzed}")
    print(f"Key-Frames with Detections : {frames_with_detections}")
    print(f"Total Bounding Box Count   : {total_detections_count}")

    print("\n--- DETECTED CLASSES BREAKDOWN ---")
    for cls_name, count in class_detections.items():
        print(f"  • {cls_name:<20}: {count} bounding boxes")

    print("\n--- SAMPLE DETECTIONS (FIRST 10) ---")
    for item in all_detections_list[:10]:
        bbox_str = f"[{item['bbox'][0]:.1f}, {item['bbox'][1]:.1f}, {item['bbox'][2]:.1f}, {item['bbox'][3]:.1f}]"
        print(f"  • Frame #{item['frame_idx']} ({item['timestamp_sec']:.2f}s) - {item['class_name']} ({item['confidence']*100:.1f}%) | bbox: {bbox_str}")

if __name__ == "__main__":
    test_real_video_yolo()
