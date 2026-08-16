"""
Diagnostic script to test YOLOv5 inference on demo_exam_session_[DEMO].mp4 and temp videos.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.config import MOTION_THRESHOLD, FRAME_SKIP
from src.utils.video_loader import VideoLoader
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.yolov5_detector import YOLOv5Detector

def diagnose_videos():
    yolo_detector = YOLOv5Detector()
    print(f"Loaded weights path: {yolo_detector.weights_path}")

    videos_to_test = []

    # Demo clip
    demo_video = Path("data/demo_clips/demo_exam_session_[DEMO].mp4")
    if demo_video.exists():
        videos_to_test.append(demo_video)

    # Temp videos
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    temp_files = sorted(list(temp_dir.glob("tmp*.mp4")), key=lambda p: p.stat().st_mtime, reverse=True)
    videos_to_test.extend(temp_files[:3])

    for vpath in videos_to_test:
        print("\n" + "=" * 60)
        print(f"TESTING VIDEO: {vpath.name}")
        print("=" * 60)

        loader = VideoLoader(vpath)
        meta = loader.metadata
        print(f"Frames: {meta['frame_count']}, Duration: {meta['duration_sec']:.2f}s")

        keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)
        kf_count = 0
        yolo_det_count = 0
        yolo_kf_with_dets = 0
        det_classes = {}

        for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
            is_kf, _, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
            if is_kf:
                kf_count += 1
                
                # Test 1: Predict on raw BGR frame
                dets_bgr = yolo_detector.predict_frame(raw_frame)
                
                # Test 2: Predict on RGB frame
                rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                dets_rgb = yolo_detector.predict_frame(rgb_frame)

                # Combine / log results
                dets = dets_rgb if dets_rgb else dets_bgr
                if dets:
                    yolo_kf_with_dets += 1
                    yolo_det_count += len(dets)
                    for d in dets:
                        cls = d["class_name"]
                        det_classes[cls] = det_classes.get(cls, 0) + 1

                if is_kf and kf_count <= 5:
                    print(f"  Key-Frame #{kf_count} (f_idx={frame_idx}): BGR dets={len(dets_bgr)}, RGB dets={len(dets_rgb)}")

        print(f"Summary: Key-Frames={kf_count}, Key-Frames with Dets={yolo_kf_with_dets}, Total Dets={yolo_det_count}")
        print(f"Class Breakdown: {det_classes}")

if __name__ == "__main__":
    diagnose_videos()
