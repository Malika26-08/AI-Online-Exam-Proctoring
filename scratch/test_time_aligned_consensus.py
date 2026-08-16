"""
Diagnostic script to test time-aligned 2-of-4 model consensus on real video key-frames.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.config import (
    CLASS_NAMES, MOTION_THRESHOLD, FRAME_SKIP,
    CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.utils.video_loader import VideoLoader
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model

def run_time_aligned_consensus_test():
    temp_dir = Path(r"C:\Users\VIghnesh\AppData\Local\Temp")
    video_files = list(temp_dir.glob("tmp*.mp4"))
    if not video_files:
        print("No temp video found.")
        return
    video_path = max(video_files, key=lambda p: p.stat().st_mtime)

    loader = VideoLoader(video_path)
    meta = loader.metadata
    print(f"Loaded real video: {video_path.name} ({meta['frame_count']} frames, {meta['duration_sec']:.2f}s)")

    CNN_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]
    preprocessors = {}
    models = {}

    for m in CNN_MODELS:
        target_size = INCEPTION_IMAGE_SIZE if "inception" in m else DEFAULT_IMAGE_SIZE
        preprocessors[m] = FramePreprocessor(target_size=target_size)
        models[m] = build_model(m, load_best_checkpoint=True)
        if hasattr(models[m], "eval"):
            models[m].eval()

    keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)
    kf_predictions = []

    for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
        is_kf, _, _ = keyframe_extractor.process_frame(raw_frame, frame_idx, timestamp_sec)
        if is_kf:
            model_preds = {}
            for m in CNN_MODELS:
                proc = preprocessors[m].preprocess(raw_frame)
                tensor_img = torch.from_numpy(proc).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    logits = models[m](tensor_img)
                    probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

                top_idx = int(np.argmax(probs))
                pred_cls = CLASS_NAMES[top_idx]
                conf = float(probs[top_idx])
                model_preds[m] = (pred_cls, conf)
            kf_predictions.append((frame_idx, timestamp_sec, model_preds))

    print(f"Extracted Key-Frames: {len(kf_predictions)}")

    # Time-Aligned 2-of-4 Consensus Evaluation
    consensus_points = []
    for frame_idx, ts, m_preds in kf_predictions:
        class_groups = {}
        for m, (cls, conf) in m_preds.items():
            if cls != "normal" and conf >= 0.65:
                if cls not in class_groups:
                    class_groups[cls] = []
                class_groups[cls].append((m, conf))

        for cls, agreeing in class_groups.items():
            if len(agreeing) >= 2:
                m_names = [m.replace("_", " ").title() for m, _ in agreeing]
                confs = [c for _, c in agreeing]
                consensus_points.append({
                    "frame_idx": frame_idx,
                    "timestamp_sec": ts,
                    "predicted_class": cls,
                    "agreeing_models": m_names,
                    "num_agreeing": len(m_names),
                    "peak_confidence": max(confs),
                    "avg_confidence": float(np.mean(confs))
                })

    print(f"\n--- TIME-ALIGNED CONSENSUS KEY-FRAME POINTS ({len(consensus_points)} points) ---")
    for cp in consensus_points:
        print(f"  Frame {cp['frame_idx']:3d} ({cp['timestamp_sec']:5.2f}s): {cp['predicted_class']:<15} | Agreeing ({cp['num_agreeing']}): {cp['agreeing_models']}")

    # Group consecutive consensus points into Consensus Events (breaking on non-consensus, class change, or gap > 1.0s)
    events = []
    curr_event = []
    max_gap_sec = 1.0

    for p in consensus_points:
        if not curr_event:
            curr_event = [p]
        else:
            prev = curr_event[-1]
            if (p["predicted_class"] == prev["predicted_class"] and
                p["timestamp_sec"] <= prev["timestamp_sec"] + max_gap_sec):
                curr_event.append(p)
            else:
                events.append(curr_event)
                curr_event = [p]

    if curr_event:
        events.append(curr_event)

    print(f"\n=======================================================")
    print(f"TIME-ALIGNED CONSENSUS EVENTS ({len(events)} events)")
    print(f"=======================================================")
    if not events:
        print("No consensus abnormal activity detected (fewer than 2 models agreed on any abnormal frame).")
    else:
        for idx, ev in enumerate(events, 1):
            start_t = ev[0]["timestamp_sec"]
            end_t = ev[-1]["timestamp_sec"]
            dur = max(0.0, end_t - start_t)
            cls = ev[0]["predicted_class"]
            all_agreeing = list(dict.fromkeys(m for p in ev for m in p["agreeing_models"]))
            peak_c = max(p["peak_confidence"] for p in ev)
            avg_c = float(np.mean([p["avg_confidence"] for p in ev]))
            kf_cnt = len(set(p["frame_idx"] for p in ev))
            print(f" {idx:2d}. [{start_t:5.2f}s - {end_t:5.2f}s] ({dur:4.1f}s) Activity: {cls:<15} | Models: {all_agreeing} ({len(all_agreeing)}) | Peak: {peak_c*100:.1f}% | Avg: {avg_c*100:.1f}% | Key-Frames: {kf_cnt}")

if __name__ == "__main__":
    run_time_aligned_consensus_test()
