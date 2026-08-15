"""
Generate Synthetic Demo Video Clip for UI & Pipeline Verification Only.
File created: data/demo_clips/demo_exam_session_[DEMO].mp4
STRICT RULE: This clip is strictly for UI/pipeline functional testing and MUST NOT be used for research metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from src.config import DEMO_CLIPS_DIR

DEMO_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
output_path = DEMO_CLIPS_DIR / "demo_exam_session_[DEMO].mp4"


def create_demo_clip(path: Path, num_frames: int = 150, fps: float = 30.0):
    """
    Generates a 5-second (150 frames @ 30fps) synthetic video clip with alternating motion
    and text watermark clearly identifying it as [DEMO] synthetic test data.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = 480, 640
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for idx in range(num_frames):
        # Create background with motion shifts every 15 frames to trigger motion threshold T=340,000
        phase = (idx // 15) % 3
        if phase == 0:
            bg_color = (40, 40, 40)       # Dark background
        elif phase == 1:
            bg_color = (180, 180, 180)   # Light background (triggers motion difference sum > 340,000)
        else:
            bg_color = (80, 120, 80)     # Muted green background

        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Draw watermark text
        cv2.putText(frame, "[DEMO] SYNTHETIC TEST CLIP", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "For Pipeline & Streamlit UI Verification Only", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {idx + 1}/{num_frames} | Time: {(idx / fps):.2f}s", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw a moving shape (circle)
        cx = int((width / 2) + 150 * np.sin(2 * np.pi * idx / 30))
        cy = int(height / 2)
        cv2.circle(frame, (cx, cy), 35, (0, 165, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"Successfully generated synthetic demo clip at: {path}")


if __name__ == "__main__":
    create_demo_clip(output_path)
