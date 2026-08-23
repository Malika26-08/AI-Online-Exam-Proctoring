"""
Screen Recording & Screen Monitoring Analysis Module.
Provides extension point for continuous candidate screen capture analysis.
Per methodology, candidate behavior (eye movement, mobile use, etc.) is evaluated
on webcam feed. Screen capture is independently recorded and monitored for session events.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from src.utils.video_loader import VideoLoader
from src.utils.logger import get_logger

logger = get_logger("screen_analyzer")


def analyze_screen_recording(
    screen_mp4_path: Optional[Path] = None,
    screen_status: str = "Completed",
    screen_events: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Analyzes the recorded candidate screen monitoring stream.

    Parameters
    ----------
    screen_mp4_path : Path or None
        Path to the transcoded screen recording MP4.
    screen_status : str
        Status string ("Completed", "Interrupted", or "Not Captured").
    screen_events : list of dicts or None
        Log of screen monitoring events during examination.

    Returns
    -------
    dict
        Screen analysis summary dictionary for inclusion in Step 7 report.
    """
    events_list = screen_events or [
        {"timestamp": "00:00", "event": "Screen Sharing Started", "status": "Normal"}
    ]

    if not screen_mp4_path or not Path(screen_mp4_path).exists():
        logger.info("Screen recording not available or not captured.")
        return {
            "captured": False,
            "status": screen_status,
            "resolution": "N/A",
            "duration_sec": 0.0,
            "events": events_list,
            "note": "Screen recording not captured or skipped during examination."
        }

    screen_path = Path(screen_mp4_path)
    meta = {}
    try:
        loader = VideoLoader(screen_path)
        meta = loader.metadata
    except Exception as exc:
        logger.warning(f"Could not read metadata for screen recording '{screen_path.name}': {exc}")

    resolution_str = f"{meta.get('width', 0)}x{meta.get('height', 0)}" if meta else "N/A"
    duration_sec = meta.get("duration_sec", 0.0) if meta else 0.0

    if screen_status == "Interrupted":
        events_list.append({
            "timestamp": f"{duration_sec:.1f}s",
            "event": "Screen Sharing Interrupted by Candidate",
            "status": "Warning"
        })

    logger.info(
        "Analyzed screen recording '%s': %s, duration=%.1fs, status=%s",
        screen_path.name, resolution_str, duration_sec, screen_status
    )

    return {
        "captured": True,
        "status": screen_status,
        "resolution": resolution_str,
        "duration_sec": duration_sec,
        "events": events_list,
        "note": "Screen recording captured successfully. Advanced screen-content classification is not enabled."
    }
