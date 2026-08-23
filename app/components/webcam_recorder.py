"""
Webcam Video Recorder — Streamlit Custom Component.

Uses st.components.v1.declare_component() to embed a browser-side
MediaRecorder HTML5 component (webcam_component/index.html).

The component records webcam + microphone + optional screen share and
returns the recorded video as a Base64 data URI when the candidate
clicks End Exam. Python then transcodes it to MP4 for the AI pipeline.

Wire-protocol fix applied in index.html:
  - isStreamlitMessage:true is added to every postMessage (required by
    Streamlit 1.61's onMessageEvent filter in index.bE3scgDe.js).
  - setComponentReady() is called synchronously, not inside window.load,
    so it always fires before the 15-second Streamlit timeout.

Transcoding strategy:
  Browser MediaRecorder always produces VP8/Vorbis WebM.
  OpenCV's cv2.VideoCapture cannot decode VP8 WebM on Windows without
  additional system codecs. Instead we use imageio-ffmpeg (which ships
  its own static FFmpeg 7.x binary) to do the WebM -> H.264 MP4 remux.
  OpenCV frame-by-frame re-encoding is used as a secondary fallback.
"""

import base64
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import streamlit.components.v1 as components

from src.utils.logger import get_logger

logger = get_logger("webcam_recorder")

# ---------------------------------------------------------------------------
# Component registration
# ---------------------------------------------------------------------------
COMPONENT_DIR = Path(__file__).parent / "webcam_component"

_webcam_component = components.declare_component(
    "webcam_recorder_component",
    path=str(COMPONENT_DIR),
)


# ---------------------------------------------------------------------------
# Internal helper — get FFmpeg binary
# ---------------------------------------------------------------------------

def _get_ffmpeg_bin() -> Optional[str]:
    """
    Return the path to the FFmpeg executable.

    Priority:
    1. imageio-ffmpeg bundled static binary (always available after install).
    2. System 'ffmpeg' on PATH.
    Returns None if neither is found.
    """
    try:
        import imageio_ffmpeg  # type: ignore
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(exe).exists():
            return exe
    except ImportError:
        pass

    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    return None


def _ffmpeg_convert(src: Path, dst: Path) -> bool:
    """
    Use FFmpeg to remux/transcode *src* into an H.264 MP4 at *dst*.

    Tries with audio first; retries without audio if audio codec fails.
    Returns True on success, False on failure.
    """
    ffmpeg_bin = _get_ffmpeg_bin()
    if not ffmpeg_bin:
        return False

    base_cmd = [ffmpeg_bin, "-y", "-i", str(src)]
    video_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                  "-movflags", "+faststart"]

    # Attempt 1: with audio
    cmd = base_cmd + video_args + ["-c:a", "aac", str(dst)]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        logger.info(
            "FFmpeg converted %s -> %s (%d bytes)",
            src.name, dst.name, dst.stat().st_size,
        )
        return True

    logger.warning(
        "FFmpeg with audio failed (rc=%d). Retrying without audio.",
        result.returncode,
    )

    # Attempt 2: without audio
    cmd_na = base_cmd + video_args + ["-an", str(dst)]
    result2 = subprocess.run(cmd_na, capture_output=True)
    if result2.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        logger.info(
            "FFmpeg (no audio) converted %s -> %s (%d bytes)",
            src.name, dst.name, dst.stat().st_size,
        )
        return True

    logger.error(
        "FFmpeg conversion failed entirely.\nSTDERR: %s",
        result2.stderr.decode(errors="replace")[-500:],
    )
    return False


def _opencv_convert(src: Path, dst: Path) -> bool:
    """
    Frame-by-frame OpenCV re-encoder fallback.
    Works when the container is readable by cv2.VideoCapture (e.g. MSMF
    on Windows can sometimes decode VP8 with system codecs installed).
    Returns True on success, False on failure.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS)
    fps    = fps if (fps and 1 <= fps <= 120) else 15.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()

    if n > 0 and dst.exists() and dst.stat().st_size > 0:
        logger.info("OpenCV re-encoded %s -> %s (%d frames)", src.name, dst.name, n)
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_webcam_recorder(key: str = "webcam_rec") -> Optional[str]:
    """
    Render the browser webcam recorder component.

    Returns
    -------
    str or None
        Raw Base64 string or JSON dual-stream payload string when End Exam is clicked,
        or None while idle / recording.
    """
    result = _webcam_component(key=key, default=None)
    if isinstance(result, str) and len(result) > 200:
        return result
    return None


def process_recorder_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Parses component payload (raw Base64 data URI or JSON dual-stream dict).
    Transcodes webcam WebM -> MP4 and screen WebM -> MP4 via FFmpeg.

    Returns
    -------
    dict or None
        Dict containing:
          - 'webcam_mp4_path': Path
          - 'screen_mp4_path': Optional[Path]
          - 'screen_status': str  ("Completed", "Interrupted", or "Not Captured")
          - 'screen_events': list
    """
    import json
    if not isinstance(payload, str) or len(payload) < 200:
        return None

    webcam_b64 = None
    screen_b64 = None
    screen_status = "Completed"
    screen_events = []

    if payload.startswith("{") and "webcam" in payload:
        try:
            data = json.loads(payload)
            webcam_b64 = data.get("webcam")
            screen_b64 = data.get("screen")
            screen_status = data.get("screen_status", "Completed")
            screen_events = data.get("screen_events", [])
        except Exception as exc:
            logger.warning(f"JSON parsing recorder payload failed: {exc}. Fallback to raw string.")
            webcam_b64 = payload
    else:
        webcam_b64 = payload

    if not webcam_b64 or not isinstance(webcam_b64, str):
        return None

    webcam_mp4 = transcode_webcam_data_to_mp4(webcam_b64)
    screen_mp4 = None
    if screen_b64 and isinstance(screen_b64, str) and len(screen_b64) > 200:
        try:
            screen_mp4 = transcode_webcam_data_to_mp4(screen_b64)
        except Exception as exc:
            logger.warning(f"Screen WebM -> MP4 transcoding warning: {exc}")

    return {
        "webcam_mp4_path": webcam_mp4,
        "screen_mp4_path": screen_mp4,
        "screen_status": screen_status,
        "screen_events": screen_events
    }


def transcode_webcam_data_to_mp4(base64_data: str) -> Path:
    """
    Convert the Base64 WebM returned by the browser recorder to a valid
    ``.mp4`` file that OpenCV / VideoLoader can open.

    Strategy
    --------
    1. Robustly clean base64 data: strip whitespace, remove data-URL prefix,
       remove internal newlines/spaces, restore missing Base64 '=' padding.
    2. Decode base64 -> raw bytes -> save as temporary ``.webm``.
    3. Try FFmpeg (imageio-ffmpeg bundled binary): WebM -> H.264 MP4.
    4. Fallback: OpenCV frame-by-frame re-encode (works if system codecs
       can decode the WebM).
    5. If both fail, raise RuntimeError (never silently pass a .webm path
       to VideoLoader, which would raise VideoValidationError).

    Returns
    -------
    Path
        Absolute path to a ``.mp4`` file (never ``.webm``).

    Raises
    ------
    TypeError
        If *base64_data* is not a string.
    ValueError
        If *base64_data* is empty, corrupted, or cannot be Base64 decoded.
    RuntimeError
        If all transcoding methods fail and no valid MP4 can be produced.
    """
    if not isinstance(base64_data, str):
        raise TypeError(
            f"base64_data must be str, got {type(base64_data).__name__}"
        )

    # 1. Strip leading and trailing whitespace
    data = base64_data.strip()

    # 2. Remove data-URL prefix if present
    if ";base64," in data:
        data = data.split(";base64,", 1)[1].strip()
    elif "base64," in data:
        data = data.split("base64,", 1)[1].strip()
    elif "," in data:
        data = data.rsplit(",", 1)[1].strip()

    # 3. Strip any internal whitespace / newlines
    data = "".join(data.split())

    if not data or len(data) < 50:
        raise ValueError("Invalid Base64 video payload (empty or too short).")

    # 4. Restore missing Base64 '=' padding
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)

    # 5. Base64 decode
    try:
        raw_bytes = base64.b64decode(data, validate=False)
    except Exception as exc:
        raise ValueError(f"Base64 decoding failed: {exc}") from exc

    if not raw_bytes or len(raw_bytes) < 100:
        raise ValueError(
            f"Decoded video blob is empty or too small ({len(raw_bytes)} bytes)."
        )

    uid = uuid.uuid4().hex[:8]
    tmp_dir = Path(tempfile.gettempdir())
    webm_path = tmp_dir / f"exam_webcam_{uid}.webm"
    mp4_path  = tmp_dir / f"exam_webcam_{uid}.mp4"

    webm_path.write_bytes(raw_bytes)
    logger.info(
        "Saved raw WebM blob: %s (%d bytes)",
        webm_path.name, len(raw_bytes),
    )

    # --- Primary: FFmpeg via imageio-ffmpeg ---
    if _ffmpeg_convert(webm_path, mp4_path):
        try:
            webm_path.unlink(missing_ok=True)
        except Exception:
            pass
        return mp4_path

    # --- Fallback: OpenCV frame-by-frame ---
    logger.warning(
        "FFmpeg conversion failed. Trying OpenCV frame-by-frame re-encoder."
    )
    if _opencv_convert(webm_path, mp4_path):
        try:
            webm_path.unlink(missing_ok=True)
        except Exception:
            pass
        return mp4_path

    # Both failed — never return a .webm path to VideoLoader
    raise RuntimeError(
        "WebM -> MP4 transcoding failed with both FFmpeg and OpenCV. "
        f"Source WebM: {webm_path} ({len(raw_bytes)} bytes). "
        "Check that imageio-ffmpeg is installed and FFmpeg codecs are available."
    )


def transcode_webcam_data_to_mp4_from_file(input_path: Path) -> Path:
    """
    Transcode an existing video file (e.g. an uploaded .webm) to MP4.

    If the input is already ``.mp4``, returns it unchanged.
    Uses FFmpeg (via imageio-ffmpeg) as primary, OpenCV as fallback.

    Raises
    ------
    FileNotFoundError
        If *input_path* does not exist.
    RuntimeError
        If all transcoding methods fail.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.suffix.lower() == ".mp4":
        return input_path

    uid = uuid.uuid4().hex[:8]
    mp4_path = Path(tempfile.gettempdir()) / f"exam_upload_{uid}.mp4"

    if _ffmpeg_convert(input_path, mp4_path):
        return mp4_path

    if _opencv_convert(input_path, mp4_path):
        return mp4_path

    raise RuntimeError(
        f"Transcoding {input_path.name} -> MP4 failed with both FFmpeg and OpenCV."
    )
