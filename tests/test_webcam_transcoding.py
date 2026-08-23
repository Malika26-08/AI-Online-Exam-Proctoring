"""
Integration tests for the webcam recording transcoding pipeline.

Validates the full data flow:
    Browser Base64 WebM  ->  transcode_webcam_data_to_mp4()
    ->  .mp4 file  ->  VideoLoader accepts it  ->  OpenCV reads frames.

Also validates transcode_webcam_data_to_mp4_from_file() for uploaded WebM files.

These tests use imageio-ffmpeg to synthesise a realistic VP8/Vorbis WebM
(the same codec the browser MediaRecorder produces on Chrome/Firefox) so
they exercise the real transcoding code path, not a synthetic edge case.
"""

import base64
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.components.webcam_recorder import (
    transcode_webcam_data_to_mp4,
    transcode_webcam_data_to_mp4_from_file,
    _get_ffmpeg_bin,
)
from src.utils.video_loader import VideoLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_bin():
    """Return the FFmpeg binary path; skip test if unavailable."""
    exe = _get_ffmpeg_bin()
    if exe is None:
        pytest.skip("FFmpeg binary not available (imageio-ffmpeg not installed)")
    return exe


def _make_vp8_webm(frames: int = 30, fps: float = 10.0,
                   width: int = 320, height: int = 240) -> Path:
    """
    Create a temporary VP8/Vorbis WebM file that mimics browser MediaRecorder output.
    Uses FFmpeg to do the encoding.
    """
    ffmpeg = _ffmpeg_bin()

    # 1. Create a source MP4 (OpenCV can write this)
    src_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    src_tmp.close()
    src_path = Path(src_tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(src_path), fourcc, fps, (width, height))
    for _ in range(frames):
        writer.write(np.random.randint(0, 200, (height, width, 3), dtype=np.uint8))
    writer.release()

    # 2. Convert to VP8 WebM with FFmpeg
    webm_tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    webm_tmp.close()
    webm_path = Path(webm_tmp.name)

    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(src_path),
         "-c:v", "libvpx", "-b:v", "500k",
         "-c:a", "libvorbis", str(webm_path)],
        capture_output=True,
    )
    src_path.unlink(missing_ok=True)

    assert result.returncode == 0, (
        f"Failed to create test WebM: {result.stderr.decode(errors='replace')[-200:]}"
    )
    assert webm_path.stat().st_size > 0, "Created WebM is empty"
    return webm_path


def _webm_to_base64_uri(webm_path: Path) -> str:
    """Encode a WebM file as a data URI string (as the browser sends to Python)."""
    raw = webm_path.read_bytes()
    return "data:video/webm;base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Test: FFmpeg binary is available
# ---------------------------------------------------------------------------

def test_ffmpeg_binary_available():
    """imageio-ffmpeg must provide a usable FFmpeg executable."""
    exe = _get_ffmpeg_bin()
    assert exe is not None, "FFmpeg binary not found. Run: pip install imageio-ffmpeg"
    assert Path(exe).exists(), f"FFmpeg binary path does not exist: {exe}"

    result = subprocess.run([exe, "-version"], capture_output=True)
    assert result.returncode == 0, "FFmpeg binary did not exit cleanly with -version"
    version_line = result.stdout.decode(errors="replace").split("\n")[0]
    assert "ffmpeg version" in version_line.lower(), f"Unexpected version output: {version_line}"


# ---------------------------------------------------------------------------
# Test: base64 WebM -> MP4 conversion
# ---------------------------------------------------------------------------

def test_transcode_webcam_data_to_mp4_returns_mp4_extension():
    """Output path must have .mp4 suffix, never .webm."""
    webm_path = _make_vp8_webm(frames=15)
    b64 = _webm_to_base64_uri(webm_path)
    webm_path.unlink(missing_ok=True)

    result = transcode_webcam_data_to_mp4(b64)
    try:
        assert result.suffix.lower() == ".mp4", (
            f"Expected .mp4 suffix, got '{result.suffix}'"
        )
    finally:
        result.unlink(missing_ok=True)


def test_transcode_webcam_data_to_mp4_file_exists_and_non_empty():
    """Transcoded MP4 must physically exist with non-zero size."""
    webm_path = _make_vp8_webm(frames=20)
    b64 = _webm_to_base64_uri(webm_path)
    webm_path.unlink(missing_ok=True)

    result = transcode_webcam_data_to_mp4(b64)
    try:
        assert result.exists(), f"Output MP4 does not exist: {result}"
        size = result.stat().st_size
        assert size > 1000, f"Output MP4 is suspiciously small: {size} bytes"
    finally:
        result.unlink(missing_ok=True)


def test_transcode_webcam_data_strips_data_uri_prefix():
    """Function must accept both plain base64 and data:...,base64 URIs."""
    webm_path = _make_vp8_webm(frames=10)
    raw = webm_path.read_bytes()
    webm_path.unlink(missing_ok=True)

    plain_b64 = base64.b64encode(raw).decode("ascii")
    uri_b64 = "data:video/webm;base64," + plain_b64

    for b64_input in [plain_b64, uri_b64]:
        result = transcode_webcam_data_to_mp4(b64_input)
        try:
            assert result.exists() and result.stat().st_size > 0
            assert result.suffix.lower() == ".mp4"
        finally:
            result.unlink(missing_ok=True)


def test_transcode_webcam_data_type_error_on_wrong_type():
    """Passing a non-string must raise TypeError immediately."""
    with pytest.raises(TypeError):
        transcode_webcam_data_to_mp4(b"raw bytes")  # type: ignore

    with pytest.raises(TypeError):
        transcode_webcam_data_to_mp4(12345)  # type: ignore


def test_transcode_webcam_data_unpadded_and_whitespace():
    """
    Test that base64 strings with missing '=' padding, leading/trailing spaces,
    embedded newlines, and complex data URI headers decode and transcode correctly.
    """
    webm_path = _make_vp8_webm(frames=15)
    raw = webm_path.read_bytes()
    webm_path.unlink(missing_ok=True)

    b64_plain = base64.b64encode(raw).decode("ascii")
    # Strip '=' padding to simulate unpadded Base64
    unpadded_b64 = b64_plain.rstrip("=")

    # Add extra newlines, spaces, and full data URI header
    dirty_b64 = (
        "  data:video/webm;codecs=vp8,opus;base64,\n"
        + unpadded_b64[:100] + "\n \r\n "
        + unpadded_b64[100:300] + "\n  "
        + unpadded_b64[300:] + "   \n"
    )

    result = transcode_webcam_data_to_mp4(dirty_b64)
    try:
        assert result.exists(), f"Output MP4 does not exist: {result}"
        assert result.suffix.lower() == ".mp4"
        assert result.stat().st_size > 0

        # Verify VideoLoader accepts it
        loader = VideoLoader(result)
        assert loader.metadata["frame_count"] > 0
    finally:
        result.unlink(missing_ok=True)


def test_transcode_webcam_data_invalid_payload_raises():
    """Passing empty or short invalid strings must raise ValueError."""
    with pytest.raises(ValueError):
        transcode_webcam_data_to_mp4("")

    with pytest.raises(ValueError):
        transcode_webcam_data_to_mp4("short string")


# ---------------------------------------------------------------------------
# Test: VideoLoader accepts the transcoded MP4
# ---------------------------------------------------------------------------

def test_transcoded_mp4_passes_video_loader():
    """
    VideoLoader must accept the transcoded MP4 without raising
    VideoValidationError (which rejects unsupported extensions like .webm).
    """
    FRAMES = 30
    webm_path = _make_vp8_webm(frames=FRAMES, fps=10.0, width=320, height=240)
    b64 = _webm_to_base64_uri(webm_path)
    webm_path.unlink(missing_ok=True)

    mp4_path = transcode_webcam_data_to_mp4(b64)
    try:
        loader = VideoLoader(mp4_path)
        meta = loader.metadata

        assert meta["file_name"].endswith(".mp4"), (
            f"VideoLoader metadata shows non-.mp4 name: {meta['file_name']}"
        )
        assert meta["frame_count"] > 0, "VideoLoader reports 0 frames"
        assert meta["fps"] > 0, "VideoLoader reports 0 FPS"
        assert meta["duration_sec"] > 0, "VideoLoader reports 0 duration"
    finally:
        mp4_path.unlink(missing_ok=True)


def test_transcoded_mp4_readable_by_opencv():
    """OpenCV must be able to read individual frames from the transcoded MP4."""
    FRAMES = 20
    webm_path = _make_vp8_webm(frames=FRAMES, fps=10.0, width=320, height=240)
    b64 = _webm_to_base64_uri(webm_path)
    webm_path.unlink(missing_ok=True)

    mp4_path = transcode_webcam_data_to_mp4(b64)
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        assert cap.isOpened(), f"OpenCV cannot open transcoded MP4: {mp4_path}"

        read_count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            assert frame.shape[2] == 3, "Frames must be 3-channel BGR"
            read_count += 1
        cap.release()

        assert read_count > 0, "OpenCV read 0 frames from transcoded MP4"
    finally:
        mp4_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test: transcode_webcam_data_to_mp4_from_file (uploaded WebM)
# ---------------------------------------------------------------------------

def test_transcode_from_file_webm_to_mp4():
    """Uploaded .webm file must be converted to .mp4."""
    webm_path = _make_vp8_webm(frames=15)
    try:
        result = transcode_webcam_data_to_mp4_from_file(webm_path)
        try:
            assert result.suffix.lower() == ".mp4"
            assert result.exists()
            assert result.stat().st_size > 0

            loader = VideoLoader(result)
            assert loader.metadata["frame_count"] > 0
        finally:
            result.unlink(missing_ok=True)
    finally:
        webm_path.unlink(missing_ok=True)


def test_transcode_from_file_mp4_passthrough():
    """If input is already .mp4, it must be returned as-is (no re-encoding)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    src = Path(tmp.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(src), fourcc, 10.0, (160, 120))
    for _ in range(5):
        vw.write(np.zeros((120, 160, 3), dtype=np.uint8))
    vw.release()

    try:
        result = transcode_webcam_data_to_mp4_from_file(src)
        assert result == src, f"MP4 passthrough should return same path, got {result}"
    finally:
        src.unlink(missing_ok=True)


def test_transcode_from_file_missing_raises():
    """Missing input file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        transcode_webcam_data_to_mp4_from_file(Path("/nonexistent/file.webm"))
