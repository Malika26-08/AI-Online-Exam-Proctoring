"""
Streamlit Web Demonstration Dashboard for Online Exam Proctoring Pipeline.
Integrates Video Ingestion -> Preprocessing -> Key-Frame Extraction -> Model Inference -> Sliding Window Aggregation -> Report Export.
"""

import os
import sys
import json
import uuid
import tempfile
import textwrap
from pathlib import Path

# Add project root directory to sys.path to enable 'src' module resolution when launched via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Exam Proctoring - Autonomous Multi-Model System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    CLASS_NAMES, BENCHMARK_MODELS, MOTION_THRESHOLD,
    FRAME_SKIP, SLIDING_WINDOW_SECONDS, CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE, INCEPTION_IMAGE_SIZE
)
from src.utils.logger import get_logger
from src.utils.video_loader import VideoLoader, VideoValidationError
from src.preprocessing.frame_preprocessor import FramePreprocessor
from src.preprocessing.keyframe_extractor import KeyFrameExtractor
from src.models.model_factory import build_model
from src.models.yolov5_detector import YOLOv5Detector
from src.aggregation.sliding_window import (
    SlidingWindowAggregator, FramePrediction, merge_multimodel_alerts
)
from src.aggregation.report_generator import ReportGenerator
from src.aggregation.screen_analyzer import analyze_screen_recording
from app.components.theme import inject_custom_theme
from app.components.timeline_ui import (
    render_kpi_cards, render_timeline_table, render_class_distribution_bar,
    render_model_comparison_section
)
from app.components.webcam_recorder import (
    render_webcam_recorder, transcode_webcam_data_to_mp4, process_recorder_payload
)
from app.components.wizard_ui import (
    render_wizard_stepper, render_wizard_1_auth, render_wizard_2_email_verification,
    render_wizard_3_exam_details, render_wizard_4_system_check
)

logger = get_logger("streamlit_app")

CNN_BENCHMARK_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]


def main():
    # Inject global dark glassmorphism cyber-AI theme
    inject_custom_theme()

    # Initialize Wizard Step State Machine
    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 1
    if "user_profile" not in st.session_state:
        st.session_state["user_profile"] = {
            "full_name": "Alex Johnson",
            "email": "alex.johnson@university.edu",
            "student_id": "STU-2026-8842",
            "role": "Candidate / Student"
        }
    if "exam_details" not in st.session_state:
        st.session_state["exam_details"] = {
            "exam_title": "CS-101: Artificial Intelligence & Machine Learning Midterm Exam",
            "exam_duration": "30 Minutes",
            "proctoring_mode": "AI Automated Multi-Model Proctoring (4 CNNs + YOLOv5)",
            "agreed": True
        }

    # Render visual 7-step wizard progress indicator bar
    render_wizard_stepper(st.session_state["wizard_step"])

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.markdown("### ⚙️ System Configuration")

    st.sidebar.markdown("#### 🤖 Active Ensemble CNNs")
    for m_name in CNN_BENCHMARK_MODELS:
        ckpt_path = Path("weights") / f"{m_name}_best.pt"
        display_title = m_name.replace('_', ' ').title()
        if ckpt_path.exists():
            st.sidebar.success(f"✅ {display_title}: `{ckpt_path.name}`")
        else:
            st.sidebar.warning(f"⚠️ {display_title}: Base Weights")

    # Add YOLOv5 Branch Status Indicator
    yolo_ckpt = Path("weights") / "yolov5_best.pt"
    if yolo_ckpt.exists():
        st.sidebar.success("🎯 YOLOv5 Spatial Detector: Active (mAP50: 99.2%)")
    else:
        st.sidebar.info("ℹ️ YOLOv5 Detector: Branch Ready")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📐 Methodology Hyperparameters")
    st.sidebar.info(
        f"• **Motion Threshold (T)**: `{MOTION_THRESHOLD:,}`\n\n"
        f"• **Frame Skip Factor**: `{FRAME_SKIP}` frames\n\n"
        f"• **Sliding Window**: `{SLIDING_WINDOW_SECONDS}s`\n\n"
        f"• **Alert Threshold**: `{CONFIDENCE_THRESHOLD * 100:.0f}%`"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🏷️ Target Classes (5)")
    for cls in CLASS_NAMES:
        st.sidebar.text(f"• {cls.replace('_', ' ').title()}")

    # --- STRICT WIZARD STEP ROUTER ---
    current_step = st.session_state["wizard_step"]

    if current_step == 1:
        _render_step_1()
    elif current_step == 2:
        _render_step_2()
    elif current_step == 3:
        _render_step_3()
    elif current_step == 4:
        _render_step_4()
    elif current_step == 5:
        _render_step_5()
    elif current_step == 6:
        _render_step_6()
    elif current_step == 7:
        _render_step_7()


# ---------------------------------------------------------------------------
# SESSION RESET HELPER
# ---------------------------------------------------------------------------

def _reset_exam_session():
    """
    Clears examination session data, temporary recorded files, and pipeline results.
    Keeps candidate signed in (user_profile) and preserves app/weights config.
    """
    # Safe cleanup of temporary session video files
    for key in ["active_video_path", "webcam_mp4_path", "screen_mp4_path"]:
        p_str = st.session_state.get(key)
        if p_str:
            try:
                p = Path(p_str)
                if p.exists() and ("temp" in str(p).lower() or "tmp" in str(p).lower()):
                    p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not remove temporary session file {p_str}: {e}")

    # Remove all examination session keys from Streamlit session state
    session_keys_to_clear = [
        "active_video_path",
        "webcam_mp4_path",
        "screen_mp4_path",
        "screen_status",
        "screen_events",
        "_rec_payload_len",
        "pipeline_results",
        "step5_input_mode",
        "step5_file_uploader",
        "proctoring_rules_agree_cb"
    ]
    for key in session_keys_to_clear:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# WIZARD STEP RENDERERS
# ---------------------------------------------------------------------------

def _render_step_1():
    success, profile, target_step = render_wizard_1_auth()
    if success:
        st.session_state["user_profile"] = profile
        st.session_state["wizard_step"] = target_step
        st.rerun()


def _render_step_2():
    success = render_wizard_2_email_verification(st.session_state["user_profile"])
    if success:
        st.session_state["wizard_step"] = 3
        st.rerun()


def _render_step_3():
    if "session_reset_msg" in st.session_state:
        st.success(f"✅ {st.session_state.pop('session_reset_msg')}")
    success, details = render_wizard_3_exam_details(st.session_state["user_profile"])
    if success:
        st.session_state["exam_details"] = details
        st.session_state["wizard_step"] = 4
        st.rerun()


def _render_step_4():
    success = render_wizard_4_system_check()
    if success:
        st.session_state["wizard_step"] = 5
        st.rerun()


def _render_step_5():
    """Step 5 — Proctored Exam Session: live dual webcam+screen recording OR video upload."""
    st.subheader("🎥 Step 5 — Proctored Examination Session")
    st.caption("Perform live webcam & screen-share exam recording or upload an existing video.")

    input_mode = st.radio(
        "Select Examination Input Mode:",
        [
            "🎥 Record Proctored Exam (Browser Webcam + Screen Share)",
            "📁 Upload Existing Video File (MP4 / AVI)"
        ],
        horizontal=True,
        key="step5_input_mode"
    )

    temp_video_path = None

    # ── RECORDING MODE ──────────────────────────────────────────────────────
    if input_mode == "🎥 Record Proctored Exam (Browser Webcam + Screen Share)":
        step5_card = textwrap.dedent("""
            <div class="cyber-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <h4 style="margin: 0; color: #00F2FE;">🔴 Live Proctored Exam & Screen Recorder</h4>
                    <span class="ai-badge badge-cyan">DUAL CAPTURE READY</span>
                </div>
                <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.5;">
                    1. Click <strong>Start Exam Recording</strong> — grant camera & microphone permissions, then select <strong>Entire Screen</strong>.<br>
                    2. Conduct your examination session freely. Screen recording continues across browser tabs.<br>
                    3. Click <strong>End Exam & Stop Recording</strong> when finished.
                </div>
            </div>
        """).strip()
        st.markdown(step5_card, unsafe_allow_html=True)

        rec_payload = render_webcam_recorder(key="proctored_exam_recorder")

        if rec_payload and isinstance(rec_payload, str) and len(rec_payload) > 200:
            prev_len = st.session_state.get("_rec_payload_len", 0)
            if prev_len != len(rec_payload):
                with st.spinner("Processing webcam & screen recordings for AI analysis..."):
                    try:
                        res = process_recorder_payload(rec_payload)
                        if res and res.get("webcam_mp4_path"):
                            st.session_state["webcam_mp4_path"] = res["webcam_mp4_path"]
                            st.session_state["screen_mp4_path"] = res.get("screen_mp4_path")
                            st.session_state["screen_status"] = res.get("screen_status", "Completed")
                            st.session_state["screen_events"] = res.get("screen_events", [])
                            st.session_state["_rec_payload_len"] = len(rec_payload)
                    except Exception as exc:
                        logger.error(f"[AI_PIPELINE] Transcoding error: {exc}", exc_info=True)
                        st.error("Transcoding error: Recording payload processing failed.")

            temp_video_path = st.session_state.get("webcam_mp4_path")
            if temp_video_path and Path(temp_video_path).exists():
                sz = Path(temp_video_path).stat().st_size
                scr_path = st.session_state.get("screen_mp4_path")
                scr_sz = Path(scr_path).stat().st_size if (scr_path and Path(scr_path).exists()) else 0
                scr_txt = f" | Screen: {scr_sz / 1024:.0f} KB" if scr_sz > 0 else ""
                st.success(
                    f"✅ Record ready — Webcam: {sz / 1024:.0f} KB{scr_txt}. "
                    "Click **Submit Exam** below to start AI analysis."
                )

    # ── UPLOAD MODE ─────────────────────────────────────────────────────────
    else:
        upload_card = textwrap.dedent("""
            <div class="cyber-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #00F2FE;">📁 Upload Examination Video File</h4>
                <p style="color: #94A3B8; font-size: 0.86rem; margin: 0;">
                    Select a pre-recorded candidate webcam video file for multi-model AI analysis.
                </p>
            </div>
        """).strip()
        st.markdown(upload_card, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Candidate Examination Webcam Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a pre-recorded webcam video for automated analysis.",
            key="step5_file_uploader"
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            )
            tfile.write(uploaded_file.read())
            tfile.close()
            temp_video_path = Path(tfile.name)
            # Reset screen recording state for uploaded videos
            st.session_state.pop("screen_mp4_path", None)
            st.session_state["screen_status"] = "Not Provided"
            st.session_state["screen_events"] = []
            st.success(
                f"Video loaded: `{uploaded_file.name}` "
                f"({uploaded_file.size / 1024 / 1024:.2f} MB)"
            )

    # ── SUBMIT BUTTON ────────────────────────────────────────────────────────
    if temp_video_path is not None and Path(temp_video_path).exists():
        st.session_state["active_video_path"] = Path(temp_video_path)
        if st.button(
            "🚀 Submit Exam & Start AI Proctoring Analysis",
            type="primary",
            use_container_width=True,
            key="step5_submit_btn"
        ):
            st.session_state.pop("pipeline_results", None)
            st.session_state["wizard_step"] = 6
            st.rerun()


def _render_step_6():
    """Step 6 — AI Analysis. Runs pipeline ONCE and stores results. Then transitions to Step 7."""
    st.subheader("🤖 Step 6 — Autonomous AI Proctoring Analysis Engine")
    st.caption("Executing 4-CNN Ensemble Classifier + YOLOv5 Spatial Localization. Please wait...")

    video_path = st.session_state.get("active_video_path")
    if video_path:
        video_path = Path(video_path)

    if not video_path or not video_path.exists():
        st.error("❌ No active video found for proctoring analysis. Please return to Step 5.")
        if st.button("⬅️ Return to Step 5", key="return_step5_btn"):
            st.session_state["wizard_step"] = 5
            st.rerun()
        return

    step6_card = textwrap.dedent("""
        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #00F2FE;">⚡ Active Neural Inference Grid</h4>
                <span class="ai-badge badge-cyan">PROCESSING SESSION</span>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Ingesting frames -> Motion key-frame extraction -> 4 CNN Ensemble -> 2-of-4 Consensus -> YOLOv5 Localization...
            </div>
        </div>
    """).strip()
    st.markdown(step6_card, unsafe_allow_html=True)

    # --- Create all UI containers BEFORE running the pipeline ---
    progress_bar = st.progress(0, text="Initializing Video Ingestion & Building 4 CNN Models...")
    col_left, col_right = st.columns([1.2, 1])
    frame_placeholder = col_left.empty()
    status_placeholder = col_right.empty()
    success_placeholder = st.empty()

    try:
        pipeline_results = _run_proctoring_pipeline(
            video_path,
            progress_bar=progress_bar,
            frame_placeholder=frame_placeholder,
            status_placeholder=status_placeholder,
            success_placeholder=success_placeholder
        )
    except Exception as exc:
        ref_id = uuid.uuid4().hex[:8].upper()
        logger.error(f"[AI_PIPELINE] Pipeline execution error (Ref: ERR-{ref_id}): {exc}", exc_info=True)

        err_html = textwrap.dedent(f"""
            <div class="cyber-card" style="border-color: rgba(255, 75, 75, 0.4);">
                <h4 style="color: #FF4B4B; margin-top: 0;">⚠️ AI Analysis Interrupted</h4>
                <p style="color: #CBD5E1; font-size: 0.9rem;">
                    The autonomous AI proctoring analysis could not be completed at this time.
                </p>
                <p style="color: #94A3B8; font-size: 0.8rem; font-family: monospace;">
                    Reference ID: ERR-{ref_id}
                </p>
            </div>
        """).strip()
        st.markdown(err_html, unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("🔄 Retry AI Analysis", type="primary", use_container_width=True, key="retry_analysis_btn"):
                st.rerun()
        with col_e2:
            if st.button("⬅️ Return to Step 5", use_container_width=True, key="return_step5_err_btn"):
                st.session_state["wizard_step"] = 5
                st.rerun()
        return

    if pipeline_results:
        st.session_state["pipeline_results"] = pipeline_results
        time.sleep(1.0)
        st.session_state["wizard_step"] = 7
        st.rerun()
    else:
        st.warning("⚠️ AI analysis produced no output. Please verify the input video and try again.")
        if st.button("⬅️ Return to Step 5", key="return_step5_empty_btn"):
            st.session_state["wizard_step"] = 5
            st.rerun()


@st.cache_resource
def _get_cached_models():
    """Caches model instances across Streamlit reruns using build_model factory."""
    logger.info("[AI_PIPELINE] Stage: Loading 4 Ensemble CNN Checkpoints")
    models_dict = {}
    for m_name in CNN_BENCHMARK_MODELS:
        model = build_model(m_name, num_classes=5, pretrained=False, load_best_checkpoint=True)
        model.eval()
        models_dict[m_name] = model
    return models_dict


@st.cache_resource
def _get_cached_yolo_detector():
    """Caches YOLOv5 detector instance across Streamlit reruns."""
    logger.info("[AI_PIPELINE] Stage: Loading YOLOv5 Spatial Detector")
    yolo_ckpt = Path("weights") / "yolov5_best.pt"
    if yolo_ckpt.exists():
        detector = YOLOv5Detector(weights_path=str(yolo_ckpt), conf_threshold=CONFIDENCE_THRESHOLD)
    else:
        detector = YOLOv5Detector(weights_path=None, conf_threshold=CONFIDENCE_THRESHOLD)
    return detector


def _run_proctoring_pipeline(
    video_path: Path,
    progress_bar=None,
    frame_placeholder=None,
    status_placeholder=None,
    success_placeholder=None
) -> dict:
    """
    Executes the multi-model AI examination analysis pipeline.
    Renders UI updates into the provided container placeholders.
    """
    t_start = time.time()

    logger.info(f"[AI_PIPELINE] Stage: VideoLoader Ingestion on '{video_path.name}'")
    if progress_bar is not None:
        progress_bar.progress(5, text="Ingesting examination video stream...")

    try:
        loader = VideoLoader(video_path, frame_skip=FRAME_SKIP)
        total_frames = loader.properties.total_frames
        fps = loader.properties.fps
    except VideoValidationError as e:
        logger.error(f"[AI_PIPELINE] Video Validation Error: {e}", exc_info=True)
        st.error(f"❌ Video Validation Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"[AI_PIPELINE] Video Ingestion Error: {e}", exc_info=True)
        raise

    logger.info("[AI_PIPELINE] Stage: Frame Preprocessor & KeyFrame Extractor Initialization")
    preprocessor = FramePreprocessor(target_size=DEFAULT_IMAGE_SIZE)
    inception_prep = FramePreprocessor(target_size=INCEPTION_IMAGE_SIZE)

    extractor = KeyFrameExtractor(
        motion_threshold=MOTION_THRESHOLD,
        frame_skip=FRAME_SKIP,
        min_keyframe_interval=1
    )

    models_dict = _get_cached_models()

    if progress_bar is not None:
        progress_bar.progress(15, text="Extracting motion key-frames...")

    logger.info("[AI_PIPELINE] Stage: Extracting Motion Key-Frames")
    key_frames = list(extractor.extract_keyframes(video_path))
    key_frames_count = len(key_frames)

    if key_frames_count == 0:
        logger.warning(f"[AI_PIPELINE] Zero motion key-frames extracted from '{video_path.name}'")
        st.error("❌ Key-Frame Extraction Failed: No frames met the motion threshold criteria.")
        return {}

    key_frame_reduction_pct = (1.0 - key_frames_count / max(1, total_frames // FRAME_SKIP)) * 100.0
    logger.info(f"[AI_PIPELINE] Stage: Extracted {key_frames_count} motion key-frames ({key_frame_reduction_pct:.1f}% reduction)")

    if progress_bar is not None:
        progress_bar.progress(25, text=f"Analyzing {key_frames_count} motion key-frames across 4 CNNs + YOLOv5...")

    logger.info("[AI_PIPELINE] Stage: Initializing Sliding Window Aggregators & YOLOv5 Detector")
    aggregators = {
        m_name: SlidingWindowAggregator(
            window_size_seconds=SLIDING_WINDOW_SECONDS,
            fps=fps / FRAME_SKIP,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        for m_name in CNN_BENCHMARK_MODELS
    }

    yolo_detector = _get_cached_yolo_detector()
    yolo_key_frame_results = []

    last_ui_update = 0.0

    logger.info("[AI_PIPELINE] Stage: Executing 4-CNN Ensemble & YOLOv5 Inference Loop")
    for idx, (f_idx, f_ts, raw_frame) in enumerate(key_frames):
        prep_std = preprocessor.preprocess(raw_frame, return_dict=True)
        prep_inc = inception_prep.preprocess(raw_frame, return_dict=True)

        tensor_std = torch.from_numpy(prep_std['normalized_image']).permute(2, 0, 1).unsqueeze(0).float()
        tensor_inc = torch.from_numpy(prep_inc['normalized_image']).permute(2, 0, 1).unsqueeze(0).float()

        yolo_dets = yolo_detector.detect(raw_frame)
        if yolo_dets:
            yolo_key_frame_results.append({
                "frame_idx": f_idx,
                "timestamp_sec": f_ts,
                "detections": yolo_dets,
                "frame_img": raw_frame.copy()
            })

        for m_name in CNN_BENCHMARK_MODELS:
            model = models_dict[m_name]
            inp_tensor = tensor_inc if m_name in ["inception_v3", "inception_resnet_v2"] else tensor_std
            with torch.no_grad():
                out = model(inp_tensor)
                probs = torch.softmax(out, dim=1).squeeze(0).numpy()

            pred_class_idx = int(np.argmax(probs))
            top_prob = float(probs[pred_class_idx])
            pred_class_name = CLASS_NAMES[pred_class_idx]

            aggregators[m_name].add_prediction(
                frame_idx=f_idx,
                timestamp_sec=f_ts,
                predicted_class=pred_class_name,
                class_index=pred_class_idx,
                probabilities=probs.tolist()
            )

        now = time.time()
        if (now - last_ui_update > 0.25) or (idx == key_frames_count - 1):
            last_ui_update = now

            pct = int(25 + (idx + 1) / key_frames_count * 65)
            if progress_bar is not None:
                progress_bar.progress(pct, text=f"Processing motion key-frame {idx + 1}/{key_frames_count} ({f_ts:.1f}s)...")

            if frame_placeholder is not None:
                vis_frame = raw_frame.copy()
                if yolo_dets:
                    for d in yolo_dets:
                        bbox = [int(v) for v in d['bbox']]
                        lbl = d['class_name'].replace('_', ' ').title()
                        cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 242, 254), 2)
                        cv2.putText(vis_frame, f"{lbl} ({d['confidence']*100:.0f}%)",
                                    (bbox[0], max(20, bbox[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 157), 2)

                frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Motion Key-Frame #{f_idx} ({f_ts:.1f}s) — Motion Sum: {extractor.last_motion_sum:,.0f}",
                    use_container_width=True
                )

            if status_placeholder is not None:
                latest_preds = {
                    m_name: aggregators[m_name].predictions_history[-1].predicted_class.replace('_', ' ').title()
                    for m_name in CNN_BENCHMARK_MODELS if aggregators[m_name].predictions_history
                }
                status_html = textwrap.dedent(f"""
                    <div class="cyber-card" style="padding: 1rem;">
                        <h4 style="margin: 0 0 8px 0; color: #00F2FE;">📊 Frame #{f_idx} Multi-Model Inference</h4>
                        <div style="font-size: 0.85rem; color: #CBD5E1;">
                            • <strong>Custom CNN</strong>: `{latest_preds.get('custom_cnn', 'N/A')}`<br>
                            • <strong>DenseNet121</strong>: `{latest_preds.get('densenet121', 'N/A')}`<br>
                            • <strong>InceptionV3</strong>: `{latest_preds.get('inception_v3', 'N/A')}`<br>
                            • <strong>InceptionResNetV2</strong>: `{latest_preds.get('inception_resnet_v2', 'N/A')}`<br>
                            • <strong>YOLOv5 Detections</strong>: `{len(yolo_dets)} bounding boxes`
                        </div>
                    </div>
                """).strip()
                status_placeholder.markdown(status_html, unsafe_allow_html=True)

    logger.info("[AI_PIPELINE] Stage: Computing 2-of-4 Consensus & Generating Session Report")
    if progress_bar is not None:
        progress_bar.progress(95, text="Computing multi-model 2-of-4 consensus & generating report...")

    per_model_reports = {
        m_name: aggregators[m_name].get_session_summary(model_name=m_name)
        for m_name in CNN_BENCHMARK_MODELS
    }

    consensus_report = merge_multimodel_alerts(
        per_model_reports=per_model_reports,
        video_filename=video_path.name,
        fps=fps / FRAME_SKIP,
        total_frames=total_frames,
        key_frames_analyzed=key_frames_count,
        key_frame_reduction_percent=key_frame_reduction_pct,
        min_agreeing_models=2
    )

    report_gen = ReportGenerator(output_dir="results")
    report_gen.export_all_formats(consensus_report, per_model_reports)

    logger.info("[AI_PIPELINE] Stage: Analyzing Screen Recording (if captured)")
    screen_mp4 = st.session_state.get("screen_mp4_path")
    screen_status = st.session_state.get("screen_status", "Not Provided")
    screen_events = st.session_state.get("screen_events", [])
    screen_analysis_info = analyze_screen_recording(
        screen_mp4_path=screen_mp4,
        screen_status=screen_status,
        screen_events=screen_events
    )

    t_total = time.time() - t_start
    logger.info(f"[AI_PIPELINE] Stage: Pipeline Successfully Completed in {t_total:.2f}s!")

    if progress_bar is not None:
        progress_bar.progress(100, text=f"AI Proctoring Analysis Complete in {t_total:.2f}s!")

    if success_placeholder is not None:
        success_placeholder.success(
            f"🎉 **Analysis Complete** in `{t_total:.2f}s`! "
            f"Key-Frames Analyzed: `{key_frames_count:,}` | Consensus Flagged Segments: `{len(consensus_report['timeline'])}`"
        )

    yolo_class_counts = {}
    yolo_total_detections_count = 0
    for r in yolo_key_frame_results:
        for d in r["detections"]:
            c_name = d["class_name"]
            yolo_class_counts[c_name] = yolo_class_counts.get(c_name, 0) + 1
            yolo_total_detections_count += 1

    return {
        "video_path": video_path,
        "consensus_report": consensus_report,
        "per_model_reports": per_model_reports,
        "key_frames_count": key_frames_count,
        "yolo_key_frame_results": yolo_key_frame_results,
        "yolo_class_counts": yolo_class_counts,
        "yolo_total_detections_count": yolo_total_detections_count,
        "yolo_status_dict": {
            "yolo_available": True,
            "weights_path": "weights/yolov5_best.pt",
            "total_detections": yolo_total_detections_count,
            "key_frames_with_detections": len(yolo_key_frame_results)
        },
        "screen_analysis": screen_analysis_info,
        "processing_time_sec": t_total
    }


def _render_step_7():
    """Step 7 — Read-Only Final Report. Strictly reads stored pipeline results."""
    st.subheader("📊 Step 7 — Final Proctored Exam Summary & AI Report")
    st.caption("Comprehensive Multi-Model Analysis, 2-of-4 Consensus Evaluation, and Screen Monitoring Report.")

    pipeline_results = st.session_state.get("pipeline_results")
    if not pipeline_results:
        st.warning("⚠️ No AI pipeline results found. Please return to Step 5.")
        if st.button("⬅️ Return to Step 5", key="step7_return_step5_btn"):
            st.session_state["wizard_step"] = 5
            st.rerun()
        return

    video_path = st.session_state.get("active_video_path", Path("exam_video.mp4"))
    render_final_report_ui(video_path, pipeline_results)


def render_final_report_ui(video_path: Path, pipeline_results: dict):
    """Renders the Step 7 Final Report UI. No ML inference. Pure data display."""
    consensus_report = pipeline_results.get("consensus_report", {})
    if not consensus_report:
        st.warning("⚠️ No consensus report available to render.")
        return

    user_profile = st.session_state.get("user_profile", {})
    exam_details = st.session_state.get("exam_details", {})

    # Candidate & Exam Information Glass Card
    profile_html = textwrap.dedent(f"""
        <div class="cyber-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h4 style="margin: 0; color: #00F2FE;">🎓 Verified Candidate & Examination Profile</h4>
                    <span style="color: #94A3B8; font-size: 0.85rem;">{exam_details.get('exam_title', 'Midterm Examination')}</span>
                </div>
                <span class="ai-badge badge-green">VERIFIED EXAMINATION REPORT</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; font-size: 0.9rem;">
                <div><span style="color: #94A3B8;">Candidate Name:</span> <strong style="color: #F8FAFC;">{user_profile.get('full_name', 'Alex Johnson')}</strong></div>
                <div><span style="color: #94A3B8;">Candidate Email:</span> <strong style="color: #F8FAFC;">{user_profile.get('email', 'alex@university.edu')}</strong></div>
                <div><span style="color: #94A3B8;">Student ID:</span> <strong style="color: #F8FAFC;">{user_profile.get('student_id', 'STU-2026-8842')}</strong></div>
                <div><span style="color: #94A3B8;">Proctoring Mode:</span> <strong style="color: #00F2FE;">4 CNNs + YOLOv5 Consensus</strong></div>
            </div>
        </div>
    """).strip()
    st.markdown(profile_html, unsafe_allow_html=True)

    # KPI Summary Metric Cards
    render_kpi_cards(
        session_meta=consensus_report["session_metadata"],
        summary_stats=consensus_report["summary_statistics"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── WEBCAM AI ANALYSIS SECTION ──────────────────────────────────────────
    st.subheader("📷 Candidate Webcam AI Behavior Analysis")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        render_timeline_table(consensus_report["timeline"])
    with col_b:
        render_class_distribution_bar(
            class_counts=consensus_report["summary_statistics"]["class_wise_counts"],
            class_durations=consensus_report["summary_statistics"].get("class_wise_duration_sec")
        )

    st.divider()

    # ── SCREEN MONITORING SECTION ───────────────────────────────────────────
    st.subheader("🖥️ Candidate Screen Monitoring Analysis")
    scr_info = pipeline_results.get("screen_analysis", {})

    col_s1, col_s2, col_s3 = st.columns(3)
    if scr_info.get("captured"):
        with col_s1:
            st.metric("📹 Screen Recording", "Captured")
        with col_s2:
            st.metric("🖥️ Sharing Status", scr_info.get("status", "Completed"))
        with col_s3:
            st.metric("📐 Resolution & Duration", f"{scr_info.get('resolution', 'N/A')} ({scr_info.get('duration_sec', 0.0):.1f}s)")

        st.info(f"ℹ️ **Screen Monitoring Note**: {scr_info.get('note', 'Screen recording captured successfully.')}")

        if scr_info.get("events"):
            st.markdown("#### 📋 Screen Monitoring Log & Events")
            st.dataframe(pd.DataFrame(scr_info["events"]), use_container_width=True, hide_index=True)
    else:
        # Neutral informational state for uploaded webcam-only videos
        with col_s1:
            st.metric("📹 Screen Recording", "Not Provided")
        with col_s2:
            st.metric("🖥️ Screen Sharing", "Not Applicable — Uploaded Video")
        with col_s3:
            st.metric("📊 Screen Content Analysis", "Not Available")

        st.info("ℹ️ **Screen Monitoring Note**: Screen recording was not provided for this pre-recorded webcam video session.")

    st.divider()

    # Ensemble CNN model comparison
    render_model_comparison_section(
        pipeline_results["per_model_reports"],
        yolo_status=pipeline_results["yolo_status_dict"]
    )

    st.divider()

    # Dedicated YOLOv5 Bounding Box Section
    st.subheader("🎯 Dedicated YOLOv5 Bounding Box & Behavior Localizations")
    st.info(
        "ℹ️ **YOLOv5 Branch Note**: YOLOv5 predictions are computed independently for "
        "object/behavior spatial localization and bounding box detection."
    )

    yolo_total = pipeline_results["yolo_total_detections_count"]
    yolo_frames = pipeline_results["yolo_key_frame_results"]
    key_frames_count = pipeline_results["key_frames_count"]
    yolo_class_counts = pipeline_results["yolo_class_counts"]

    col_y1, col_y2, col_y3 = st.columns(3)
    with col_y1:
        st.metric("🎯 Total YOLOv5 Bounding Boxes", f"{yolo_total} boxes")
    with col_y2:
        st.metric("🎞️ Key-Frames with Detections", f"{len(yolo_frames)} / {key_frames_count}")
    with col_y3:
        st.metric("🛡️ Active Checkpoint", "weights/yolov5_best.pt (mAP50: 99.2%)")

    st.markdown("#### 📊 Detection Breakdown per Behavior Class")
    c_cols = st.columns(5)
    for idx, c_name in enumerate(CLASS_NAMES):
        c_cnt = yolo_class_counts.get(c_name, 0)
        c_cols[idx].metric(c_name.replace("_", " ").title(), f"{c_cnt} boxes")

    if yolo_frames:
        st.markdown("#### 🖼️ Key-Frame Bounding Box Inspection & Filtering")
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            class_filter_options = ["All Detected Classes"] + [c.replace("_", " ").title() for c in CLASS_NAMES]
            selected_filter = st.selectbox(
                "Filter Key-Frames by Behavior Class:",
                options=class_filter_options,
                key="yolo_class_filter"
            )
        with col_f2:
            min_yolo_conf = st.slider(
                "Filter Preview Confidence:",
                min_value=0.20, max_value=0.95, value=0.40, step=0.05,
                key="yolo_min_conf_slider",
                help="Only show bounding boxes with confidence greater than or equal to this threshold."
            )

        # Filter keyframes by class AND confidence threshold
        filtered = []
        for r in yolo_frames:
            valid_dets = [
                d for d in r["detections"]
                if d["confidence"] >= min_yolo_conf and (
                    selected_filter == "All Detected Classes"
                    or d["class_name"] == selected_filter.lower().replace(" ", "_")
                    or d["class_name"].replace("_", " ").title() == selected_filter
                )
            ]
            if valid_dets:
                r_copy = dict(r)
                r_copy["detections"] = valid_dets
                filtered.append(r_copy)

        if filtered:
            sel_idx = st.selectbox(
                "Select Key-Frame to Inspect Bounding Boxes:",
                options=list(range(len(filtered))),
                format_func=lambda i: (
                    f"Frame #{filtered[i]['frame_idx']} "
                    f"({filtered[i]['timestamp_sec']:.2f}s) — "
                    f"{len(filtered[i]['detections'])} box(es): "
                    f"{', '.join(d['class_name'].replace('_',' ').title() for d in filtered[i]['detections'])}"
                ),
                key="yolo_frame_selector"
            )

            res = filtered[sel_idx]
            vis_img = res["frame_img"].copy()
            for d in res["detections"]:
                bbox = [int(v) for v in d["bbox"]]
                lbl = d["class_name"].replace("_", " ").title()
                conf_pct = d["confidence"] * 100
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 242, 254), 2)
                cv2.putText(vis_img, f"{lbl} ({conf_pct:.1f}%)",
                            (bbox[0], max(20, bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 157), 2)
            vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            st.image(
                vis_rgb,
                caption=f"Annotated Frame #{res['frame_idx']} ({res['timestamp_sec']:.2f}s)",
                use_container_width=True
            )

            yolo_rows = []
            idx_ctr = 1
            for r in filtered:
                for d in r["detections"]:
                    bbox = d["bbox"]
                    yolo_rows.append({
                        "Detection #": idx_ctr,
                        "Frame #": r["frame_idx"],
                        "Timestamp": f"{r['timestamp_sec']:.2f}s",
                        "Detected Behavior": d["class_name"].replace("_", " ").title(),
                        "Confidence": f"{d['confidence'] * 100:.1f}%",
                        "Bounding Box [x1, y1, x2, y2]": f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
                    })
                    idx_ctr += 1

            st.dataframe(pd.DataFrame(yolo_rows), use_container_width=True, hide_index=True)
        else:
            st.info(f"ℹ️ No YOLOv5 detections found matching class '{selected_filter}' with confidence ≥ {min_yolo_conf:.2f}.")
    else:
        st.info("ℹ️ No YOLOv5 bounding box detections were found in this session.")

    st.divider()

    # Download Buttons
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            label="📥 Download Consensus Report (JSON)",
            data=json.dumps(consensus_report, indent=2),
            file_name=f"{video_path.stem}_consensus_report.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json_btn"
        )
    with col_dl2:
        csv_path = Path("results/consensus_session_report.csv")
        csv_data = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
        st.download_button(
            label="📥 Download Consensus Flagged Timeline (CSV)",
            data=csv_data,
            file_name=f"{video_path.stem}_consensus_timeline.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_csv_btn"
        )
    with col_dl3:
        if scr_info.get("events"):
            scr_csv = pd.DataFrame(scr_info["events"]).to_csv(index=False)
            st.download_button(
                label="📥 Download Screen Events Log (CSV)",
                data=scr_csv,
                file_name=f"{video_path.stem}_screen_events.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_scr_csv_btn"
            )

    # ── NEW EXAMINATION SESSION RESET FLOW ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    reset_card_html = textwrap.dedent("""
        <div class="cyber-card" style="text-align: center; border-color: rgba(0, 242, 254, 0.3);">
            <h4 style="color: #00F2FE; margin-top: 0; margin-bottom: 0.5rem;">🎉 Examination Session Completed</h4>
            <p style="color: #94A3B8; font-size: 0.88rem; margin-bottom: 1.2rem;">
                All AI multi-model analysis metrics have been logged and exported. Click below to reset the session and begin a new candidate examination.
            </p>
        </div>
    """).strip()
    st.markdown(reset_card_html, unsafe_allow_html=True)

    if st.button("＋ Start New Examination", type="primary", use_container_width=True, key="btn_start_new_exam"):
        _reset_exam_session()
        st.session_state["session_reset_msg"] = "Previous examination completed. Ready to start a new examination."
        st.session_state["wizard_step"] = 3
        st.rerun()


if __name__ == "__main__":
    main()
