"""
Streamlit Web Demonstration Dashboard for Online Exam Proctoring Pipeline.
Integrates Video Ingestion -> Preprocessing -> Key-Frame Extraction -> Model Inference -> Sliding Window Aggregation -> Report Export.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root directory to sys.path to enable 'src' module resolution when launched via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import numpy as np
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="AI Exam Proctoring - Abnormal Activity Detection",
    page_icon="🎓",
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
from src.aggregation.sliding_window import (
    SlidingWindowAggregator, FramePrediction, merge_multimodel_alerts
)
from src.aggregation.report_generator import ReportGenerator
from app.components.timeline_ui import (
    render_kpi_cards, render_timeline_table, render_class_distribution_bar,
    render_model_comparison_section
)

logger = get_logger("streamlit_app")

CNN_BENCHMARK_MODELS = ["custom_cnn", "densenet121", "inception_v3", "inception_resnet_v2"]


def main():
    st.title("🎓 Online Exam Proctoring System")
    st.caption(
        "Effectiveness of Pre-Trained CNN Networks for Detecting Abnormal Activities in Online Exams "
        "(Ramzan et al., IEEE Access, 2024)"
    )
    st.divider()

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Pipeline Configuration")

    st.sidebar.subheader("🤖 Active Ensemble Classifier Models (4 CNNs)")
    for m_name in CNN_BENCHMARK_MODELS:
        ckpt_path = Path("weights") / f"{m_name}_best.pt"
        display_title = m_name.replace('_', ' ').title()
        if ckpt_path.exists():
            st.sidebar.success(f"✅ {display_title}: `{ckpt_path.name}` Loaded")
        else:
            st.sidebar.warning(f"⚠️ {display_title}: Base Weights (No Checkpoint)")

    st.sidebar.subheader("🎯 Object Detection & Localization Branch")
    yolo_ckpt = Path("weights") / "yolov5_best.pt"
    if yolo_ckpt.exists():
        st.sidebar.success("✅ YOLOv5 Detector: `yolov5_best.pt` Loaded")
    else:
        st.sidebar.info("ℹ️ YOLOv5 Detector: Trained Checkpoint Unavailable (Branch Ready for Fine-Tuning)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📐 Methodology Parameters")
    st.sidebar.info(
        f"**Motion Threshold (T)**: `{MOTION_THRESHOLD:,}`\n\n"
        f"**Frame Skip Factor**: `{FRAME_SKIP}` frames\n\n"
        f"**Sliding Window**: `{SLIDING_WINDOW_SECONDS}s`\n\n"
        f"**Alert Confidence Threshold**: `{CONFIDENCE_THRESHOLD * 100:.0f}%`"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Target Behavior Classes (5)")
    for cls in CLASS_NAMES:
        st.sidebar.text(f"• {cls.replace('_', ' ').title()}")

    # --- VIDEO INPUT SELECTION ---
    st.subheader("📹 Step 1: Input Exam Video Feed")
    input_type = st.radio("Select Video Input Mode:", ["Upload MP4/AVI Video File", "Use Sample Video Clip"], horizontal=True)

    temp_video_path = None

    if input_type == "Upload MP4/AVI Video File":
        uploaded_file = st.file_uploader(
            "Upload Examination Webcam Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload candidate webcam recording for automated cheating detection."
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
            tfile.write(uploaded_file.read())
            tfile.close()
            temp_video_path = Path(tfile.name)
    else:
        # Check for demo clips directory
        demo_dir = Path("data/demo_clips")
        demo_files = list(demo_dir.glob("*.mp4")) + list(demo_dir.glob("*.avi")) if demo_dir.exists() else []

        if demo_files:
            selected_demo = st.selectbox("Select Sample Clip:", options=[f.name for f in demo_files])
            temp_video_path = demo_dir / selected_demo
            st.caption("ℹ️ *Sample clip selected for pipeline verification.*")
        else:
            st.warning("⚠️ No sample clips found in `data/demo_clips/`. Please upload a video file above.")

    # --- INFERENCE RUNNER ---
    if temp_video_path and temp_video_path.exists():
        st.success(f"Loaded video source: `{temp_video_path.name}`")

        if st.button("🚀 Run Multi-Model Proctoring Analysis Pipeline", type="primary", use_container_width=True):
            run_proctoring_pipeline(temp_video_path)

    # Cleanup temp upload file on complete if needed
    st.markdown("---")
    st.caption("Developed following the research paper methodology by Ramzan et al. (2024).")


def run_proctoring_pipeline(video_path: Path):
    """Executes full multi-model proctoring analysis pipeline across all 4 trained CNN architectures."""
    progress_bar = st.progress(0, text="Initializing Video Ingestion & Building 4 CNN Models...")

    try:
        # 1. Ingest Video
        loader = VideoLoader(video_path)
        meta = loader.metadata
        total_frames = meta["frame_count"]

        # 2. Instantiate Preprocessors, Key-Frame Extractor, Models & Aggregators for all 4 CNNs
        keyframe_extractor = KeyFrameExtractor(threshold=MOTION_THRESHOLD, frame_skip=FRAME_SKIP)

        preprocessors = {}
        models = {}
        aggregators = {}

        for m_name in CNN_BENCHMARK_MODELS:
            target_size = INCEPTION_IMAGE_SIZE if "inception" in m_name else DEFAULT_IMAGE_SIZE
            preprocessors[m_name] = FramePreprocessor(target_size=target_size)
            models[m_name] = build_model(m_name, load_best_checkpoint=True)
            if hasattr(models[m_name], "eval"):
                models[m_name].eval()
            aggregators[m_name] = SlidingWindowAggregator(
                window_seconds=SLIDING_WINDOW_SECONDS,
                confidence_threshold=CONFIDENCE_THRESHOLD
            )

        st.subheader("⚡ Step 2: Processing & Multi-Model Inference Analysis")

        col_left, col_right = st.columns([1.2, 1])
        frame_placeholder = col_left.empty()
        status_placeholder = col_right.empty()

        key_frames_count = 0

        # Iterate over frames using frame_skip=3
        for frame_idx, timestamp_sec, raw_frame in loader.read_frames(frame_skip=FRAME_SKIP):
            # Key-frame motion threshold evaluation (|dF| > 340,000)
            is_key_frame, sum_diff, diff_meta = keyframe_extractor.process_frame(
                raw_frame, frame_idx, timestamp_sec
            )

            if is_key_frame:
                key_frames_count += 1
                latest_predictions_summary = {}

                # Run inference through all 4 trained CNN models
                for m_name in CNN_BENCHMARK_MODELS:
                    proc_frame = preprocessors[m_name].preprocess(raw_frame)
                    tensor_img = torch.from_numpy(proc_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                    with torch.no_grad():
                        logits = models[m_name](tensor_img)
                        probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()

                    top_idx = int(np.argmax(probabilities))
                    pred_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else "eye_movement"
                    conf = float(probabilities[top_idx])
                    probs = {cls: float(probabilities[i]) for i, cls in enumerate(CLASS_NAMES) if i < len(probabilities)}

                    frame_pred = FramePrediction(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        predicted_class=pred_class,
                        confidence=conf,
                        probabilities=probs
                    )
                    aggregators[m_name].add_prediction(frame_pred)
                    latest_predictions_summary[m_name] = f"{pred_class} ({conf * 100:.1f}%)"

                # Render live video preview overlay
                disp_bgr = cv2.resize(raw_frame, (480, 270))
                cv2.putText(
                    disp_bgr, f"KEY-FRAME #{key_frames_count} ({timestamp_sec:.1f}s)",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(disp_rgb, caption=f"Frame #{frame_idx} ({timestamp_sec:.1f}s) - Motion Diff: {sum_diff:,.0f}", use_container_width=True)

                summary_text = "\n\n".join([f"• **{m.replace('_', ' ').title()}**: `{pred}`" for m, pred in latest_predictions_summary.items()])
                status_placeholder.markdown(
                    f"**Current Frame**: `{frame_idx}` / `{total_frames}`\n\n"
                    f"**Extracted Key-Frames**: `{key_frames_count}`\n\n"
                    f"**Motion Sum |dF|**: `{sum_diff:,.0f}`\n\n"
                    f"### Latest 4-CNN Inference:\n{summary_text}"
                )

            # Update progress bar
            pct = min(1.0, frame_idx / max(1, total_frames))
            progress_bar.progress(pct, text=f"Processing video frame {frame_idx}/{total_frames} across 4 CNN models...")

        progress_bar.progress(1.0, text="Multi-Model Analysis Complete!")
        st.success("🎉 Multi-Model Proctoring Analysis Completed Successfully!")

        # 4. Multi-Model Consensus Aggregation & Report Generation
        report_gen = ReportGenerator()
        model_alerts_dict = {}
        per_model_reports = {}

        for m_name in CNN_BENCHMARK_MODELS:
            m_display = m_name.replace("_", " ").title()
            m_alerts = aggregators[m_name].get_merged_alerts()
            model_alerts_dict[m_display] = m_alerts

            per_model_reports[m_display] = report_gen.generate_session_report(
                video_name=video_path.name,
                total_duration_sec=meta["duration_sec"],
                total_frames=total_frames,
                key_frames_analyzed=key_frames_count,
                model_name=m_name,
                alerts=m_alerts,
                json_filename=f"{m_name}_session_report.json",
                csv_filename=f"{m_name}_session_report.csv"
            )

        # Merge overlapping/contiguous detections across models into consensus alerts
        all_predictions_map = {
            m_name.replace("_", " ").title(): aggregators[m_name].predictions
            for m_name in CNN_BENCHMARK_MODELS
        }
        consensus_alerts = merge_multimodel_alerts(model_alerts_dict, all_predictions_map=all_predictions_map)

        consensus_report = report_gen.generate_session_report(
            video_name=video_path.name,
            total_duration_sec=meta["duration_sec"],
            total_frames=total_frames,
            key_frames_analyzed=key_frames_count,
            model_name="Multi-CNN Consensus (Custom CNN, DenseNet121, InceptionV3, Inception-ResNet-v2)",
            alerts=consensus_alerts,
            json_filename="consensus_session_report.json",
            csv_filename="consensus_session_report.csv"
        )

        st.divider()
        st.subheader("📊 Step 3: Session Summary & Consensus Abnormal Activity Timeline")

        # Render KPI Cards for Consensus Results
        render_kpi_cards(
            session_meta=consensus_report["session_metadata"],
            summary_stats=consensus_report["summary_statistics"]
        )

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            render_timeline_table(consensus_report["timeline"])
        with col_b:
            render_class_distribution_bar(
                class_counts=consensus_report["summary_statistics"]["class_wise_counts"],
                class_durations=consensus_report["summary_statistics"].get("class_wise_duration_sec")
            )

        st.divider()

        # Render Model-Wise Comparison Section
        render_model_comparison_section(per_model_reports)

        st.divider()

        # Download Buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Consensus Session Report (JSON)",
                data=json.dumps(consensus_report, indent=2),
                file_name=f"{video_path.stem}_consensus_report.json",
                mime="application/json",
                use_container_width=True
            )
        with col_dl2:
            csv_path = Path("results/consensus_session_report.csv")
            csv_data = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
            st.download_button(
                label="📥 Download Consensus Flagged Timeline (CSV)",
                data=csv_data,
                file_name=f"{video_path.stem}_consensus_timeline.csv",
                mime="text/csv",
                use_container_width=True
            )

    except VideoValidationError as e:
        st.error(f"❌ Video Validation Error: {e}")
    except Exception as e:
        st.error(f"❌ Pipeline Execution Error: {e}")
        logger.exception(f"Pipeline failure on '{video_path}'")


if __name__ == "__main__":
    main()
