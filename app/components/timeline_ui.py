"""
Streamlit Timeline & Summary UI Component.
Visualizes session metadata, key-frame statistics, and flagged abnormal activity timelines.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd


from src.config import CLASS_NAMES

CLASS_COLOR_MAP = {
    "eye_movement": "🟢 Eye Movement",
    "hand_move": "🟠 Hand Movement",
    "mobile_use": "🔴 Mobile Device Usage",
    "side_watching": "🟡 Side Watching",
    "mouth_open": "🔵 Mouth Open / Talking"
}


def render_kpi_cards(session_meta: Dict[str, Any], summary_stats: Dict[str, Any]):
    """Renders top-level KPI metric cards for session overview."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="⏱️ Total Exam Duration",
            value=session_meta.get("total_duration_formatted", "00:00:00")
        )

    with col2:
        key_frames = session_meta.get("key_frames_analyzed", 0)
        total_frames = session_meta.get("total_frames", 1)
        reduction = session_meta.get("key_frame_reduction_percent", 0.0)
        st.metric(
            label="🎞️ Key-Frames Processed",
            value=f"{key_frames:,} / {total_frames:,}",
            delta=f"-{reduction}% redundant frames"
        )

    with col3:
        total_alerts = summary_stats.get("total_flagged_segments", 0)
        st.metric(
            label="⚠️ Flagged Segments",
            value=f"{total_alerts} alerts",
            delta="Requires Review" if total_alerts > 0 else "Clean Session",
            delta_color="inverse" if total_alerts > 0 else "normal"
        )

    with col4:
        class_counts = summary_stats.get("class_wise_counts", {})
        top_abnormal = max(class_counts, key=class_counts.get) if class_counts and max(class_counts.values()) > 0 else "None"
        st.metric(
            label="🚨 Primary Risk Flag",
            value=top_abnormal.replace("_", " ").title() if top_abnormal != "None" else "None (Clean)"
        )


def render_timeline_table(timeline: List[Dict[str, Any]]):
    """Renders interactive timeline data table with formatted timestamps, model consensus, and confidence scores."""
    st.subheader("🚩 Time-Aligned Multi-Model Consensus Timeline")

    if not timeline:
        st.info("ℹ️ **No consensus abnormal activity detected** (fewer than 2 of 4 CNN models agreed on any abnormal behavior class).")
        return

    df = pd.DataFrame(timeline)

    # Format table for display
    display_df = pd.DataFrame()
    display_df["Alert #"] = df["alert_id"]
    display_df["Start Time"] = df["start_time"]
    display_df["End Time"] = df["end_time"]
    display_df["Duration"] = df["duration_sec"].apply(lambda x: f"{x:.1f}s")
    display_df["Predicted Activity"] = df["predicted_class"].apply(
        lambda x: CLASS_COLOR_MAP.get(x, x.replace("_", " ").title())
    )

    if "agreeing_models" in df.columns:
        display_df["Agreeing Models"] = df["agreeing_models"]
        display_df["# Models"] = df["num_agreeing_models"]

    display_df["Peak Confidence"] = df["peak_confidence"].apply(lambda x: f"{x * 100:.1f}%")
    display_df["Avg Confidence"] = df["average_confidence"].apply(lambda x: f"{x * 100:.1f}%")
    display_df["Key-Frames"] = df["key_frame_count"]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def render_class_distribution_bar(
    class_counts: Dict[str, int],
    class_durations: Optional[Dict[str, float]] = None
):
    """Renders breakdown charts showing both Consensus Alert Count and Total Abnormal Duration per class across all 5 classes."""
    st.subheader("📊 Class-Wise Consensus Distribution")

    tab1, tab2 = st.tabs(["🔢 Consensus Alert Count", "⏱️ Abnormal Duration (s)"])

    with tab1:
        full_counts = {cls: class_counts.get(cls, 0) for cls in CLASS_NAMES}
        df_counts = pd.DataFrame([
            {"Behavior Class": k.replace("_", " ").title(), "Alert Count": v}
            for k, v in full_counts.items()
        ])
        st.bar_chart(
            data=df_counts,
            x="Behavior Class",
            y="Alert Count",
            use_container_width=True
        )

    with tab2:
        dur_map = class_durations if class_durations else {}
        full_durations = {cls: dur_map.get(cls, 0.0) for cls in CLASS_NAMES}
        df_durations = pd.DataFrame([
            {"Behavior Class": k.replace("_", " ").title(), "Abnormal Duration (s)": v}
            for k, v in full_durations.items()
        ])
        st.bar_chart(
            data=df_durations,
            x="Behavior Class",
            y="Abnormal Duration (s)",
            use_container_width=True
        )


def render_model_comparison_section(
    per_model_reports: Dict[str, Dict[str, Any]],
    yolo_status: Optional[Dict[str, Any]] = None
):
    """Renders model-wise comparison table and bar chart showing detections per model and YOLOv5 branch status."""
    st.subheader("🤖 Benchmark Model-Wise Detections & Branch Comparison")

    comparison_rows = []
    for model_name, report in per_model_reports.items():
        stats = report.get("summary_statistics", {})
        counts = stats.get("class_wise_counts", {})
        total_alerts = stats.get("total_flagged_segments", 0)
        top_cls = max(counts, key=counts.get) if counts and max(counts.values()) > 0 else "None"

        comparison_rows.append({
            "Benchmark Model / Branch": model_name.replace("_", " ").title(),
            "Branch Architecture": "5-Class Classification",
            "Checkpoint Status": "Trained Checkpoint Active",
            "Flagged Segments": total_alerts,
            "Primary Detected Class": top_cls.replace("_", " ").title() if top_cls != "None" else "Clean",
            "Flagged Time": f"{stats.get('total_flagged_time_sec', 0.0):.1f}s"
        })

    # Add YOLOv5 Object Detection Branch Status Row
    yolo_ckpt = Path("weights") / "yolov5_best.pt"
    yolo_status_str = "Trained Checkpoint Active" if yolo_ckpt.exists() else "Untrained / Checkpoint Unavailable"
    yolo_alerts = yolo_status.get("total_flagged_segments", 0) if (yolo_status and yolo_ckpt.exists()) else 0

    comparison_rows.append({
        "Benchmark Model / Branch": "YOLOv5 (Object Detection)",
        "Branch Architecture": "Bounding Box Localization",
        "Checkpoint Status": yolo_status_str,
        "Flagged Segments": yolo_alerts if yolo_ckpt.exists() else 0,
        "Primary Detected Class": "Bounding Boxes" if yolo_ckpt.exists() else "N/A (Ready for Training)",
        "Flagged Time": f"{yolo_status.get('total_flagged_time_sec', 0.0):.1f}s" if (yolo_status and yolo_ckpt.exists()) else "N/A"
    })

    df_comp = pd.DataFrame(comparison_rows)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # Bar chart comparing detected alerts per model
    df_chart = pd.DataFrame([
        {"Model Name": row["Benchmark Model / Branch"], "Flagged Segments": row["Flagged Segments"]}
        for row in comparison_rows
    ])
    st.bar_chart(df_chart, x="Model Name", y="Flagged Segments", use_container_width=True)

    if not yolo_ckpt.exists():
        st.info(
            "ℹ️ **YOLOv5 Branch Note**: Trained weights `weights/yolov5_best.pt` for the project's 5 behavior classes are currently unavailable. "
            "The YOLOv5 object detection branch, custom dataset configuration (`data/yolov5/data.yaml`), and `YOLOv5Detector` class remain "
            "fully preserved in the codebase for future dataset fine-tuning."
        )

