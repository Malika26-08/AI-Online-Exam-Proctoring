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


def _format_timestamp_sec(seconds: float) -> str:
    """Formats seconds float to HH:MM:SS string format."""
    s = float(seconds)
    hrs = int(s // 3600)
    mins = int((s % 3600) // 60)
    secs = int(s % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def render_kpi_cards(session_meta: Dict[str, Any], summary_stats: Dict[str, Any]):
    """Renders top-level KPI metric cards for session overview."""
    col1, col2, col3, col4 = st.columns(4)

    dur_sec = session_meta.get("total_duration_sec", 0.0)
    fmt_dur = session_meta.get("total_duration_formatted", _format_timestamp_sec(dur_sec))

    with col1:
        st.metric(
            label="⏱️ Total Exam Duration",
            value=fmt_dur
        )

    with col2:
        key_frames = session_meta.get("key_frames_analyzed", 0)
        total_frames = session_meta.get("total_frames", 1)
        reduction = session_meta.get("key_frame_reduction_percent", 0.0)
        st.metric(
            label="🎞️ Key-Frames Processed",
            value=f"{key_frames:,} / {total_frames:,}",
            delta=f"-{reduction:.1f}% redundant frames"
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
        st.info("ℹ️ **No flagged abnormal segments detected** (fewer than 2 of 4 CNN models agreed on any abnormal behavior class).")
        return

    df = pd.DataFrame(timeline)

    # Format table for display safely without assuming alert_id exists
    display_df = pd.DataFrame()

    if "alert_id" in df.columns:
        display_df["Alert #"] = df["alert_id"]
    else:
        display_df["Alert #"] = range(1, len(df) + 1)

    if "start_time" in df.columns:
        display_df["Start Time"] = df["start_time"]
    elif "start_time_sec" in df.columns:
        display_df["Start Time"] = df["start_time_sec"].apply(_format_timestamp_sec)

    if "end_time" in df.columns:
        display_df["End Time"] = df["end_time"]
    elif "end_time_sec" in df.columns:
        display_df["End Time"] = df["end_time_sec"].apply(_format_timestamp_sec)

    if "duration_sec" in df.columns:
        display_df["Duration"] = df["duration_sec"].apply(lambda x: f"{float(x):.1f}s")

    if "predicted_class" in df.columns:
        display_df["Predicted Activity"] = df["predicted_class"].apply(
            lambda x: CLASS_COLOR_MAP.get(str(x), str(x).replace("_", " ").title())
        )

    if "agreeing_models" in df.columns:
        display_df["Agreeing Models"] = df["agreeing_models"].apply(
            lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else str(x)
        )

    if "num_agreeing_models" in df.columns:
        display_df["# Models"] = df["num_agreeing_models"]

    if "evaluation_status" in df.columns:
        display_df["Evaluation Status"] = df["evaluation_status"]

    if "peak_confidence" in df.columns:
        display_df["Peak Confidence"] = df["peak_confidence"].apply(
            lambda x: f"{float(x) * 100:.1f}%" if float(x) <= 1.0 else f"{float(x):.1f}%"
        )

    if "average_confidence" in df.columns:
        display_df["Avg Confidence"] = df["average_confidence"].apply(
            lambda x: f"{float(x) * 100:.1f}%" if float(x) <= 1.0 else f"{float(x):.1f}%"
        )

    if "key_frame_count" in df.columns:
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
    st.markdown("#### 📊 Abnormal Behavior Class Breakdown")

    chart_data = []
    for cls_name in CLASS_NAMES:
        display_label = CLASS_COLOR_MAP.get(cls_name, cls_name.replace("_", " ").title())
        cnt = class_counts.get(cls_name, 0)
        dur = class_durations.get(cls_name, 0.0) if class_durations else 0.0
        chart_data.append({
            "Behavior Class": display_label,
            "Alert Count": cnt,
            "Duration (s)": dur
        })

    chart_df = pd.DataFrame(chart_data)

    tab1, tab2 = st.tabs(["🔢 Alert Counts", "⏱️ Duration (Seconds)"])
    with tab1:
        st.bar_chart(
            chart_df.set_index("Behavior Class")["Alert Count"],
            use_container_width=True
        )
    with tab2:
        st.bar_chart(
            chart_df.set_index("Behavior Class")["Duration (s)"],
            use_container_width=True
        )


def render_model_comparison_section(
    per_model_reports: Dict[str, Dict[str, Any]],
    yolo_status: Optional[Dict[str, Any]] = None
):
    """Renders multi-model comparative performance table across all 4 CNNs + YOLOv5 branch."""
    st.subheader("⚖️ 4-CNN Ensemble + YOLOv5 Multi-Model Evaluation Grid")
    st.caption("Individual model classification output metrics before 2-of-4 consensus temporal voting.")

    rows = []
    for m_name, report in per_model_reports.items():
        summary = report.get("summary_statistics", {})
        rows.append({
            "Model Name": m_name.replace("_", " ").title(),
            "Total Key-Frames": summary.get("total_key_frames", 0),
            "Flagged Abnormal Frames": summary.get("abnormal_key_frames", 0),
            "Abnormal Ratio (%)": f"{summary.get('abnormal_percentage', 0.0):.1f}%",
            "Flagged Segments": len(report.get("timeline", []))
        })

    if yolo_status and yolo_status.get("yolo_available"):
        rows.append({
            "Model Name": "YOLOv5 Detector (Branch)",
            "Total Key-Frames": yolo_status.get("key_frames_with_detections", 0),
            "Flagged Abnormal Frames": yolo_status.get("total_detections", 0),
            "Abnormal Ratio (%)": "Spatial Localizer",
            "Flagged Segments": yolo_status.get("total_detections", 0)
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
