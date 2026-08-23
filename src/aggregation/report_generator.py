"""
Report Generator Module for Online Exam Proctoring Pipeline.
Compiles session-level abnormal activity timelines and exports structured JSON and CSV reports.
As specified in project_report.pdf (Ramzan et al., 2024).
"""

import json
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import CLASS_NAMES, RESULTS_DIR
from src.aggregation.sliding_window import FlaggedSegmentAlert
from src.utils.logger import get_logger

logger = get_logger("report_generator")


class ReportGenerator:
    """Compiles session-level flagged activity alerts into exportable JSON and CSV reports."""

    def __init__(self, output_dir: Path = RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_session_report(
        self,
        video_name: str,
        total_duration_sec: float,
        total_frames: int,
        key_frames_analyzed: int,
        model_name: str,
        alerts: List[FlaggedSegmentAlert],
        json_filename: str = "session_report.json",
        csv_filename: str = "session_report.csv"
    ) -> Dict[str, Any]:
        """
        Compiles structured session summary and exports both JSON and CSV files.
        """
        class_wise_counts: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
        class_wise_duration_sec: Dict[str, float] = {cls: 0.0 for cls in CLASS_NAMES}
        total_flagged_time = 0.0

        for alert in alerts:
            if alert.predicted_class not in class_wise_counts:
                class_wise_counts[alert.predicted_class] = 0
                class_wise_duration_sec[alert.predicted_class] = 0.0
            class_wise_counts[alert.predicted_class] += 1
            class_wise_duration_sec[alert.predicted_class] += alert.duration_sec
            total_flagged_time += alert.duration_sec

        for cls in class_wise_duration_sec:
            class_wise_duration_sec[cls] = round(class_wise_duration_sec[cls], 2)

        timeline = []
        for idx, alert in enumerate(alerts, start=1):
            entry = {
                "alert_id": idx,
                "start_time": self._format_timestamp(alert.start_time_sec),
                "end_time": self._format_timestamp(alert.end_time_sec),
                "start_time_sec": alert.start_time_sec,
                "end_time_sec": alert.end_time_sec,
                "duration_sec": alert.duration_sec,
                "predicted_class": alert.predicted_class,
                "peak_confidence": alert.peak_confidence,
                "average_confidence": alert.average_confidence,
                "key_frame_count": alert.key_frame_count
            }
            if hasattr(alert, "agreeing_models"):
                entry["agreeing_models"] = ", ".join(alert.agreeing_models)
                entry["num_agreeing_models"] = alert.num_agreeing_models
            timeline.append(entry)

        report_data = {
            "session_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "video_name": video_name,
                "model_name": model_name,
                "total_duration_sec": round(total_duration_sec, 2),
                "total_duration_formatted": self._format_timestamp(total_duration_sec),
                "total_frames": total_frames,
                "key_frames_analyzed": key_frames_analyzed,
                "key_frame_reduction_percent": round(
                    (1.0 - (key_frames_analyzed / max(1, total_frames))) * 100.0, 2
                )
            },
            "summary_statistics": {
                "total_flagged_segments": len(alerts),
                "total_flagged_time_sec": round(total_flagged_time, 2),
                "class_wise_counts": class_wise_counts,
                "class_wise_duration_sec": class_wise_duration_sec
            },
            "timeline": timeline
        }

        # Export JSON
        json_path = self.output_dir / json_filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved session JSON report to {json_path}")

        # Export CSV
        csv_path = self.output_dir / csv_filename
        self._export_csv(timeline, csv_path)
        logger.info(f"Saved session CSV report to {csv_path}")

        return report_data

    def export_all_formats(
        self,
        consensus_report: Dict[str, Any],
        per_model_reports: Optional[Dict[str, Any]] = None,
        json_filename: str = "consensus_session_report.json",
        csv_filename: str = "consensus_session_report.csv"
    ) -> Dict[str, Any]:
        """
        Exports both JSON and CSV files for the consensus report session.
        """
        json_path = self.output_dir / json_filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(consensus_report, f, indent=2)
        logger.info(f"Saved consensus JSON report to {json_path}")

        csv_path = self.output_dir / csv_filename
        timeline = consensus_report.get("timeline", [])
        self._export_csv(timeline, csv_path)
        logger.info(f"Saved consensus CSV report to {csv_path}")

        return consensus_report

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Formats seconds float to HH:MM:SS format."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    @staticmethod
    def _export_csv(timeline: List[Dict[str, Any]], csv_path: Path):
        """Exports timeline array to CSV file."""
        if not timeline:
            fieldnames = [
                "alert_id", "start_time", "end_time", "start_time_sec",
                "end_time_sec", "duration_sec", "predicted_class",
                "peak_confidence", "average_confidence", "key_frame_count"
            ]
        else:
            fieldnames = list(timeline[0].keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in timeline:
                writer.writerow(row)
