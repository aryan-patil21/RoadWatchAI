"""
RoadWatch AI - Milestone 6: End-to-End Risk Estimation Pipeline
--------------------------------------------------------------
Connects the full pipeline:
Camera Frame -> YOLO -> ByteTrack -> Geometry -> Behavioural Features -> Risk Engine
Overlays dynamic safety alerts, color-coded risk bounding boxes, and rider HUD.
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional
import cv2
import pandas as pd

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vehicle_tracker import VehicleTracker
from src.road_geometry import RoadGeometryAnalyzer
from src.feature_extractor import BehaviouralFeatureExtractor
from src.risk_engine import RiskEngine

RISK_COLORS = {
    "LOW": (0, 230, 115),     # Vibrant Green
    "MEDIUM": (0, 165, 255),  # Amber / Orange
    "HIGH": (0, 0, 255),      # Bright Red
}


class RiskPipeline:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.tracker = VehicleTracker(conf_threshold=conf_threshold, device=device)
        self.geometry_analyzer = RoadGeometryAnalyzer()
        self.feature_extractor: Optional[BehaviouralFeatureExtractor] = None
        self.risk_engine = RiskEngine()

    def process_video(
        self,
        input_path: str,
        output_video_path: str,
        output_csv_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes video frames and overlays dynamic risk estimation.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit_frames = min(total_frames, max_frames) if max_frames else total_frames

        self.geometry_analyzer.frame_width = width
        self.geometry_analyzer.frame_height = height
        self.geometry_analyzer.ego_min_x = 0.35 * width
        self.geometry_analyzer.ego_max_x = 0.65 * width

        self.feature_extractor = BehaviouralFeatureExtractor(
            fps=fps, frame_width=width, frame_height=height, window_size=8
        )

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        risk_log: List[Dict[str, Any]] = []
        frame_idx = 0
        start_time = time.time()

        print(f"[RiskPipeline] Processing {limit_frames} frames -> '{output_video_path}'")

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, tracks = self.tracker.track_frame(frame, persist=True)
            frame_idx += 1

            max_risk_level = "LOW"
            highest_risk_msg = "ROADWATCH: ROAD SAFE - MAINTAIN DISTANCE"

            for t in tracks:
                track_id = t["track_id"]
                if track_id is None:
                    continue

                spatial_data = self.geometry_analyzer.analyze_vehicle(frame_idx, t)
                proximity = spatial_data["proximity_score"]

                features = self.feature_extractor.extract_features(
                    frame_idx=frame_idx,
                    vehicle_id=track_id,
                    class_name=t["class_name"],
                    bbox=t["bbox"],
                    proximity_score=proximity,
                )

                # Evaluate Risk Model
                risk_eval = self.risk_engine.evaluate_risk(features)

                # Merge log
                combined_record = {**features, **risk_eval}
                risk_log.append(combined_record)

                risk_level = risk_eval["risk_level"]
                risk_score = risk_eval["risk_score"]
                color = RISK_COLORS[risk_level]

                if risk_level == "HIGH":
                    max_risk_level = "HIGH"
                    highest_risk_msg = f"DANGER: {risk_eval['alert_message']}"
                elif risk_level == "MEDIUM" and max_risk_level != "HIGH":
                    max_risk_level = "MEDIUM"
                    highest_risk_msg = f"CAUTION: {risk_eval['alert_message']}"

                # Draw risk-coded bounding box
                x1, y1, x2, y2 = t["bbox"]
                box_thickness = 3 if risk_level == "HIGH" else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)

                # Risk badge
                badge_title = f"ID #{track_id} {t['class_name'].upper()} | RISK {risk_score:.0f} ({risk_level})"
                badge_detail = f"Factor: {risk_eval['primary_factor']}"

                (w1, h1), _ = cv2.getTextSize(badge_title, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                (w2, h2), _ = cv2.getTextSize(badge_detail, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
                bw = max(w1, w2) + 8
                bh = h1 + h2 + 10

                by1 = max(0, y1 - bh - 4)
                cv2.rectangle(annotated_frame, (x1, by1), (x1 + bw, y1), (20, 20, 20), -1)
                cv2.rectangle(annotated_frame, (x1, by1), (x1 + bw, y1), color, 1)

                cv2.putText(
                    annotated_frame, badge_title, (x1 + 4, by1 + h1 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame, badge_detail, (x1 + 4, by1 + h1 + h2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA
                )

            # Rider Alert HUD Banner at top of screen
            hud_bg_color = (0, 100, 0) if max_risk_level == "LOW" else (
                (0, 80, 180) if max_risk_level == "MEDIUM" else (0, 0, 160)
            )
            cv2.rectangle(annotated_frame, (0, 0), (width, 36), hud_bg_color, -1)
            hud_icon = "🟢" if max_risk_level == "LOW" else ("⚠️" if max_risk_level == "MEDIUM" else "🚨")
            cv2.putText(
                annotated_frame,
                f"{hud_icon}  {highest_risk_msg}",
                (20, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == limit_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Risk evaluated {frame_idx}/{limit_frames} frames ({curr_fps:.1f} FPS)...")

        total_elapsed = time.time() - start_time
        avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        cap.release()
        out.release()

        # Save risk telemetry
        if risk_log:
            df = pd.DataFrame(risk_log)
            df.to_csv(output_csv_path, index=False)

        metrics = {
            "processed_frames": frame_idx,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_risk_records": len(risk_log),
            "output_video": output_video_path,
            "output_csv": output_csv_path,
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Risk Engine Pipeline Summary")
        print("=" * 60)
        print(f"Frames Processed : {frame_idx}")
        print(f"Processing Speed : {avg_fps:.1f} FPS (Real-time on M4)")
        print(f"Risk Samples     : {len(risk_log)} evaluations")
        print(f"Annotated Video  : {output_video_path}")
        print(f"Risk Telemetry   : {output_csv_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    in_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    out_video = "outputs/annotated_risk_sample.mp4"
    out_csv = "outputs/risk_telemetry.csv"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    pipeline = RiskPipeline(conf_threshold=0.25)
    pipeline.process_video(in_video, out_video, out_csv, max_frames=limit)
