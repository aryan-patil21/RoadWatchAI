"""
RoadWatch AI - Milestone 8: Rider Warning Pipeline & Event Logger
-----------------------------------------------------------------
Integrates Video -> YOLO -> ByteTrack -> Geometry -> Features -> Risk -> WarningEngine.
Renders on-screen motorcycle instrument alert banner and exports structured warning logs.
"""

import os
import sys
import json
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
from src.warning_engine import WarningEngine, AlertLevel, RiderAlert


class WarningPipeline:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        debounce_frames: int = 3,
        cooldown_seconds: float = 3.5,
    ):
        self.tracker = VehicleTracker(conf_threshold=conf_threshold, device=device)
        self.geometry_analyzer = RoadGeometryAnalyzer()
        self.feature_extractor: Optional[BehaviouralFeatureExtractor] = None
        self.risk_engine = RiskEngine()
        self.warning_engine = WarningEngine(
            debounce_frames=debounce_frames,
            cooldown_seconds=cooldown_seconds,
        )

    def process_video(
        self,
        input_path: str,
        output_video_path: str,
        output_json_path: str,
        output_csv_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes video frames and logs debounced rider alerts.
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

        fired_alerts: List[Dict[str, Any]] = []
        frame_idx = 0
        start_time = time.time()

        print(f"[WarningPipeline] Processing {limit_frames} frames -> '{output_video_path}'")

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, tracks = self.tracker.track_frame(frame, persist=True)
            frame_idx += 1
            timestamp_sec = round(frame_idx / fps, 2)

            vehicle_evaluations: List[Dict[str, Any]] = []

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

                risk_eval = self.risk_engine.evaluate_risk(features)
                combined = {**features, **risk_eval, "lateral_zone": spatial_data["lateral_zone"]}
                vehicle_evaluations.append(combined)

                # Draw minimal vehicle tag
                x1, y1, x2, y2 = t["bbox"]
                score = risk_eval["risk_score"]
                color = (0, 0, 255) if score >= 70 else ((0, 165, 255) if score >= 40 else (0, 230, 115))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                tag = f"ID #{track_id} | Risk: {score:.0f}"
                cv2.putText(annotated_frame, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # Pass evaluations to Warning Engine
            new_alert = self.warning_engine.process_frame(frame_idx, timestamp_sec, vehicle_evaluations)
            if new_alert:
                fired_alerts.append(new_alert.to_dict())
                print(f"  [FRAME {frame_idx} | {timestamp_sec:.2f}s] 🔥 FIRED: {new_alert.title} -> {new_alert.message}")

            # Render Instrument Cluster HUD Banner at Top
            active_alert = self.warning_engine.active_display_alert

            if active_alert:
                if active_alert.level == AlertLevel.CRITICAL:
                    bg_color = (0, 0, 180)  # Red
                    icon = "🚨 CRITICAL WARNING"
                else:
                    bg_color = (0, 100, 200)  # Amber
                    icon = "⚠️ CAUTION"

                # Banner Box
                cv2.rectangle(annotated_frame, (0, 0), (width, 55), bg_color, -1)
                dir_arrow = "◀" if active_alert.direction == "left" else (
                    "▶" if active_alert.direction == "right" else "▲"
                )

                hud_header = f"{icon}  [{dir_arrow} {active_alert.direction.upper()}] - {active_alert.title}"
                hud_action = f"ACTION: {active_alert.suggested_action}"

                cv2.putText(annotated_frame, hud_header, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, hud_action, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
            else:
                # Normal Peaceful State
                cv2.rectangle(annotated_frame, (0, 0), (width, 34), (20, 60, 20), -1)
                cv2.putText(
                    annotated_frame,
                    "🟢 ROADWATCH | SYSTEM ARMED | NORMAL DRIVING CONDITIONS",
                    (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (100, 255, 100),
                    1,
                    cv2.LINE_AA,
                )

            out.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == limit_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_idx}/{limit_frames} frames ({curr_fps:.1f} FPS)...")

        total_elapsed = time.time() - start_time
        avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        cap.release()
        out.release()

        # Save Fired Alerts to JSON
        with open(output_json_path, "w") as f:
            json.dump(fired_alerts, f, indent=2)

        # Save to CSV
        if fired_alerts:
            df = pd.DataFrame(fired_alerts)
            df.to_csv(output_csv_path, index=False)
        else:
            pd.DataFrame(columns=["alert_id", "timestamp_sec", "level", "title", "message"]).to_csv(output_csv_path, index=False)

        metrics = {
            "processed_frames": frame_idx,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_alerts_fired": len(fired_alerts),
            "output_video": output_video_path,
            "output_json": output_json_path,
            "output_csv": output_csv_path,
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Warning Engine Pipeline Summary")
        print("=" * 60)
        print(f"Frames Processed : {frame_idx}")
        print(f"Processing Speed : {avg_fps:.1f} FPS")
        print(f"Total Alerts     : {len(fired_alerts)} (Zero alert fatigue spam!)")
        print(f"Output Video     : {output_video_path}")
        print(f"Warning Log JSON : {output_json_path}")
        print(f"Warning Log CSV  : {output_csv_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    in_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    out_video = "outputs/annotated_warning_sample.mp4"
    out_json = "outputs/rider_warnings_log.json"
    out_csv = "outputs/rider_warnings_log.csv"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    pipeline = WarningPipeline(conf_threshold=0.25)
    pipeline.process_video(in_video, out_video, out_json, out_csv, max_frames=limit)
