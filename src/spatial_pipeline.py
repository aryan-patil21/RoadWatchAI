"""
RoadWatch AI - Milestone 4: Spatial Pipeline & Telemetry Exporter
----------------------------------------------------------------
Combines VehicleTracker with RoadGeometryAnalyzer:
- Overlays rider lane corridor (Ego Lane vs Left/Right zones)
- Visualizes proximity bands and movement direction (Approaching/Receding/Lateral)
- Exports structured spatial telemetry to JSON and CSV
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

from src.vehicle_tracker import VehicleTracker, get_id_color
from src.road_geometry import RoadGeometryAnalyzer


PROXIMITY_COLORS = {
    "VERY_CLOSE": (0, 0, 255),    # Red
    "CLOSE": (0, 140, 255),       # Orange
    "MEDIUM": (0, 255, 255),      # Yellow
    "FAR": (0, 255, 100),         # Green
}


class SpatialPipeline:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.tracker = VehicleTracker(conf_threshold=conf_threshold, device=device)
        self.geometry_analyzer = RoadGeometryAnalyzer()

    def process_video(
        self,
        input_path: str,
        output_video_path: str,
        output_json_path: str,
        output_csv_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes video frames, extracts spatial telemetry, and outputs video + datasets.
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

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        telemetry_log: List[Dict[str, Any]] = []
        frame_idx = 0
        start_time = time.time()

        print(f"[SpatialPipeline] Processing {limit_frames} frames -> '{output_video_path}'")

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, tracks = self.tracker.track_frame(frame, persist=True)
            frame_idx += 1

            # Draw subtle ego-lane guidelines (dashed/translucent corridor lines)
            ego_x1 = int(self.geometry_analyzer.ego_min_x)
            ego_x2 = int(self.geometry_analyzer.ego_max_x)
            cv2.line(annotated_frame, (ego_x1, height), (ego_x1, int(height * 0.45)), (100, 200, 100), 1, cv2.LINE_AA)
            cv2.line(annotated_frame, (ego_x2, height), (ego_x2, int(height * 0.45)), (100, 200, 100), 1, cv2.LINE_AA)
            cv2.putText(
                annotated_frame,
                "EGO TRAVEL CORRIDOR",
                (ego_x1 + 10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 220, 100),
                1,
                cv2.LINE_AA,
            )

            # Analyze spatial metrics for each tracked vehicle
            for t in tracks:
                spatial_data = self.geometry_analyzer.analyze_vehicle(frame_idx, t)
                spatial_data["frame_idx"] = frame_idx
                spatial_data["timestamp_sec"] = round(frame_idx / fps, 2)
                telemetry_log.append(spatial_data)

                # Render rich spatial tag
                track_id = spatial_data["vehicle_id"]
                bbox = spatial_data["bbox"]
                x1, y1, x2, y2 = bbox
                prox_band = spatial_data["proximity_band"]
                long_move = spatial_data["longitudinal_movement"]
                lat_zone = spatial_data["lateral_zone"]

                tag_color = PROXIMITY_COLORS.get(prox_band, (0, 255, 0))

                # Movement indicator icon
                move_icon = ">>" if long_move == "APPROACHING" else ("<<" if long_move == "RECEDING" else "==")
                hud_line1 = f"ID #{track_id} | {spatial_data['class_name'].upper()}"
                hud_line2 = f"{lat_zone} | {prox_band} ({move_icon})"

                # Draw bottom ground-contact crosshair
                cx, gy = spatial_data["ground_contact_point"]
                cv2.drawMarker(annotated_frame, (cx, gy), tag_color, cv2.MARKER_TILTED_CROSS, 8, 2)

                # Overlay spatial info box
                (w1, h1), _ = cv2.getTextSize(hud_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                (w2, h2), _ = cv2.getTextSize(hud_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tag_w = max(w1, w2) + 8
                tag_h = h1 + h2 + 10

                tag_y1 = max(0, y1 - tag_h - 4)
                cv2.rectangle(annotated_frame, (x1, tag_y1), (x1 + tag_w, y1), (20, 20, 20), -1)
                cv2.rectangle(annotated_frame, (x1, tag_y1), (x1 + tag_w, y1), tag_color, 1)

                cv2.putText(
                    annotated_frame,
                    hud_line1,
                    (x1 + 4, tag_y1 + h1 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    hud_line2,
                    (x1 + 4, tag_y1 + h1 + h2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    tag_color,
                    1,
                    cv2.LINE_AA,
                )

            # Top telemetry banner
            status_text = f"Frame {frame_idx}/{limit_frames} | Tracked Vehicles: {len(tracks)}"
            cv2.putText(
                annotated_frame,
                status_text,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
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

        # Save telemetry to JSON
        with open(output_json_path, "w") as f:
            json.dump(telemetry_log, f, indent=2)

        # Save telemetry to CSV
        if telemetry_log:
            df = pd.DataFrame(telemetry_log)
            # Flatten ground_contact_point tuple
            if "ground_contact_point" in df.columns:
                df["ground_x"] = df["ground_contact_point"].apply(lambda p: p[0])
                df["ground_y"] = df["ground_contact_point"].apply(lambda p: p[1])
                df.drop(columns=["ground_contact_point"], inplace=True)
            df.to_csv(output_csv_path, index=False)

        metrics = {
            "processed_frames": frame_idx,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_records": len(telemetry_log),
            "output_video": output_video_path,
            "output_json": output_json_path,
            "output_csv": output_csv_path,
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Spatial Pipeline Summary")
        print("=" * 60)
        print(f"Frames Processed : {frame_idx}")
        print(f"Average FPS      : {avg_fps:.1f} FPS")
        print(f"Spatial Records  : {len(telemetry_log)}")
        print(f"Output Video     : {output_video_path}")
        print(f"Telemetry JSON   : {output_json_path}")
        print(f"Telemetry CSV    : {output_csv_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    video_input = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    out_video = "outputs/annotated_geometry_sample.mp4"
    out_json = "outputs/spatial_telemetry_sample.json"
    out_csv = "outputs/spatial_telemetry_sample.csv"
    frame_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    pipeline = SpatialPipeline(conf_threshold=0.25)
    pipeline.process_video(video_input, out_video, out_json, out_csv, max_frames=frame_limit)
