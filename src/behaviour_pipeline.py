"""
RoadWatch AI - Milestone 5: Behavioural Feature Pipeline & Exporter
-------------------------------------------------------------------
Integrates Tracking, Road Geometry, and Behavioural Feature Extraction:
- Computes numerical vectors for each tracked vehicle over time
- Overlays real-time behavioral metrics (Approach Rate, TTC, Lateral velocity, Cut alert)
- Exports complete numerical dataset to CSV for machine-learning model training
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


class BehaviourPipeline:
    def __init__(
        self,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.tracker = VehicleTracker(conf_threshold=conf_threshold, device=device)
        self.geometry_analyzer = RoadGeometryAnalyzer()
        self.feature_extractor: Optional[BehaviouralFeatureExtractor] = None

    def process_video(
        self,
        input_path: str,
        output_video_path: str,
        output_csv_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes video stream and produces behavioural feature dataset + annotated video.
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

        feature_records: List[Dict[str, Any]] = []
        frame_idx = 0
        start_time = time.time()

        print(f"[BehaviourPipeline] Processing {limit_frames} frames -> '{output_video_path}'")

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, tracks = self.tracker.track_frame(frame, persist=True)
            frame_idx += 1

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
                feature_records.append(features)

                # Render rich behavioural tags on vehicle
                x1, y1, x2, y2 = t["bbox"]
                v_long = features["v_long_proxy"]
                v_lat = features["v_lat_proxy"]
                app_rate = features["approach_rate"]
                ttc = features["ttc_proxy_sec"]
                is_cut = features["is_cutting_lane"]

                # Determine HUD color
                if ttc < 3.0 or is_cut:
                    hud_border_color = (0, 0, 255)  # Red for high dynamics
                elif app_rate > 0.15:
                    hud_border_color = (0, 165, 255)  # Orange for rapid approach
                else:
                    hud_border_color = (0, 255, 100)  # Green for calm/stable

                # Multi-line behavioural HUD
                line1 = f"ID #{track_id} {t['class_name'].upper()}"
                line2 = f"AppRate: {app_rate:+.2f}/s | V_lat: {v_lat:+.0f}px/s"
                ttc_str = f"TTC: {ttc:.1f}s" if ttc < 90 else "TTC: SAFE"
                cut_str = " | CUTTING!" if is_cut else ""
                line3 = f"{ttc_str}{cut_str}"

                (w1, h1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                (w2, h2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                (w3, h3), _ = cv2.getTextSize(line3, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                box_w = max(w1, w2, w3) + 8
                box_h = h1 + h2 + h3 + 12

                tag_y1 = max(0, y1 - box_h - 4)
                cv2.rectangle(annotated_frame, (x1, tag_y1), (x1 + box_w, y1), (25, 25, 25), -1)
                cv2.rectangle(annotated_frame, (x1, tag_y1), (x1 + box_w, y1), hud_border_color, 1)

                cv2.putText(
                    annotated_frame, line1, (x1 + 4, tag_y1 + h1 + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame, line2, (x1 + 4, tag_y1 + h1 + h2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame, line3, (x1 + 4, tag_y1 + h1 + h2 + h3 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, hud_border_color, 1, cv2.LINE_AA
                )

            # Top screen stats banner
            header = f"Frame {frame_idx}/{limit_frames} | Features Logged: {len(feature_records)}"
            cv2.putText(
                annotated_frame, header, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA
            )

            out.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == limit_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Extracted {frame_idx}/{limit_frames} frames ({curr_fps:.1f} FPS)...")

        total_elapsed = time.time() - start_time
        avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        cap.release()
        out.release()

        # Save to CSV using pandas
        if feature_records:
            df = pd.DataFrame(feature_records)
            df.to_csv(output_csv_path, index=False)

        metrics = {
            "processed_frames": frame_idx,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "total_feature_rows": len(feature_records),
            "output_video": output_video_path,
            "output_csv": output_csv_path,
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Behavioural Feature Pipeline Summary")
        print("=" * 60)
        print(f"Frames Processed : {frame_idx}")
        print(f"Extraction Speed : {avg_fps:.1f} FPS (MPS Accelerated)")
        print(f"Feature Rows     : {len(feature_records)} numerical samples")
        print(f"Annotated Video  : {output_video_path}")
        print(f"Feature Dataset  : {output_csv_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    in_video = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    out_video = "outputs/annotated_behaviour_sample.mp4"
    out_csv = "outputs/behavioural_features.csv"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    pipeline = BehaviourPipeline(conf_threshold=0.25)
    pipeline.process_video(in_video, out_video, out_csv, max_frames=limit)
