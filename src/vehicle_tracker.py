"""
RoadWatch AI - Milestone 3: Object Tracking
-------------------------------------------
This module transitions RoadWatch from single-frame detection to multi-frame tracking:
- Assigns persistent Tracking IDs (e.g., Vehicle #1, Vehicle #3) across frames
- Uses ByteTrack to match detections across consecutive time steps
- Maintains historical trajectory paths (motion trails) for each tracked vehicle
- Handles temporary occlusions and missed detections
- Outputs annotated video with persistent IDs and trajectory trails
"""

import os
import sys
import time
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Road vehicle classes to track
ROAD_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct colors assigned to persistent vehicle IDs
PALETTE = [
    (0, 230, 115),   # Emerald Green
    (255, 140, 0),   # Neon Orange
    (30, 144, 255),  # Dodger Blue
    (238, 130, 238), # Violet
    (0, 215, 255),   # Golden Yellow
    (255, 105, 180), # Hot Pink
    (127, 255, 0),   # Chartreuse
    (0, 255, 255),   # Cyan
]


def get_id_color(track_id: int) -> Tuple[int, int, int]:
    """Returns a deterministic bright color for a given tracking ID."""
    return PALETTE[track_id % len(PALETTE)]


class VehicleTracker:
    """
    Tracks road vehicles across video frames using YOLOv8 + ByteTrack.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        max_trajectory_len: int = 30,
    ):
        """
        Initializes the Vehicle Tracker.

        Args:
            model_path: Path to YOLO weights.
            conf_threshold: Detection confidence cutoff.
            device: 'mps', 'cpu', or auto-detected.
            max_trajectory_len: Number of historical points to remember for trajectory tails.
        """
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.target_class_ids = list(ROAD_CLASSES.keys())
        self.max_trajectory_len = max_trajectory_len

        # Trajectory history: map of track_id -> deque of (center_x, center_y)
        self.trajectories: Dict[int, deque] = {}

        print(f"[VehicleTracker] Initializing YOLO with ByteTrack on {self.device.upper()}...")
        self.model = YOLO(model_path)
        print(f"[VehicleTracker] Conf threshold: {self.conf_threshold}")

    def track_frame(
        self, frame: np.ndarray, persist: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single frame, updates tracking state, and draws annotations.

        Args:
            frame: Input BGR frame.
            persist: Whether to persist tracks from the previous frame.

        Returns:
            Tuple of:
            - annotated_frame: Frame with drawn bounding boxes, IDs, and trajectory trails.
            - tracked_objects: List of active vehicle records for this frame.
        """
        annotated_frame = frame.copy()
        tracked_objects: List[Dict[str, Any]] = []

        # Run tracker using ByteTrack
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker="bytetrack.yaml",
            classes=self.target_class_ids,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results:
            return annotated_frame, tracked_objects

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return annotated_frame, tracked_objects

        # Extract bounding box values
        xyxy_coords = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # Track IDs assigned by ByteTrack (can be None on unconfirmed tracks)
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(xyxy_coords)
        )

        for i in range(len(xyxy_coords)):
            track_id = int(track_ids[i]) if track_ids[i] is not None else None
            conf = float(confidences[i])
            cls_id = int(class_ids[i])
            cls_name = ROAD_CLASSES.get(cls_id, self.model.names.get(cls_id, f"cls_{cls_id}"))

            x1, y1, x2, y2 = [int(v) for v in xyxy_coords[i]]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            color = get_id_color(track_id) if track_id is not None else (180, 180, 180)

            # Update trajectory memory
            if track_id is not None:
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = deque(maxlen=self.max_trajectory_len)
                self.trajectories[track_id].append((cx, cy))

            record = {
                "track_id": track_id,
                "class_name": cls_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": (cx, cy),
                "width": x2 - x1,
                "height": y2 - y1,
                "history_points": len(self.trajectories.get(track_id, [])) if track_id else 0,
            }
            tracked_objects.append(record)

            # Draw trajectory path (motion trail)
            if track_id is not None and len(self.trajectories[track_id]) > 1:
                pts = list(self.trajectories[track_id])
                for j in range(1, len(pts)):
                    # Fade thickness toward older points
                    thickness = max(1, int(3 * (j / len(pts))))
                    cv2.line(annotated_frame, pts[j - 1], pts[j], color, thickness)

            # Draw vehicle bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw center point marker
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)

            # Header tag: e.g. "ID #1 | TRUCK 0.78"
            id_str = f"ID #{track_id}" if track_id is not None else "DETECTING"
            tag_text = f"{id_str} | {cls_name.upper()} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            banner_y1 = max(0, y1 - th - bl - 4)
            cv2.rectangle(annotated_frame, (x1, banner_y1), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated_frame,
                tag_text,
                (x1 + 3, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated_frame, tracked_objects

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Streams through video, tracks all vehicles across frames, and saves annotated MP4.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit_frames = min(total_frames, max_frames) if max_frames else total_frames

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"[VehicleTracker] Processing '{input_path}' ({limit_frames} frames) -> '{output_path}'")

        unique_ids_seen = set()
        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, tracks = self.track_frame(frame, persist=True)
            frame_idx += 1

            for t in tracks:
                if t["track_id"] is not None:
                    unique_ids_seen.add(t["track_id"])

            # Render top-left HUD
            hud_text = f"Frame {frame_idx}/{limit_frames} | Active: {len(tracks)} | Total Unique IDs: {len(unique_ids_seen)}"
            cv2.putText(
                annotated_frame,
                hud_text,
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            out.write(annotated_frame)

            if frame_idx % 50 == 0 or frame_idx == limit_frames:
                elapsed = time.time() - start_time
                curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Tracked {frame_idx}/{limit_frames} frames ({curr_fps:.1f} FPS)...")

        total_elapsed = time.time() - start_time
        avg_fps = frame_idx / total_elapsed if total_elapsed > 0 else 0

        cap.release()
        out.release()

        metrics = {
            "input_path": input_path,
            "output_path": output_path,
            "processed_frames": frame_idx,
            "elapsed_seconds": round(total_elapsed, 2),
            "average_fps": round(avg_fps, 2),
            "unique_vehicle_ids": sorted(list(unique_ids_seen)),
            "total_unique_vehicles": len(unique_ids_seen),
        }

        print("\n" + "=" * 60)
        print(" RoadWatch AI - Object Tracking Summary")
        print("=" * 60)
        print(f"Frames Processed       : {frame_idx}")
        print(f"Elapsed Time           : {total_elapsed:.2f} s")
        print(f"Tracking Speed         : {avg_fps:.1f} FPS (MPS Accelerated)")
        print(f"Unique Vehicle IDs     : {len(unique_ids_seen)} -> {sorted(list(unique_ids_seen))}")
        print(f"Output Video           : {output_path}")
        print("=" * 60)

        return metrics


if __name__ == "__main__":
    video_input = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample_dashcam.mp4"
    video_output = "outputs/annotated_tracking_sample.mp4"
    frame_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    tracker = VehicleTracker(conf_threshold=0.25)
    tracker.process_video(video_input, video_output, max_frames=frame_limit)
