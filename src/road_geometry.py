"""
RoadWatch AI - Milestone 4: Road Geometry & Relative Position
------------------------------------------------------------
This module extracts spatial and geometric properties from tracked vehicles:
- Coordinate mapping and ground-contact estimation
- Lateral lane zoning (Left Zone, Ego Lane, Right Zone)
- Monocular distance proxies (box height ratio and ground-plane Y position)
- Movement trend estimation (Approaching, Receding, Stable)
- Lateral movement direction (Moving Left, Moving Right, Stable)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class RoadGeometryAnalyzer:
    """
    Analyzes spatial geometry and relative movement of vehicles from monocular camera frames.
    """

    def __init__(
        self,
        frame_width: int = 854,
        frame_height: int = 480,
        ego_lane_bounds: Tuple[float, float] = (0.35, 0.65),
        history_window: int = 8,
    ):
        """
        Args:
            frame_width: Image width in pixels.
            frame_height: Image height in pixels.
            ego_lane_bounds: Horizontal fractions (min, max) defining the rider's direct travel corridor.
            history_window: Number of recent frames used to compute approach/lateral velocity trends.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ego_min_x = ego_lane_bounds[0] * frame_width
        self.ego_max_x = ego_lane_bounds[1] * frame_width
        self.history_window = history_window

        # Rolling track history for geometric features: track_id -> list of snapshots
        # Each snapshot: {'frame_idx': int, 'cx': int, 'cy': int, 'y2': int, 'h': int, 'w': int}
        self.track_geometry_history: Dict[int, List[Dict[str, Any]]] = {}

    def get_lateral_zone(self, cx: float) -> str:
        """
        Determines the lateral position relative to the rider's camera.
        """
        if cx < self.ego_min_x:
            return "LEFT_ZONE"
        elif cx > self.ego_max_x:
            return "RIGHT_ZONE"
        else:
            return "EGO_LANE"

    def estimate_proximity_band(self, bbox: List[int]) -> Tuple[str, float]:
        """
        Computes a normalized proximity proxy [0.0 - 1.0] and categorizes distance band.
        
        Note: On monocular cameras, true physical distance in meters requires 3D calibration.
        We use an optical proxy combining:
        1. Ground contact Y-coordinate (lower down in frame = closer to camera).
        2. Bounding box height ratio (larger height in pixels = closer).
        """
        x1, y1, x2, y2 = bbox
        box_h = max(1, y2 - y1)

        # Fraction of image height occupied by the vehicle
        h_ratio = box_h / self.frame_height

        # Position of tires/bottom on ground plane relative to horizon
        ground_y_ratio = y2 / self.frame_height

        # Blended proximity proxy (0.0 = very far / horizon, 1.0 = right at rider wheel)
        proximity_score = min(1.0, (0.5 * h_ratio + 0.5 * max(0.0, (ground_y_ratio - 0.4) / 0.6)))

        if proximity_score > 0.65 or ground_y_ratio > 0.85:
            band = "VERY_CLOSE"
        elif proximity_score > 0.40 or ground_y_ratio > 0.68:
            band = "CLOSE"
        elif proximity_score > 0.20 or ground_y_ratio > 0.50:
            band = "MEDIUM"
        else:
            band = "FAR"

        return band, round(proximity_score, 3)

    def estimate_movement_trends(self, track_id: int) -> Tuple[str, str]:
        """
        Analyzes change in bounding box geometry over recent frames to determine:
        - Longitudinal movement: 'APPROACHING', 'RECEDING', or 'STABLE'
        - Lateral movement: 'MOVING_LEFT', 'MOVING_RIGHT', or 'LATERAL_STABLE'
        """
        history = self.track_geometry_history.get(track_id, [])
        if len(history) < 3:
            return "STABLE", "LATERAL_STABLE"

        recent = history[-self.history_window :]
        first_snap = recent[0]
        last_snap = recent[-1]
        delta_frames = max(1, last_snap["frame_idx"] - first_snap["frame_idx"])

        # Height growth rate (pixels per frame)
        delta_h = (last_snap["h"] - first_snap["h"]) / delta_frames
        # Ground plane descent rate (pixels per frame down toward camera)
        delta_y2 = (last_snap["y2"] - first_snap["y2"]) / delta_frames

        # Lateral shift rate (pixels per frame)
        delta_cx = (last_snap["cx"] - first_snap["cx"]) / delta_frames

        # Longitudinal trend: A vehicle closing in expands in height and shifts downward
        if delta_h > 0.35 or delta_y2 > 0.50:
            longitudinal = "APPROACHING"
        elif delta_h < -0.35 or delta_y2 < -0.50:
            longitudinal = "RECEDING"
        else:
            longitudinal = "STABLE"

        # Lateral trend
        if delta_cx > 0.75:
            lateral = "MOVING_RIGHT"
        elif delta_cx < -0.75:
            lateral = "MOVING_LEFT"
        else:
            lateral = "LATERAL_STABLE"

        return longitudinal, lateral

    def analyze_vehicle(
        self,
        frame_idx: int,
        vehicle_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Computes spatial and geometric attributes for a single tracked vehicle.
        """
        track_id = vehicle_record.get("track_id")
        bbox = vehicle_record["bbox"]
        x1, y1, x2, y2 = bbox
        cx, cy = vehicle_record["center"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Update historical geometry for this track ID
        if track_id is not None:
            if track_id not in self.track_geometry_history:
                self.track_geometry_history[track_id] = []
            self.track_geometry_history[track_id].append(
                {"frame_idx": frame_idx, "cx": cx, "cy": cy, "y2": y2, "h": h, "w": w}
            )
            # Prune older records to avoid unbounded memory growth
            if len(self.track_geometry_history[track_id]) > 50:
                self.track_geometry_history[track_id].pop(0)

        zone = self.get_lateral_zone(cx)
        proximity_band, proximity_score = self.estimate_proximity_band(bbox)

        longitudinal_trend, lateral_trend = (
            self.estimate_movement_trends(track_id) if track_id is not None else ("STABLE", "LATERAL_STABLE")
        )

        lateral_offset_ratio = round((cx - (self.frame_width / 2)) / (self.frame_width / 2), 3)

        return {
            "vehicle_id": track_id,
            "class_name": vehicle_record["class_name"],
            "confidence": vehicle_record["confidence"],
            "bbox": bbox,
            "ground_contact_point": (cx, y2),
            "lateral_zone": zone,
            "lateral_offset_ratio": lateral_offset_ratio,
            "proximity_band": proximity_band,
            "proximity_score": proximity_score,
            "longitudinal_movement": longitudinal_trend,
            "lateral_movement": lateral_trend,
        }
