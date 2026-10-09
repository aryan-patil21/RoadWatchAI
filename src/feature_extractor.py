"""
RoadWatch AI - Milestone 5: Behavioural Feature Extraction
---------------------------------------------------------
This module transforms spatial tracking data across time into rich numerical
behavioural feature vectors suitable for machine-learning and risk analysis:
- Longitudinal velocity proxy (closing / opening rate)
- Lateral velocity proxy (lane cutting / swerving)
- Approach rate (rate of change of optical proximity)
- Acceleration proxy (sudden braking or acceleration)
- Trajectory direction angle (degrees)
- Time-to-Collision proxy (TTC in seconds)
- Ego-corridor trajectory conflict indicator
"""

import math
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BehaviouralFeatureExtractor:
    """
    Extracts numerical motion and behavioural features from vehicle tracking snapshots.
    """

    def __init__(
        self,
        fps: float = 25.0,
        frame_width: int = 854,
        frame_height: int = 480,
        window_size: int = 8,
    ):
        """
        Args:
            fps: Frame rate of the video (used for delta time dt calculation).
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            window_size: Number of frames in rolling window for derivative calculation.
        """
        self.fps = fps
        self.dt = 1.0 / fps if fps > 0 else 0.04
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.window_size = window_size

        # Ego lane boundaries (35% to 65% of screen width)
        self.ego_min_x = 0.35 * frame_width
        self.ego_max_x = 0.65 * frame_width

        # Rolling history per track_id:
        # deque of dicts: {'timestamp': float, 'cx': float, 'cy': float, 'y2': float, 'proximity': float, 'v_long': float}
        self.history: Dict[int, deque] = {}

    def extract_features(
        self,
        frame_idx: int,
        vehicle_id: int,
        class_name: str,
        bbox: List[int],
        proximity_score: float,
    ) -> Dict[str, Any]:
        """
        Calculates behavioural feature vector for a vehicle at the current frame.

        Args:
            frame_idx: Current frame index.
            vehicle_id: Persistent tracking ID.
            class_name: Detected vehicle class ('car', 'truck', etc.).
            bbox: Bounding box [x1, y1, x2, y2].
            proximity_score: Optical proximity score [0.0 - 1.0] from Milestone 4.

        Returns:
            Dictionary containing numerical behavioural features.
        """
        timestamp = round(frame_idx * self.dt, 3)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        if vehicle_id not in self.history:
            self.history[vehicle_id] = deque(maxlen=self.window_size * 2)

        # Base snapshot
        snapshot = {
            "timestamp": timestamp,
            "cx": cx,
            "cy": cy,
            "y2": y2,
            "box_h": box_h,
            "proximity": proximity_score,
            "v_long": 0.0,
            "v_lat": 0.0,
        }

        hist = self.history[vehicle_id]
        hist.append(snapshot)

        # Default feature values when observation history is short (< 3 frames)
        v_long = 0.0
        v_lat = 0.0
        approach_rate = 0.0
        acceleration_proxy = 0.0
        trajectory_angle = 0.0
        ttc_proxy = 99.9  # 99.9s represents no imminent collision
        is_cutting_lane = 0
        sudden_braking_flag = 0

        if len(hist) >= 3:
            # Look back across window_size frames or all available
            idx_start = max(0, len(hist) - self.window_size)
            old_snap = hist[idx_start]
            dt_span = max(self.dt, snapshot["timestamp"] - old_snap["timestamp"])

            # 1. Longitudinal Velocity Proxy (pixels/sec down toward rider)
            v_long = (snapshot["y2"] - old_snap["y2"]) / dt_span

            # 2. Lateral Velocity Proxy (pixels/sec horizontally)
            v_lat = (snapshot["cx"] - old_snap["cx"]) / dt_span

            # 3. Approach Rate (rate of change of optical proximity per second)
            approach_rate = (snapshot["proximity"] - old_snap["proximity"]) / dt_span

            # 4. Trajectory Heading Angle in 2D image coordinates (degrees: -180 to +180)
            dx = snapshot["cx"] - old_snap["cx"]
            dy = snapshot["y2"] - old_snap["y2"]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                trajectory_angle = math.degrees(math.atan2(dy, dx))

            # Store computed velocities in current snapshot for acceleration calculation
            snapshot["v_long"] = v_long
            snapshot["v_lat"] = v_lat

            # 5. Longitudinal Acceleration Proxy (rate of change of v_long: pixels/sec^2)
            if len(hist) >= 5:
                mid_snap = hist[len(hist) // 2]
                dt_mid = max(self.dt, snapshot["timestamp"] - mid_snap["timestamp"])
                acceleration_proxy = (v_long - mid_snap["v_long"]) / dt_mid

            # 6. Time-To-Collision Proxy (TTC)
            # If approaching rapidly and proximity is significant:
            if approach_rate > 0.05 and proximity_score > 0.15:
                # Time in seconds until proximity reaches 1.0 (contact)
                remaining_headroom = max(0.01, 1.0 - proximity_score)
                ttc_proxy = min(99.9, round(remaining_headroom / approach_rate, 2))

            # 7. Aggressive Lane Cutting Indicator
            # High lateral speed moving into or across ego corridor
            # CRITICAL: Far-away horizon objects (proximity < 0.35) have natural perspective drift and must NOT be flagged as lane cutting
            in_ego_lane = self.ego_min_x <= cx <= self.ego_max_x
            moving_toward_ego = (cx < self.ego_min_x and v_lat > 35.0) or (
                cx > self.ego_max_x and v_lat < -35.0
            )
            if abs(v_lat) > 40.0 and (in_ego_lane or moving_toward_ego) and proximity_score > 0.35:
                is_cutting_lane = 1

            # 8. Sudden Braking / Rapid Deceleration Indicator
            # Vehicle in front decelerates sharply (v_long drops or acceleration is heavily negative)
            if acceleration_proxy < -120.0 and proximity_score > 0.30:
                sudden_braking_flag = 1

        # Trajectory Conflict: Vehicle is in our travel lane or trajectory intersects it
        in_ego_corridor = 1 if (self.ego_min_x <= cx <= self.ego_max_x) else 0

        feature_vector = {
            "vehicle_id": vehicle_id,
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp,
            "class_name": class_name,
            "cx": round(cx, 1),
            "cy": round(cy, 1),
            "ground_y": round(y2, 1),
            "box_width": int(box_w),
            "box_height": int(box_h),
            "proximity_score": round(proximity_score, 3),
            "v_long_proxy": round(v_long, 2),
            "v_lat_proxy": round(v_lat, 2),
            "approach_rate": round(approach_rate, 3),
            "acceleration_proxy": round(acceleration_proxy, 2),
            "trajectory_angle_deg": round(trajectory_angle, 1),
            "ttc_proxy_sec": ttc_proxy,
            "in_ego_corridor": in_ego_corridor,
            "is_cutting_lane": is_cutting_lane,
            "sudden_braking_flag": sudden_braking_flag,
        }

        return feature_vector
