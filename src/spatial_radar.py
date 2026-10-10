"""
RoadWatch AI - Metric Projection & Spatial Radar Module (Milestone 13)
---------------------------------------------------------------------
Computes:
1. Metric ground-plane coordinates (X_meters lateral, Z_meters longitudinal)
   from 2D image coordinates using Inverse Perspective Mapping (IPM).
2. True relative approach/recede speeds in km/h.
3. Renders a top-down 2D Bird's-Eye View (BEV) radar widget onto the HUD.
"""

from typing import Tuple, Dict, Any, List, Optional
import cv2
import numpy as np


class MetricGroundProjector:
    """
    Inverse Perspective Mapping (IPM) model that projects 2D image ground-contact
    points (bottom-center of bounding boxes) into real-world physical metric coordinates (X, Z)
    where:
    - Z is longitudinal distance ahead in meters (0m at rider's front wheel, up to 60m).
    - X is lateral offset in meters (0m at rider center, negative = left, positive = right).
    """

    def __init__(
        self,
        frame_width: int = 854,
        frame_height: int = 480,
        camera_height_m: float = 1.25,     # Motorcycle handlebar/dashcam height
        pitch_angle_deg: float = 6.0,      # Slight downward tilt of dashcam
        focal_length_px: float = 720.0,    # Typical horizontal focal length for ~65 deg FOV
        horizon_y: float = 265.0,          # Vanishing horizon in pixels
        fps: float = 25.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.camera_height_m = camera_height_m
        self.pitch_rad = np.radians(pitch_angle_deg)
        self.focal_length = focal_length_px
        self.horizon_y = horizon_y
        self.fps = fps

        # Track history for metric speed calculation: track_id -> [(timestamp_sec, X_m, Z_m)]
        self.metric_track_history: Dict[int, List[Tuple[float, float, float]]] = {}

    def project_point_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """
        Projects an image coordinate (u, v) onto ground plane (X_m, Z_m).
        u: pixel x (0 to frame_width)
        v: pixel y (0 to frame_height, where v > horizon_y)
        Returns:
            (X_m, Z_m) in meters.
        """
        # If point is above or near horizon, clamp to far distance
        delta_v = v - self.horizon_y
        if delta_v <= 6.0:
            return (0.0, 75.0)

        # Standard Inverse Perspective Mapping (IPM) using vanishing horizon line:
        # Z = (camera_height_m * focal_length) / (v - horizon_y)
        z_m = (self.camera_height_m * self.focal_length) / delta_v
        z_m = max(1.5, min(75.0, z_m))

        # Lateral distance X (meters)
        u_diff = u - (self.frame_width / 2.0)
        x_m = (u_diff * z_m) / self.focal_length

        return round(float(x_m), 2), round(float(z_m), 2)

    def estimate_vehicle_metric_state(
        self,
        track_id: int,
        bbox: List[int],
        timestamp_sec: float,
    ) -> Dict[str, Any]:
        """
        Computes metric position (X, Z) and relative velocity in km/h for a tracked vehicle.
        """
        x1, y1, x2, y2 = bbox
        bottom_cx = (x1 + x2) / 2.0
        bottom_cy = float(y2)

        x_m, z_m = self.project_point_to_ground(bottom_cx, bottom_cy)

        if track_id not in self.metric_track_history:
            self.metric_track_history[track_id] = []

        history = self.metric_track_history[track_id]
        history.append((timestamp_sec, x_m, z_m))

        # Keep last 10 snapshots (~0.4s)
        if len(history) > 10:
            history.pop(0)

        # Compute relative longitudinal speed (km/h)
        # Positive speed = vehicle moving away (receding)
        # Negative speed = vehicle closing in (approaching)
        rel_speed_kmh = 0.0
        if len(history) >= 4:
            dt = history[-1][0] - history[0][0]
            if dt > 0.08:
                dz = history[-1][2] - history[0][2]
                v_mps = dz / dt
                rel_speed_kmh = round(v_mps * 3.6, 1)

        return {
            "x_m": x_m,
            "z_m": z_m,
            "rel_speed_kmh": rel_speed_kmh,
        }


class CockpitRadarWidget:
    """
    Renders a high-contrast top-down 2D Bird's-Eye View (BEV) radar widget
    showing surrounding traffic, distances, and relative threats.
    """

    def __init__(
        self,
        widget_width: int = 140,
        widget_height: int = 180,
        range_m: float = 40.0,
        width_m: float = 12.0,
    ):
        self.w = widget_width
        self.h = widget_height
        self.range_m = range_m
        self.width_m = width_m

    def render(
        self,
        vehicles: List[Dict[str, Any]],
        top_threat_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Creates a 2D BEV radar image.
        vehicles: list of dicts with 'track_id', 'x_m', 'z_m', 'risk_level' ('NORMAL', 'CAUTION', 'CRITICAL')
        """
        # Create dark semi-translucent cockpit radar canvas
        canvas = np.full((self.h, self.w, 3), (18, 22, 26), dtype=np.uint8)

        # Draw radar distance concentric arcs (10m, 20m, 30m, 40m)
        cx = self.w // 2
        bottom_y = self.h - 14

        for dist_m in [10.0, 20.0, 30.0]:
            r_px = int((dist_m / self.range_m) * (self.h - 32))
            cv2.ellipse(
                canvas,
                (cx, bottom_y),
                (r_px, r_px),
                0,
                180,
                360,
                (45, 55, 65),
                1,
                cv2.LINE_AA,
            )
            # Distance labels
            cv2.putText(
                canvas,
                f"{int(dist_m)}m",
                (cx + 6, bottom_y - r_px + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (120, 140, 150),
                1,
                cv2.LINE_AA,
            )

        # Draw Ego Lane Corridor Guidelines (3.5m wide road lane)
        lane_half_w_px = int(((1.75) / self.width_m) * self.w)
        cv2.line(canvas, (cx - lane_half_w_px, bottom_y), (cx - lane_half_w_px, 16), (40, 70, 45), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx + lane_half_w_px, bottom_y), (cx + lane_half_w_px, 16), (40, 70, 45), 1, cv2.LINE_AA)

        # Draw Center Line
        cv2.line(canvas, (cx, bottom_y), (cx, 16), (35, 45, 55), 1, cv2.LINE_AA)

        # Draw Ego Rider Symbol (Cyan Motorcycle triangle/dot)
        rider_pts = np.array([
            [cx, bottom_y - 8],
            [cx - 5, bottom_y + 4],
            [cx + 5, bottom_y + 4],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [rider_pts], (255, 200, 0))

        # Plot Surrounding Tracked Vehicles
        for v in vehicles:
            x_m = v.get("x_m", 0.0)
            z_m = v.get("z_m", 15.0)
            risk_lvl = str(v.get("risk_level", "LOW")).upper()
            v_id = v.get("vehicle_id") or v.get("track_id")

            # Project into radar canvas coordinates
            # X mapping: [-width_m/2, +width_m/2] -> [0, w]
            vx_px = int(cx + (x_m / (self.width_m / 2.0)) * (self.w // 2 - 8))
            # Z mapping: [0, range_m] -> [bottom_y, 20]
            vz_px = int(bottom_y - (z_m / self.range_m) * (self.h - 36))

            # Clamp inside widget
            vx_px = max(6, min(self.w - 6, vx_px))
            vz_px = max(16, min(self.h - 6, vz_px))

            # Color code based on risk severity
            is_top_threat = (top_threat_id is not None and v_id == top_threat_id)
            if risk_lvl in ["CRITICAL", "HIGH", "HIGH_RISK"] or is_top_threat:
                veh_color = (0, 0, 240)    # Red Emergency
                dot_r = 5
            elif risk_lvl in ["CAUTION", "MEDIUM"]:
                veh_color = (0, 165, 255)  # Amber Caution
                dot_r = 4
            else:
                veh_color = (0, 230, 115)  # Green Safe
                dot_r = 3

            # Draw vehicle dot
            cv2.circle(canvas, (vx_px, vz_px), dot_r, veh_color, -1, cv2.LINE_AA)

            # If top threat, draw pulsing hazard reticle
            if is_top_threat:
                cv2.circle(canvas, (vx_px, vz_px), dot_r + 4, veh_color, 1, cv2.LINE_AA)

        # Radar Header Title
        cv2.rectangle(canvas, (0, 0), (self.w, 14), (28, 36, 42), -1)
        cv2.putText(canvas, "360 SPATIAL RADAR", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 210, 230), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(canvas, (0, 0), (self.w - 1, self.h - 1), (60, 80, 95), 1)

        return canvas
