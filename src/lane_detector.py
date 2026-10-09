"""
RoadWatch AI - Lane Departure Warning (LDW) Detector
---------------------------------------------------
Detects ego lane line markings on the road surface and alerts the rider
if their path drifts across lane boundaries without a deliberate turn.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np


class LaneDepartureDetector:
    """
    Classical vision-based Lane Departure Warning (LDW) detector.
    Analyzes road surface markings within a perspective Region of Interest (ROI),
    estimates left and right lane boundaries, and monitors lateral ego vehicle offset.
    """

    def __init__(
        self,
        frame_width: int = 854,
        frame_height: int = 480,
        departure_threshold_ratio: float = 0.16,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.departure_threshold = departure_threshold_ratio * frame_width  # in pixels
        
        # Smoothed lane positions across frames
        self.prev_left_x: Optional[float] = None
        self.prev_right_x: Optional[float] = None
        self.departure_status: str = "NORMAL"  # "NORMAL", "DRIFT_LEFT", "DRIFT_RIGHT"
        self.offset_px: float = 0.0

    def process_frame(self, frame: cv2.Mat) -> Dict[str, Any]:
        """
        Detects road lane lines and computes rider center offset.
        Returns:
            Dict containing departure status, lateral offset, and lane line coordinates.
        """
        h, w = frame.shape[:2]

        # 1. Convert to Grayscale & Edge Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. Region of Interest (ROI) - Asphalt Trapezoid
        mask = np.zeros_like(edges)
        roi_vertices = np.array([
            [
                (int(w * 0.10), h),
                (int(w * 0.40), int(h * 0.60)),
                (int(w * 0.60), int(h * 0.60)),
                (int(w * 0.90), h),
            ]
        ], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # 3. Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=30,
            maxLineGap=20,
        )

        left_lines: List[Tuple[float, float]] = []
        right_lines: List[Tuple[float, float]] = []
        center_x = w / 2.0

        if lines is not None:
            for line in lines:
                coords = line.flatten()
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = coords[:4]
                if x2 == x1:
                    continue
                slope = float(y2 - y1) / float(x2 - x1)
                
                # Filter out near-horizontal noise lines
                if abs(slope) < 0.35:
                    continue

                # Left lane lines have negative slope in image coordinates
                if slope < 0 and max(x1, x2) < center_x + 50:
                    left_lines.append((slope, y1 - slope * x1))
                # Right lane lines have positive slope
                elif slope > 0 and min(x1, x2) > center_x - 50:
                    right_lines.append((slope, y1 - slope * x1))

        # 4. Fit Representative Left and Right Lines
        y_bottom = float(h)
        left_x_bottom = self._calculate_line_x(left_lines, y_bottom, self.prev_left_x)
        right_x_bottom = self._calculate_line_x(right_lines, y_bottom, self.prev_right_x)

        # Temporal smoothing
        if left_x_bottom is not None:
            self.prev_left_x = left_x_bottom if self.prev_left_x is None else 0.7 * self.prev_left_x + 0.3 * left_x_bottom
        if right_x_bottom is not None:
            self.prev_right_x = right_x_bottom if self.prev_right_x is None else 0.7 * self.prev_right_x + 0.3 * right_x_bottom

        # 5. Evaluate Lateral Departure
        # Default lane center is image center
        lane_center = center_x
        if self.prev_left_x is not None and self.prev_right_x is not None:
            lane_center = (self.prev_left_x + self.prev_right_x) / 2.0
            self.offset_px = center_x - lane_center
        else:
            self.offset_px = 0.0

        # Departure threshold
        if self.offset_px > self.departure_threshold:
            self.departure_status = "DRIFT_RIGHT"
        elif self.offset_px < -self.departure_threshold:
            self.departure_status = "DRIFT_LEFT"
        else:
            self.departure_status = "NORMAL"

        return {
            "status": self.departure_status,
            "offset_px": round(self.offset_px, 1),
            "left_lane_x": self.prev_left_x,
            "right_lane_x": self.prev_right_x,
        }

    def _calculate_line_x(
        self,
        lines: List[Tuple[float, float]],
        y_eval: float,
        fallback: Optional[float],
    ) -> Optional[float]:
        if not lines:
            return fallback
        # Average slopes and intercepts
        avg_slope = float(np.mean([m for m, _ in lines]))
        avg_intercept = float(np.mean([b for _, b in lines]))
        if abs(avg_slope) < 1e-4:
            return fallback
        return (y_eval - avg_intercept) / avg_slope

    def draw_lanes(self, frame: cv2.Mat, ldw_info: Dict[str, Any]) -> None:
        """Renders subtle lane guides on the lower road surface."""
        h, w = frame.shape[:2]
        left_x = ldw_info.get("left_lane_x")
        right_x = ldw_info.get("right_lane_x")
        status = ldw_info.get("status", "NORMAL")

        color = (0, 220, 0) if status == "NORMAL" else (0, 140, 255)

        if left_x and 0 <= left_x <= w:
            cv2.line(frame, (int(left_x), h), (int(w * 0.42), int(h * 0.65)), color, 2, cv2.LINE_AA)
        if right_x and 0 <= right_x <= w:
            cv2.line(frame, (int(right_x), h), (int(w * 0.58), int(h * 0.65)), color, 2, cv2.LINE_AA)
