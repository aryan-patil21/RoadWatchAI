"""
Unit and integration tests for Milestone 13 Metric Ground Projector and 2D BEV Radar Widget.
"""

import unittest
import numpy as np
from src.spatial_radar import MetricGroundProjector, CockpitRadarWidget


class TestSpatialRadar(unittest.TestCase):
    """Validates IPM metric projection and 2D Bird's-Eye View radar rendering."""

    def setUp(self):
        self.projector = MetricGroundProjector(
            frame_width=854,
            frame_height=480,
            camera_height_m=1.25,
            pitch_angle_deg=6.0,
            focal_length_px=720.0,
            horizon_y=265.0,
            fps=25.0,
        )
        self.radar = CockpitRadarWidget(widget_width=150, widget_height=190, range_m=40.0)

    def test_project_point_to_ground(self):
        """Verifies that lower Y coordinates project further away and higher Y closer."""
        # Contact point at bottom of frame (close to rider)
        x_close, z_close = self.projector.project_point_to_ground(427.0, 450.0)
        self.assertAlmostEqual(x_close, 0.0, delta=0.5)
        self.assertLess(z_close, 5.0)

        # Contact point higher up near horizon (far ahead)
        x_far, z_far = self.projector.project_point_to_ground(427.0, 280.0)
        self.assertGreater(z_far, 15.0)

        # Contact point to the right
        x_right, z_right = self.projector.project_point_to_ground(650.0, 350.0)
        self.assertGreater(x_right, 0.0)

        # Contact point to the left
        x_left, z_left = self.projector.project_point_to_ground(200.0, 350.0)
        self.assertLess(x_left, 0.0)

    def test_estimate_vehicle_metric_state_speed(self):
        """Verifies calculation of relative speed in km/h across timestamps."""
        track_id = 99
        bbox_1 = [400, 300, 460, 360]
        state_1 = self.projector.estimate_vehicle_metric_state(track_id, bbox_1, timestamp_sec=1.0)
        self.assertIn("z_m", state_1)
        self.assertIn("rel_speed_kmh", state_1)

        # Vehicle moves closer over time (closing in)
        for i in range(1, 5):
            t = 1.0 + i * 0.04
            # moving down in frame = closer
            bbox_next = [400, 300 + i * 5, 460, 360 + i * 5]
            state_next = self.projector.estimate_vehicle_metric_state(track_id, bbox_next, timestamp_sec=t)

        self.assertLess(state_next["rel_speed_kmh"], 0.0)  # Approaching / closing in

    def test_radar_widget_render(self):
        """Verifies that radar widget returns a valid 3-channel image array of correct shape."""
        mock_vehicles = [
            {"track_id": 1, "x_m": -1.5, "z_m": 8.0, "risk_level": "LOW"},
            {"track_id": 2, "x_m": 1.2, "z_m": 5.4, "risk_level": "CRITICAL"},
        ]
        radar_img = self.radar.render(vehicles=mock_vehicles, top_threat_id=2)

        self.assertIsInstance(radar_img, np.ndarray)
        self.assertEqual(radar_img.shape, (190, 150, 3))
        self.assertEqual(radar_img.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
