"""
Unit tests for Milestone 5: Behavioural Feature Extractor
"""

import unittest
from src.feature_extractor import BehaviouralFeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        # 25 FPS, 854x480 resolution
        self.extractor = BehaviouralFeatureExtractor(fps=25.0, frame_width=854, frame_height=480, window_size=5)

    def test_feature_vector_keys(self):
        feat = self.extractor.extract_features(
            frame_idx=1,
            vehicle_id=10,
            class_name="car",
            bbox=[100, 100, 200, 250],
            proximity_score=0.45,
        )
        expected_keys = [
            "vehicle_id",
            "frame_idx",
            "timestamp_sec",
            "class_name",
            "cx",
            "cy",
            "ground_y",
            "box_width",
            "box_height",
            "proximity_score",
            "v_long_proxy",
            "v_lat_proxy",
            "approach_rate",
            "acceleration_proxy",
            "trajectory_angle_deg",
            "ttc_proxy_sec",
            "in_ego_corridor",
            "is_cutting_lane",
            "sudden_braking_flag",
        ]
        for k in expected_keys:
            self.assertIn(k, feat)

    def test_approaching_vehicle_ttc(self):
        vid = 20
        # Feed 6 frames of a rapidly approaching vehicle
        feat = None
        for i in range(6):
            prox = 0.20 + (i * 0.08)  # proximity grows rapidly
            y2 = 250 + (i * 10)       # moving down towards rider
            feat = self.extractor.extract_features(
                frame_idx=i + 1,
                vehicle_id=vid,
                class_name="truck",
                bbox=[350, y2 - 100, 480, y2],
                proximity_score=prox,
            )

        self.assertIsNotNone(feat)
        self.assertGreater(feat["approach_rate"], 0.05)
        self.assertGreater(feat["v_long_proxy"], 0.0)
        # Should calculate an imminent TTC in seconds (< 10s)
        self.assertLess(feat["ttc_proxy_sec"], 10.0)

    def test_lane_cutting_detection(self):
        vid = 30
        # Simulate car swiftly swerving from left lane into ego lane
        # Left boundary is 0.35 * 854 = 298.9
        feat = None
        for i in range(5):
            cx = 250 + (i * 20)  # rapidly shifting right (+20 px per frame)
            feat = self.extractor.extract_features(
                frame_idx=i + 1,
                vehicle_id=vid,
                class_name="car",
                bbox=[int(cx - 30), 200, int(cx + 30), 320],
                proximity_score=0.40,
            )

        self.assertIsNotNone(feat)
        self.assertGreater(feat["v_lat_proxy"], 30.0)
        self.assertEqual(feat["is_cutting_lane"], 1)


if __name__ == "__main__":
    unittest.main()
