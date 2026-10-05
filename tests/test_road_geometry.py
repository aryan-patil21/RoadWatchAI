"""
Unit tests for Milestone 4: Road Geometry & Relative Position
"""

import unittest
from src.road_geometry import RoadGeometryAnalyzer


class TestRoadGeometry(unittest.TestCase):
    def setUp(self):
        # 854 x 480 frame with ego lane bounds between 35% and 65% (approx 298.9 to 555.1)
        self.analyzer = RoadGeometryAnalyzer(frame_width=854, frame_height=480)

    def test_lateral_zones(self):
        # Far left
        self.assertEqual(self.analyzer.get_lateral_zone(100), "LEFT_ZONE")
        # Center ego lane
        self.assertEqual(self.analyzer.get_lateral_zone(427), "EGO_LANE")
        # Far right
        self.assertEqual(self.analyzer.get_lateral_zone(700), "RIGHT_ZONE")

    def test_proximity_band(self):
        # Very close vehicle taking up large bottom area of the image
        close_bbox = [200, 200, 600, 460]
        band, score = self.analyzer.estimate_proximity_band(close_bbox)
        self.assertIn(band, ["VERY_CLOSE", "CLOSE"])
        self.assertGreaterEqual(score, 0.4)

        # Distant small vehicle near horizon
        far_bbox = [400, 200, 450, 240]
        band_far, score_far = self.analyzer.estimate_proximity_band(far_bbox)
        self.assertIn(band_far, ["FAR", "MEDIUM"])
        self.assertLess(score_far, 0.35)

    def test_movement_trends_approaching(self):
        track_id = 99
        # Simulate approaching vehicle over consecutive frames (height grows, y2 moves down)
        for f_idx in range(6):
            h = 50 + (f_idx * 5)
            y2 = 250 + (f_idx * 6)
            bbox = [400, y2 - h, 450, y2]
            vehicle_record = {
                "track_id": track_id,
                "class_name": "car",
                "confidence": 0.85,
                "bbox": bbox,
                "center": (425, int(y2 - (h / 2))),
            }
            res = self.analyzer.analyze_vehicle(f_idx, vehicle_record)

        long_trend, lat_trend = self.analyzer.estimate_movement_trends(track_id)
        self.assertEqual(long_trend, "APPROACHING")

    def test_movement_trends_receding(self):
        track_id = 100
        # Simulate receding vehicle (height shrinks, y2 moves up toward horizon)
        for f_idx in range(6):
            h = 80 - (f_idx * 5)
            y2 = 350 - (f_idx * 6)
            bbox = [400, y2 - h, 450, y2]
            vehicle_record = {
                "track_id": track_id,
                "class_name": "car",
                "confidence": 0.85,
                "bbox": bbox,
                "center": (425, int(y2 - (h / 2))),
            }
            self.analyzer.analyze_vehicle(f_idx, vehicle_record)

        long_trend, lat_trend = self.analyzer.estimate_movement_trends(track_id)
        self.assertEqual(long_trend, "RECEDING")


if __name__ == "__main__":
    unittest.main()
