"""
Unit tests for Milestone 3: Object Tracking (ByteTrack)
"""

import os
import unittest
import numpy as np
import cv2
from src.vehicle_tracker import VehicleTracker, get_id_color


class TestVehicleTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracker = VehicleTracker(conf_threshold=0.25, device="cpu")
        cls.sample_video_path = "data/raw/sample_dashcam.mp4"

    def test_get_id_color(self):
        color1 = get_id_color(1)
        color2 = get_id_color(2)
        self.assertEqual(len(color1), 3)
        self.assertEqual(len(color2), 3)
        self.assertIsInstance(color1[0], int)

    def test_track_frame_synthetic(self):
        dummy_frame = np.zeros((480, 854, 3), dtype=np.uint8)
        annotated, tracks = self.tracker.track_frame(dummy_frame, persist=False)
        self.assertEqual(annotated.shape, dummy_frame.shape)
        self.assertIsInstance(tracks, list)

    def test_tracking_across_video_frames(self):
        if not os.path.exists(self.sample_video_path):
            self.skipTest(f"Video not found at {self.sample_video_path}")

        cap = cv2.VideoCapture(self.sample_video_path)
        tracks_seen = []
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
            annotated, tracks = self.tracker.track_frame(frame, persist=True)
            self.assertEqual(annotated.shape, (480, 854, 3))
            tracks_seen.extend(tracks)
        cap.release()

        # Verify record format for any tracked vehicle
        for t in tracks_seen:
            self.assertIn("class_name", t)
            self.assertIn("bbox", t)
            self.assertIn("center", t)
            self.assertIn("history_points", t)
            if t["track_id"] is not None:
                self.assertIsInstance(t["track_id"], int)


if __name__ == "__main__":
    unittest.main()
