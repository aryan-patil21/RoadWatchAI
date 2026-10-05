"""
Unit test for Milestone 1: Video Input & Inspection
"""

import os
import unittest
from src.inspect_video import inspect_video


class TestVideoInspection(unittest.TestCase):
    def setUp(self):
        self.video_path = "data/raw/sample_dashcam.mp4"
        self.output_dir = "outputs"

    def test_sample_video_exists(self):
        self.assertTrue(os.path.exists(self.video_path), f"Video not found at {self.video_path}")

    def test_inspect_video_metadata(self):
        metadata = inspect_video(self.video_path, self.output_dir)
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)
        self.assertIn("fps", metadata)
        self.assertIn("total_frames", metadata)
        self.assertIn("duration_seconds", metadata)

        # Confirm dimensions and FPS match expected dashcam values
        self.assertEqual(metadata["width"], 854)
        self.assertEqual(metadata["height"], 480)
        self.assertAlmostEqual(metadata["fps"], 25.0, places=1)
        self.assertGreater(metadata["total_frames"], 1000)
        self.assertTrue(metadata["all_samples_read"])

        # Confirm sample files were created
        for frame_file in metadata["sample_frames_extracted"]:
            self.assertTrue(os.path.exists(frame_file), f"Sample frame file missing: {frame_file}")


if __name__ == "__main__":
    unittest.main()
