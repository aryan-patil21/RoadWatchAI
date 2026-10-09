"""
Unit and integration tests for Milestone 12 Real-Time Stream Performance Profiler.
"""

import os
import unittest
from src.profiler import StreamProfiler, ProfilerReport


class TestStreamProfiler(unittest.TestCase):
    """Validates Milestone 12 profiler functionality, latency metrics, and reporting."""

    def setUp(self):
        self.video_path = "data/raw/sample_dashcam.mp4"
        self.test_output_dir = "outputs"

    def test_profiler_execution_and_metrics(self):
        """Runs a short profiling session on 25 frames and verifies report metrics."""
        profiler = StreamProfiler(
            video_path=self.video_path,
            target_fps=25.0,
            device="mps",
            save_annotated=False,
            output_dir=self.test_output_dir,
        )

        report = profiler.run(max_frames=25, start_frame=1050)

        self.assertIsInstance(report, ProfilerReport)
        self.assertEqual(report.total_frames, 25)
        self.assertGreater(report.achieved_fps, 10.0)
        self.assertGreater(report.p50_frame_latency_ms, 0.0)
        self.assertGreater(report.p95_frame_latency_ms, 0.0)
        self.assertGreater(report.p99_frame_latency_ms, 0.0)
        self.assertTrue(len(report.stages) >= 8)

        # Check that stage percentages roughly sum up to ~100%
        stage_names = [st.name for st in report.stages]
        self.assertIn("2_yolo_bytetrack", stage_names)
        self.assertIn("6_lane_corridor_ldw", stage_names)

        # Check export files exist
        json_path = os.path.join(self.test_output_dir, "profiler_benchmark_report.json")
        md_path = os.path.join(self.test_output_dir, "profiler_benchmark_report.md")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))


if __name__ == "__main__":
    unittest.main()
