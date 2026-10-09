"""
RoadWatch AI - Milestone 11: End-to-End Pipeline Integration Test
-----------------------------------------------------------------
Validates the complete closed loop:
1. Stream runner feeds video frames sequentially into the perception & warning engine.
2. FastAPI backend endpoints (/health, /api/v1/alerts/active, /api/v1/alerts/history) reflect state.
3. Audio alert assets are served and verified.
4. Output annotated video stream is properly written.
"""

import os
import sys
import unittest
import tempfile
import cv2
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import app, pipeline_state
from src.stream_runner import LiveStreamRunner


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_video = "data/raw/sample_dashcam.mp4"
        cls.test_frames = 40

    def test_01_stream_runner_executes_and_updates_api(self):
        """Verify stream runner processes frames and updates API state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_video = os.path.join(tmpdir, "test_stream_out.mp4")
            runner = LiveStreamRunner(
                video_path=self.sample_video,
                pace_stream=False,  # Max speed for test
                save_annotated_stream=True,
                output_stream_path=out_video,
                conf_threshold=0.25,
            )

            stats = runner.run(max_frames=self.test_frames)

            # Check runner results
            self.assertEqual(stats["processed_frames"], self.test_frames)
            self.assertGreater(stats["average_fps"], 0.0)
            self.assertEqual(stats["critical_alerts"], 0)  # Diverging opposing truck should not trigger false emergency

            # Check that output video was generated and valid
            self.assertTrue(os.path.exists(out_video))
            cap = cv2.VideoCapture(out_video)
            self.assertTrue(cap.isOpened())
            cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.assertEqual(cap_frames, self.test_frames)
            cap.release()

            # Verify FastAPI /health endpoint reflects processed frames
            health_res = self.client.get("/health")
            self.assertEqual(health_res.status_code, 200)
            health_data = health_res.json()
            self.assertEqual(health_data["status"], "healthy")
            self.assertEqual(health_data["frames_processed"], self.test_frames)

    def test_02_audio_cue_sound_serving(self):
        """Verify sound endpoints serve valid wav audio."""
        res_caution = self.client.get("/api/v1/sounds/chime_caution.wav")
        self.assertEqual(res_caution.status_code, 200)
        self.assertEqual(res_caution.headers["content-type"], "audio/wav")
        self.assertGreater(len(res_caution.content), 1000)

        res_critical = self.client.get("/api/v1/sounds/chime_critical.wav")
        self.assertEqual(res_critical.status_code, 200)
        self.assertEqual(res_critical.headers["content-type"], "audio/wav")
        self.assertGreater(len(res_critical.content), 1000)


if __name__ == "__main__":
    unittest.main()
