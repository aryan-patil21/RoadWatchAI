"""
Unit tests for Milestone 9: FastAPI Backend Service
"""

import os
import unittest
from fastapi.testclient import TestClient
from src.api import app


class TestFastAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_image = "outputs/sample_frame_00000.jpg"

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs_url", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "RoadWatch AI")
        self.assertIn("device", data)

    def test_get_sound_asset(self):
        response = self.client.get("/api/v1/sounds/chime_caution.wav")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")

    def test_get_alerts_endpoints(self):
        # Active alert endpoint
        active_resp = self.client.get("/api/v1/alerts/active")
        self.assertEqual(active_resp.status_code, 200)
        self.assertIn("has_active_alert", active_resp.json())

        # History endpoint
        hist_resp = self.client.get("/api/v1/alerts/history")
        self.assertEqual(hist_resp.status_code, 200)
        self.assertIn("total_alerts", hist_resp.json())

    def test_analyze_frame_endpoint(self):
        if not os.path.exists(self.sample_image):
            self.skipTest(f"Sample image missing at {self.sample_image}")

        with open(self.sample_image, "rb") as f:
            files = {"file": ("frame.jpg", f, "image/jpeg")}
            response = self.client.post("/api/v1/analyze/frame", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("frame_idx", data)
        self.assertIn("timestamp_sec", data)
        self.assertIn("tracked_vehicles", data)
        self.assertIn("system_status", data)
        self.assertIsInstance(data["tracked_vehicles"], list)


if __name__ == "__main__":
    unittest.main()
