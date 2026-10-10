"""
Unit and integration tests for Milestone 14 Incident Blackbox Auto-Recorder.
"""

import json
import os
import unittest
import numpy as np
from src.incident_recorder import IncidentBlackboxRecorder
from src.warning_engine import RiderAlert, AlertLevel


class TestIncidentBlackbox(unittest.TestCase):
    """Validates rolling memory buffer, trigger logic, and incident export."""

    def setUp(self):
        self.output_dir = "outputs/test_incidents"
        self.recorder = IncidentBlackboxRecorder(
            output_dir=self.output_dir,
            pre_event_seconds=1.0,
            post_event_seconds=1.0,
            fps=25.0,
        )

    def test_incident_trigger_and_report_generation(self):
        """Verifies that a CRITICAL alert triggers recording and exports MP4 + JSON."""
        dummy_frame = np.zeros((480, 854, 3), dtype=np.uint8)

        # Push 20 normal frames
        for f in range(20):
            self.recorder.push_frame(
                frame=dummy_frame,
                frame_idx=f,
                timestamp_sec=f * 0.04,
                active_alert=None,
            )

        self.assertFalse(self.recorder.is_recording)

        # Trigger critical alert at frame 21
        alert = RiderAlert(
            alert_id="ALERT_001",
            timestamp_sec=21 * 0.04,
            frame_idx=21,
            vehicle_id=42,
            level=AlertLevel.CRITICAL,
            title="DANGER: Truck approaching rapidly",
            message="Rapid approach detected",
            suggested_action="Brake immediately",
            risk_score=85.0,
            direction="ahead",
        )
        mock_telemetry = {
            "vehicle_id": 42,
            "z_m": 8.5,
            "rel_speed_kmh": -32.0,
            "ttc_proxy_sec": 1.8,
        }

        # Push frame with critical alert
        self.recorder.push_frame(
            frame=dummy_frame,
            frame_idx=21,
            timestamp_sec=21 * 0.04,
            active_alert=alert,
            top_vehicle_telemetry=mock_telemetry,
        )
        self.assertTrue(self.recorder.is_recording)

        # Feed post-event frames until auto-finalized
        finalized_path = None
        for f in range(22, 55):
            res = self.recorder.push_frame(
                frame=dummy_frame,
                frame_idx=f,
                timestamp_sec=f * 0.04,
                active_alert=alert,
                top_vehicle_telemetry=mock_telemetry,
            )
            if res:
                finalized_path = res
                break

        self.assertIsNotNone(finalized_path)
        self.assertTrue(os.path.exists(finalized_path))

        # Check corresponding JSON report
        json_path = finalized_path.replace(".mp4", "_telemetry.json")
        self.assertTrue(os.path.exists(json_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["alert_title"], "DANGER: Truck approaching rapidly")
            self.assertEqual(data["forensics_summary"]["minimum_ttc_seconds"], 1.8)
            self.assertEqual(data["forensics_summary"]["peak_closing_speed_kmh"], 32.0)

    def tearDown(self):
        # Cleanup test files
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)


if __name__ == "__main__":
    unittest.main()
