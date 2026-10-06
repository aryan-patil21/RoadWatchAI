"""
Unit tests for Milestone 8: Warning Engine (Debouncing, Cooldown, Prioritization)
"""

import unittest
from src.warning_engine import WarningEngine, AlertLevel


class TestWarningEngine(unittest.TestCase):
    def setUp(self):
        # 3-frame debounce, 3.0-second cooldown
        self.engine = WarningEngine(
            caution_threshold=40.0,
            critical_threshold=70.0,
            debounce_frames=3,
            cooldown_seconds=3.0,
        )

    def test_debouncing_prevents_single_frame_glitches(self):
        vid = 1
        vehicle_eval = {
            "vehicle_id": vid,
            "risk_score": 75.0,
            "class_name": "car",
            "primary_factor": "rapid approach",
            "lateral_zone": "EGO_LANE",
        }

        # Frame 1: High risk occurs
        alert_f1 = self.engine.process_frame(frame_idx=1, timestamp_sec=0.04, vehicle_evaluations=[vehicle_eval])
        self.assertIsNone(alert_f1, "Alert fired on frame 1 without debouncing!")

        # Frame 2: High risk persists
        alert_f2 = self.engine.process_frame(frame_idx=2, timestamp_sec=0.08, vehicle_evaluations=[vehicle_eval])
        self.assertIsNone(alert_f2, "Alert fired on frame 2 without debouncing!")

        # Frame 3: Debounce threshold satisfied (3 consecutive frames) -> MUST fire
        alert_f3 = self.engine.process_frame(frame_idx=3, timestamp_sec=0.12, vehicle_evaluations=[vehicle_eval])
        self.assertIsNotNone(alert_f3, "Debounced alert failed to fire on frame 3!")
        self.assertEqual(alert_f3.level, AlertLevel.CRITICAL)

    def test_cooldown_suppresses_redundant_alerts(self):
        vid = 2
        vehicle_eval = {
            "vehicle_id": vid,
            "risk_score": 50.0,
            "class_name": "truck",
            "primary_factor": "closing distance",
            "lateral_zone": "LEFT_ZONE",
        }

        # Satisfy debounce
        for f in range(1, 4):
            alert = self.engine.process_frame(frame_idx=f, timestamp_sec=f * 0.04, vehicle_evaluations=[vehicle_eval])

        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, AlertLevel.CAUTION)

        # Immediate next frame (1.0s later, within 3.0s cooldown) should be suppressed
        alert_during_cooldown = self.engine.process_frame(
            frame_idx=10, timestamp_sec=1.00, vehicle_evaluations=[vehicle_eval]
        )
        self.assertIsNone(alert_during_cooldown, "Cooldown failed to suppress repeat alert!")

    def test_escalation_overrides_cooldown(self):
        vid = 3
        caution_eval = {
            "vehicle_id": vid,
            "risk_score": 45.0,
            "class_name": "car",
            "primary_factor": "closing distance",
            "lateral_zone": "RIGHT_ZONE",
        }
        critical_eval = {
            "vehicle_id": vid,
            "risk_score": 85.0,
            "class_name": "car",
            "primary_factor": "aggressive lane cutting",
            "lateral_zone": "EGO_LANE",
        }

        # Fire Caution alert
        for f in range(1, 4):
            alert = self.engine.process_frame(frame_idx=f, timestamp_sec=f * 0.04, vehicle_evaluations=[caution_eval])
        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, AlertLevel.CAUTION)

        # Danger escalates sharply 0.5s later (during cooldown) -> MUST override cooldown!
        fired_crit = None
        for f in range(5, 8):
            res = self.engine.process_frame(
                frame_idx=f, timestamp_sec=0.50 + (f * 0.04), vehicle_evaluations=[critical_eval]
            )
            if res:
                fired_crit = res

        self.assertIsNotNone(fired_crit, "Escalation to CRITICAL was blocked by cooldown!")
        self.assertEqual(fired_crit.level, AlertLevel.CRITICAL)



if __name__ == "__main__":
    unittest.main()
