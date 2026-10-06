"""
Unit tests for Milestone 6: Risk Estimation Engine & ML Model
"""

import os
import unittest
import joblib
from src.risk_engine import RiskEngine

FEATURE_COLUMNS = [
    "proximity_score",
    "v_long_proxy",
    "v_lat_proxy",
    "approach_rate",
    "acceleration_proxy",
    "ttc_proxy_sec",
    "in_ego_corridor",
    "is_cutting_lane",
    "sudden_braking_flag",
]


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()

    def test_low_risk_scenario(self):
        features = {
            "vehicle_id": 1,
            "proximity_score": 0.15,
            "approach_rate": 0.0,
            "v_long_proxy": -10.0,
            "v_lat_proxy": 0.0,
            "ttc_proxy_sec": 99.9,
            "in_ego_corridor": 0,
            "is_cutting_lane": 0,
            "sudden_braking_flag": 0,
        }
        res = self.engine.evaluate_risk(features)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertLess(res["risk_score"], 40.0)

    def test_high_risk_rapid_approach_scenario(self):
        features = {
            "vehicle_id": 2,
            "proximity_score": 0.75,
            "approach_rate": 0.28,
            "v_long_proxy": 80.0,
            "v_lat_proxy": 0.0,
            "ttc_proxy_sec": 1.2,
            "in_ego_corridor": 1,
            "is_cutting_lane": 0,
            "sudden_braking_flag": 0,
        }
        res = self.engine.evaluate_risk(features)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertGreaterEqual(res["risk_score"], 70.0)

    def test_high_risk_lane_cutting_scenario(self):
        features = {
            "vehicle_id": 3,
            "proximity_score": 0.50,
            "approach_rate": 0.10,
            "v_long_proxy": 30.0,
            "v_lat_proxy": 65.0,
            "ttc_proxy_sec": 4.0,
            "in_ego_corridor": 1,
            "is_cutting_lane": 1,
            "sudden_braking_flag": 0,
        }
        res = self.engine.evaluate_risk(features)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertIn("aggressive lane cutting", res["all_factors"])

    def test_ml_model_prediction(self):
        import pandas as pd
        model_path = "models/risk_model_rf.joblib"
        if not os.path.exists(model_path):
            self.skipTest(f"Model artifact not found at {model_path}")

        rf_model = joblib.load(model_path)
        sample_df = pd.DataFrame(
            [[0.84, 80.0, 0.0, 0.20, 50.0, 1.0, 1, 0, 0]],
            columns=FEATURE_COLUMNS,
        )
        pred = rf_model.predict(sample_df)
        self.assertIn(pred[0], ["HIGH", "MEDIUM", "LOW"])
        self.assertEqual(pred[0], "HIGH")




if __name__ == "__main__":
    unittest.main()
