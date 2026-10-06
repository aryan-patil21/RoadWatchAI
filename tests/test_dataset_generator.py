"""
Unit tests for Milestone 7: Custom Dataset Curation & Data Leakage Prevention
"""

import os
import unittest
import pandas as pd
from src.dataset_generator import assign_event_label, create_custom_dataset


class TestDatasetGenerator(unittest.TestCase):
    def test_assign_event_label_rapid_approach(self):
        row = {
            "proximity_score": 0.75,
            "approach_rate": 0.25,
            "v_long_proxy": 50.0,
            "v_lat_proxy": 0.0,
            "ttc_proxy_sec": 1.5,
            "in_ego_corridor": 1,
            "is_cutting_lane": 0,
            "sudden_braking_flag": 0,
        }
        evt, lvl = assign_event_label(row)
        self.assertEqual(evt, "rapid_approach")
        self.assertEqual(lvl, "HIGH_RISK")

    def test_assign_event_label_lane_cut(self):
        row = {
            "proximity_score": 0.45,
            "approach_rate": 0.05,
            "v_long_proxy": 20.0,
            "v_lat_proxy": 45.0,
            "ttc_proxy_sec": 4.0,
            "in_ego_corridor": 1,
            "is_cutting_lane": 1,
            "sudden_braking_flag": 0,
        }
        evt, lvl = assign_event_label(row)
        self.assertEqual(evt, "unsafe_lane_change")
        self.assertEqual(lvl, "HIGH_RISK")

    def test_assign_event_label_close_following(self):
        row = {
            "proximity_score": 0.65,
            "approach_rate": 0.02,
            "v_long_proxy": 5.0,
            "v_lat_proxy": 0.0,
            "ttc_proxy_sec": 99.9,
            "in_ego_corridor": 1,
            "is_cutting_lane": 0,
            "sudden_braking_flag": 0,
        }
        evt, lvl = assign_event_label(row)
        self.assertEqual(evt, "close_following")
        self.assertEqual(lvl, "CAUTION")

    def test_dataset_splits_and_leakage_free(self):
        meta = create_custom_dataset(
            input_features_path="outputs/behavioural_features.csv",
            output_dir="data/processed",
        )
        self.assertTrue(meta["is_leakage_free"], "Data leakage detected between train and test!")
        self.assertEqual(len(meta["data_leakage_overlap"]), 0)

        # Confirm all split CSV files exist and have data
        for split in ["train", "val", "test", "all"]:
            csv_path = f"data/processed/roadwatch_dataset_{split}.csv"
            self.assertTrue(os.path.exists(csv_path), f"Missing split file: {csv_path}")
            df = pd.read_csv(csv_path)
            self.assertGreater(len(df), 0)
            self.assertIn("event_label", df.columns)
            self.assertIn("safety_level", df.columns)


if __name__ == "__main__":
    unittest.main()
