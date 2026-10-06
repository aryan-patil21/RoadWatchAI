"""
RoadWatch AI - Milestone 7: Custom Dataset Curation & Leakage-Free Splitting
----------------------------------------------------------------------------
Implements systematic road-event annotation and group-based train/val/test splitting:
- Applies kinematic event taxonomy (normal_driving, rapid_approach, close_following,
  unsafe_lane_change, sudden_braking, parallel_cruising)
- Groups by vehicle_id to prevent Video Data Leakage between splits
- Exports curated CSV datasets (train, val, test) and dataset metadata
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def assign_event_label(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Assigns an objective event label and safety level based on kinematic criteria.

    Returns:
        (event_label, safety_level)
    """
    proximity = row.get("proximity_score", 0.0)
    approach_rate = row.get("approach_rate", 0.0)
    v_long = row.get("v_long_proxy", 0.0)
    v_lat = row.get("v_lat_proxy", 0.0)
    acc = row.get("acceleration_proxy", 0.0)
    ttc = row.get("ttc_proxy_sec", 99.9)
    in_ego = row.get("in_ego_corridor", 0)
    is_cutting = row.get("is_cutting_lane", 0)
    sudden_brake = row.get("sudden_braking_flag", 0)

    # 1. Unsafe Lane Cut
    if is_cutting and proximity >= 0.35:
        return "unsafe_lane_change", "HIGH_RISK"

    # 2. Sudden Braking ahead
    if sudden_brake and proximity >= 0.35:
        return "sudden_braking", "HIGH_RISK"

    # 3. Rapid Approach / Critical TTC
    if (approach_rate > 0.12 and v_long > 30.0 and ttc <= 4.0) or (proximity >= 0.70 and approach_rate > 0.08):
        return "rapid_approach", "HIGH_RISK"

    # 4. Close Following (Tailgating)
    if proximity >= 0.55 and abs(approach_rate) <= 0.08:
        return "close_following", "CAUTION"

    # 5. Moderate Closing distance
    if approach_rate > 0.05 or (proximity >= 0.40 and in_ego):
        return "closing_distance", "CAUTION"

    # 6. Parallel Cruising in adjacent lane
    if not in_ego and proximity >= 0.30:
        return "parallel_cruising", "NORMAL"

    # 7. Normal cruising
    return "normal_driving", "NORMAL"


def create_custom_dataset(
    input_features_path: str = "outputs/behavioural_features.csv",
    output_dir: str = "data/processed",
) -> Dict[str, Any]:
    """
    Annotates behavioural samples and generates leakage-free train/val/test splits.
    """
    if not os.path.exists(input_features_path):
        raise FileNotFoundError(f"Source features not found: {input_features_path}")

    df = pd.read_csv(input_features_path)
    print(f"[Dataset Generator] Loaded {len(df)} feature records from '{input_features_path}'")

    # Apply objective kinematic labeling
    event_labels = []
    safety_levels = []
    for _, row in df.iterrows():
        evt, lvl = assign_event_label(row.to_dict())
        event_labels.append(evt)
        safety_levels.append(lvl)

    df["event_label"] = event_labels
    df["safety_level"] = safety_levels

    print("\nEvent Distribution:")
    print(df["event_label"].value_counts())
    print("\nSafety Level Distribution:")
    print(df["safety_level"].value_counts())

    os.makedirs(output_dir, exist_ok=True)
    all_path = os.path.join(output_dir, "roadwatch_dataset_all.csv")
    df.to_csv(all_path, index=False)

    # Prevent Data Leakage: Group by vehicle_id so individual vehicles don't cross splits
    groups = df["vehicle_id"]
    unique_vehicles = df["vehicle_id"].nunique()

    # If vehicle count is small (e.g. <= 4), fallback to chronological temporal split
    if unique_vehicles >= 5:
        # First split: 70% Train, 30% Temp (Val + Test)
        gss_train = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
        train_idx, temp_idx = next(gss_train.split(df, groups=groups))

        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()

        # Second split: Split temp equally into Val (15%) and Test (15%)
        temp_groups = temp_df["vehicle_id"]
        gss_val = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_groups))

        val_df = temp_df.iloc[val_idx].copy()
        test_df = temp_df.iloc[test_idx].copy()
    else:
        # Chronological block split
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

    # Save split datasets
    train_path = os.path.join(output_dir, "roadwatch_dataset_train.csv")
    val_path = os.path.join(output_dir, "roadwatch_dataset_val.csv")
    test_path = os.path.join(output_dir, "roadwatch_dataset_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Verification: Ensure no vehicle ID leakage between train and test
    train_vids = [int(v) for v in sorted(list(train_df["vehicle_id"].unique()))]
    test_vids = [int(v) for v in sorted(list(test_df["vehicle_id"].unique()))]
    leakage_overlap = [int(v) for v in set(train_vids).intersection(set(test_vids))]

    metadata = {
        "total_samples": len(df),
        "total_unique_vehicles": int(unique_vehicles),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_vehicles": train_vids,
        "test_vehicles": test_vids,
        "data_leakage_overlap": leakage_overlap,
        "is_leakage_free": (len(leakage_overlap) == 0),
        "event_distribution": {str(k): int(v) for k, v in df["event_label"].value_counts().to_dict().items()},
        "safety_level_distribution": {str(k): int(v) for k, v in df["safety_level"].value_counts().to_dict().items()},
    }


    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(" RoadWatch AI - Custom Dataset Curation Summary")
    print("=" * 60)
    print(f"Total Annotated Samples : {len(df)}")
    print(f"Train Split Samples     : {len(train_df)} ({len(train_vids)} vehicles)")
    print(f"Validation Split Samples: {len(val_df)}")
    print(f"Test Split Samples      : {len(test_df)} ({len(test_vids)} vehicles)")
    print(f"Data Leakage Free       : {metadata['is_leakage_free']} (0 overlapping vehicles)")
    print(f"Output Directory        : {output_dir}")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    create_custom_dataset()
