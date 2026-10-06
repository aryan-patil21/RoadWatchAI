"""
RoadWatch AI - Milestone 6: Machine Learning Risk Model Trainer
---------------------------------------------------------------
Trains and compares supervised machine-learning models (Logistic Regression & Random Forest)
on behavioural features extracted from dashcam video.

Produces:
- Model evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix)
- Feature Importance ranking
- Saved trained model artifact (models/risk_model_rf.joblib)
- JSON evaluation report (outputs/risk_model_evaluation.json)
"""

import os
import sys
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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


def train_and_evaluate_models(
    dataset_path: str = "outputs/behavioural_features.csv",
    models_dir: str = "models",
    output_report_path: str = "outputs/risk_model_evaluation.json",
) -> Dict[str, Any]:
    """
    Trains Logistic Regression and Random Forest models on behavioural features
    and exports evaluation metrics.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Feature dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"[ML Risk Trainer] Loaded {len(df)} samples from '{dataset_path}'")

    # Generate baseline risk labels using RiskEngine
    risk_engine = RiskEngine()
    labels = []
    scores = []
    for _, row in df.iterrows():
        eval_res = risk_engine.evaluate_risk(row.to_dict())
        labels.append(eval_res["risk_level"])
        scores.append(eval_res["risk_score"])

    df["risk_label"] = labels
    df["risk_score"] = scores

    print(f"[ML Risk Trainer] Class distribution:\n{df['risk_label'].value_counts()}")

    X = df[FEATURE_COLUMNS]
    y = df["risk_label"]

    # Stratified 80/20 train-test split
    # If any class has very few samples (e.g. < 2), stratify fallback
    min_class_count = df["risk_label"].value_counts().min()
    stratify = y if min_class_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )
    print(f"[ML Risk Trainer] Training set: {len(X_train)} | Test set: {len(X_test)}")

    # 1. Model A: Logistic Regression (Interpretable Linear Baseline with Scaling)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    log_reg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)

    # 2. Model B: Random Forest Classifier (Non-linear Ensemble with Balanced Weights)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf, labels=sorted(list(set(y)))).tolist()

    # Feature Importance from Random Forest
    importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf.feature_importances_]))
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    print("\n" + "=" * 60)
    print(" RoadWatch AI - Machine Learning Model Comparison")
    print("=" * 60)
    print(f"Logistic Regression Accuracy : {acc_lr * 100:.2f}%")
    print(f"Random Forest Accuracy       : {acc_rf * 100:.2f}%")
    print("\nTop Feature Importances (Random Forest):")
    for feat, imp in list(sorted_importances.items())[:5]:
        print(f"  - {feat:<22}: {imp:.4f} ({imp*100:.1f}%)")
    print("=" * 60)

    # Save Random Forest model artifact
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "risk_model_rf.joblib")
    joblib.dump(rf, model_save_path)
    print(f"[ML Risk Trainer] Saved trained Random Forest model to '{model_save_path}'")

    evaluation_summary = {
        "dataset_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution": df["risk_label"].value_counts().to_dict(),
        "logistic_regression": {
            "accuracy": round(acc_lr, 4),
            "report": report_lr,
        },
        "random_forest": {
            "accuracy": round(acc_rf, 4),
            "report": report_rf,
            "confusion_matrix": conf_matrix_rf,
            "classes": sorted(list(set(y))),
            "feature_importances": sorted_importances,
            "saved_model": model_save_path,
        },
    }

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(evaluation_summary, f, indent=2)

    print(f"[ML Risk Trainer] Saved evaluation report to '{output_report_path}'")
    return evaluation_summary


if __name__ == "__main__":
    train_and_evaluate_models()
