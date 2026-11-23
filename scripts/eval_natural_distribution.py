"""Evaluate XGBoost and Random Forest on realistic Natural Class Distribution.

Re-weights the held-out test slices (same chronological test slices as
src/train_xgboost.py) to a natural/realistic mix:
  - Normal: 90.0% (all 40,000 normal test frames)
  - DoS: 2.5% (1,111 frames, subsampled with fixed seed)
  - Fuzzy: 2.5% (1,111 frames, subsampled with fixed seed)
  - Spoof_Gear: 2.5% (1,111 frames, subsampled with fixed seed)
  - Spoof_RPM: 2.5% (1,111 frames, subsampled with fixed seed)
  Total: 44,444 frames (Normal: 90.00%, each attack: 2.50%)

Calculates per-class precision, recall, F1, false positive rate on Normal,
and number of false alarms per 100,000 normal frames.
Saves metrics to reports/natural_distribution_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from backend.config import settings
from src.train_multiclass import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    PREPARED_DATASET_PATH,
    load_and_prepare_data,
)

_ROOT = settings.PROJECT_ROOT
XGB_MODEL_PATH = _ROOT / "models" / "xgb_model.pkl"
RF_MODEL_PATH = _ROOT / "models" / "multiclass_model.pkl"
LABEL_ENC_PATH = _ROOT / "models" / "multiclass_label_encoder.pkl"
OUTPUT_REPORT_PATH = _ROOT / "reports" / "natural_distribution_metrics.json"

ATTACK_CLASSES = ("DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM")


def evaluate_model_on_distribution(
    model: Any,
    X: np.ndarray,
    y_true_raw: np.ndarray,
    label_encoder: Any,
) -> Dict[str, Any]:
    """Evaluate a trained model and compute metrics including Normal FPR."""
    y_pred_enc = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    report = classification_report(
        y_true_raw,
        y_pred_labels,
        labels=list(CLASS_LABELS),
        output_dict=True,
    )
    cm = confusion_matrix(y_true_raw, y_pred_labels, labels=list(CLASS_LABELS))

    # Normal False Positive Rate & False Alarms per 100k
    normal_mask = (y_true_raw == "Normal")
    total_normal = int(np.sum(normal_mask))
    # False alarms: ground truth Normal predicted as an attack
    fp_normal = int(np.sum((y_true_raw == "Normal") & (y_pred_labels != "Normal")))
    fpr_normal = float(fp_normal / total_normal) if total_normal > 0 else 0.0
    false_alarms_per_100k = float(fpr_normal * 100000.0)

    per_class = {}
    for cls in CLASS_LABELS:
        m = report[cls]
        per_class[cls] = {
            "precision": round(float(m["precision"]), 6),
            "recall": round(float(m["recall"]), 6),
            "f1-score": round(float(m["f1-score"]), 6),
            "support": int(m["support"]),
        }

    return {
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
        "false_positive_rate_normal": round(fpr_normal, 6),
        "false_alarms_per_100k_normal": round(false_alarms_per_100k, 2),
        "total_normal_frames": total_normal,
        "false_positive_normal_count": fp_normal,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(CLASS_LABELS),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on realistic natural distribution mix.")
    parser.add_argument("--random-state", type=int, default=42, help="Fixed random seed for subsampling.")
    parser.add_argument("--window-n", type=int, default=50, help="Window size for feature extraction.")
    args = parser.parse_args()

    # Verify model files exist
    if not XGB_MODEL_PATH.exists():
        sys.exit(f"Error: {XGB_MODEL_PATH} not found. Run training first.")
    if not RF_MODEL_PATH.exists():
        sys.exit(f"Error: {RF_MODEL_PATH} not found. Run training first.")
    if not LABEL_ENC_PATH.exists():
        sys.exit(f"Error: {LABEL_ENC_PATH} not found.")

    # 1. Load data using the exact same chronological split as src/train_xgboost.py
    _, test_feat_df, _ = load_and_prepare_data(
        dataset_path=PREPARED_DATASET_PATH,
        max_train_rows=300000,
        test_ratio=0.20,
        window_n=args.window_n,
    )

    # 2. Re-weight to natural mix: Normal 90%, each attack 2.5%
    normal_df = test_feat_df[test_feat_df["label"] == "Normal"]
    n_normal = len(normal_df)  # 40,000

    # Subsampling attack classes: 2.5% / 90.0% = 1 / 36
    n_attack_per_class = int(round(n_normal * (0.025 / 0.90)))  # 1,111 frames

    sampled_dfs = [normal_df]
    counts = {"Normal": n_normal}

    for cls in ATTACK_CLASSES:
        cls_df = test_feat_df[test_feat_df["label"] == cls]
        sampled_cls = cls_df.sample(n=n_attack_per_class, random_state=args.random_state)
        sampled_dfs.append(sampled_cls)
        counts[cls] = len(sampled_cls)

    natural_test_df = pd.concat(sampled_dfs, ignore_index=True)
    total_frames = len(natural_test_df)

    percentages = {cls: round((count / total_frames) * 100.0, 2) for cls, count in counts.items()}

    print("=" * 78)
    print("  NATURAL DISTRIBUTION TEST SET COUNTS & PROPORTIONS")
    print("=" * 78)
    for cls in CLASS_LABELS:
        print(f"  {cls:<14}: {counts[cls]:>8,} frames ({percentages[cls]:>5.2f}%)")
    print("  " + "-" * 56)
    print(f"  {'Total':<14}: {total_frames:>8,} frames (100.00%)")
    print("=" * 78)

    # 3. Prepare test matrices
    X_test = natural_test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test_raw = np.array(natural_test_df["label"].to_list(), dtype=str)

    # Load label encoder
    label_encoder = joblib.load(LABEL_ENC_PATH)

    # 4. Evaluate XGBoost
    print("\nEvaluating XGBoost...")
    xgb_model = joblib.load(XGB_MODEL_PATH)
    xgb_metrics = evaluate_model_on_distribution(xgb_model, X_test, y_test_raw, label_encoder)

    # 5. Evaluate Random Forest
    print("Evaluating Random Forest...")
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_metrics = evaluate_model_on_distribution(rf_model, X_test, y_test_raw, label_encoder)

    # 6. Print comparison results
    print("\n" + "=" * 78)
    print("  EVALUATION RESULTS UNDER NATURAL DISTRIBUTION (90% Normal / 2.5% per Attack)")
    print("=" * 78)
    print(f"{'Class':<14} | {'RF Prec':>8} {'RF Rec':>8} {'RF F1':>8} | {'XGB Prec':>8} {'XGB Rec':>8} {'XGB F1':>8} | {'Support':>8}")
    print("-" * 78)
    for cls in CLASS_LABELS:
        rf_m = rf_metrics["per_class"][cls]
        xgb_m = xgb_metrics["per_class"][cls]
        print(
            f"{cls:<14} | {rf_m['precision']:>8.4f} {rf_m['recall']:>8.4f} {rf_m['f1-score']:>8.4f} | "
            f"{xgb_m['precision']:>8.4f} {xgb_m['recall']:>8.4f} {xgb_m['f1-score']:>8.4f} | "
            f"{rf_m['support']:>8,}"
        )
    print("-" * 78)
    print(
        f"{'Accuracy':<14} | {'':>8} {'':>8} {rf_metrics['accuracy']:>8.4f} | "
        f"{'':>8} {'':>8} {xgb_metrics['accuracy']:>8.4f} | {total_frames:>8,}"
    )
    print(
        f"{'Macro F1':<14} | {'':>8} {'':>8} {rf_metrics['macro_f1']:>8.4f} | "
        f"{'':>8} {'':>8} {xgb_metrics['macro_f1']:>8.4f} |"
    )
    print(
        f"{'Weighted F1':<14} | {'':>8} {'':>8} {rf_metrics['weighted_f1']:>8.4f} | "
        f"{'':>8} {'':>8} {xgb_metrics['weighted_f1']:>8.4f} |"
    )
    print("=" * 78)

    print("\n" + "=" * 78)
    print("  NORMAL CLASS FALSE POSITIVE & FALSE ALARM METRICS")
    print("=" * 78)
    print(f"  Random Forest:")
    print(f"    - False Positive Rate (FPR) on Normal: {rf_metrics['false_positive_rate_normal']:.6f} ({rf_metrics['false_positive_rate_normal']*100:.4f}%)")
    print(f"    - False Alarms per 100,000 Normal Frames: {rf_metrics['false_alarms_per_100k_normal']:,.1f}")
    print(f"  XGBoost:")
    print(f"    - False Positive Rate (FPR) on Normal: {xgb_metrics['false_positive_rate_normal']:.6f} ({xgb_metrics['false_positive_rate_normal']*100:.4f}%)")
    print(f"    - False Alarms per 100,000 Normal Frames: {xgb_metrics['false_alarms_per_100k_normal']:,.1f}")
    print("=" * 78)

    # 7. Write reports/natural_distribution_metrics.json
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now(timezone.utc).isoformat(),
        "random_state": args.random_state,
        "distribution": {
            "counts": counts,
            "percentages": percentages,
            "total_frames": total_frames,
        },
        "xgboost": xgb_metrics,
        "random_forest": rf_metrics,
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] Metrics saved to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
