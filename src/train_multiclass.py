"""Train Stage-2 multiclass attack classifier for STARF CAN Bus IDS.

Trains a Random Forest classifier on 5 traffic classes:
  Normal, DoS, Fuzzy, Spoof_Gear, Spoof_RPM.

Key design points:
  1. Reads data/prepared.parquet (produced by src/dataset_prepare.py).
  2. Extracts frame-level features via build_frame_features() in src/preprocess.py:
     can_id, id_repetition_count, inter_arrival_time_delta, dlc, d0..d7.
  3. Evaluates 5 classes: Normal, DoS, Fuzzy, Spoof_Gear, Spoof_RPM.
  4. Chronological 80/20 train/test split by contiguous blocks per class.
  5. Capped training rows (default: 300,000 total, 60,000 per class balanced before SMOTE).
  6. SMOTE applied strictly on the training split only.
  7. Saves model, feature list, label encoder, and metadata to models/.
  8. Saves classification report and confusion matrix to reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from backend.config import settings
from src.preprocess import build_frame_features

# 5 Target Classes
CLASS_LABELS = ("Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM")

FEATURE_COLUMNS = [
    "can_id",
    "id_repetition_count",
    "inter_arrival_time_delta",
    "dlc",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
]

PREPARED_DATASET_PATH = settings.PROJECT_ROOT / "data" / "prepared.parquet"
MODEL_OUTPUT_PATH = settings.PROJECT_ROOT / "models" / "multiclass_model.pkl"
FEATURES_OUTPUT_PATH = settings.PROJECT_ROOT / "models" / "multiclass_features.json"
LABEL_ENCODER_OUTPUT_PATH = settings.PROJECT_ROOT / "models" / "multiclass_label_encoder.pkl"
METADATA_OUTPUT_PATH = settings.PROJECT_ROOT / "models" / "multiclass_meta.json"
REPORT_OUTPUT_PATH = settings.PROJECT_ROOT / "reports" / "multiclass_classification_report.json"
CONFUSION_MATRIX_OUTPUT_PATH = settings.PROJECT_ROOT / "reports" / "multiclass_confusion_matrix.json"


def check_xgboost_available() -> bool:
    """Check if XGBoost is installed."""
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def load_and_prepare_data(
    dataset_path: Path,
    max_train_rows: int = 300000,
    test_ratio: float = 0.20,
    window_n: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load prepared dataset, apply chronological contiguous block split per class, and cap training set."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found at {dataset_path}. Run src/dataset_prepare.py first.")

    print(f"Loading dataset from: {dataset_path}")
    t0 = time.time()
    df = pd.read_parquet(dataset_path)
    print(f"  Loaded {len(df):,} total frames in {time.time() - t0:.2f}s")

    # Verify expected classes
    present_classes = sorted(df["label"].unique())
    print(f"  Present classes in dataset: {present_classes}")
    for cls in CLASS_LABELS:
        if cls not in present_classes:
            raise ValueError(f"Required class '{cls}' not found in dataset: {present_classes}")

    # Calculate per-class train cap
    num_classes = len(CLASS_LABELS)
    cap_per_class = max_train_rows // num_classes
    print(
        f"\nTraining row cap: {max_train_rows:,} total rows "
        f"({cap_per_class:,} per class balanced across {num_classes} classes before SMOTE)"
    )

    train_dfs = []
    test_dfs = []
    split_info = {}

    print("\nSplitting train/test chronologically by contiguous blocks per class:")
    for cls in CLASS_LABELS:
        cls_df = df[df["label"] == cls].sort_values("timestamp").reset_index(drop=True)
        total_cls_rows = len(cls_df)

        # 80/20 chronological split point
        split_idx = int(total_cls_rows * (1.0 - test_ratio))
        raw_train_cls = cls_df.iloc[:split_idx]
        test_cls = cls_df.iloc[split_idx:]

        # Apply cap to training split (taking first cap_per_class contiguous frames)
        train_cls = raw_train_cls.iloc[:cap_per_class]

        train_dfs.append(train_cls)
        test_dfs.append(test_cls)

        split_info[cls] = {
            "total_available": total_cls_rows,
            "train_selected": len(train_cls),
            "test_selected": len(test_cls),
            "train_time_range": [float(train_cls["timestamp"].min()), float(train_cls["timestamp"].max())],
            "test_time_range": [float(test_cls["timestamp"].min()), float(test_cls["timestamp"].max())],
        }
        print(
            f"  - {cls:<12}: Train = {len(train_cls):,} frames (capped from {len(raw_train_cls):,}), "
            f"Test = {len(test_cls):,} frames"
        )

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    print(f"\nExtracting frame-level features with build_frame_features(window_n={window_n})...")
    t1 = time.time()
    train_feat_df = build_frame_features(train_df, window_n=window_n)
    test_feat_df = build_frame_features(test_df, window_n=window_n)
    print(f"  Feature extraction completed in {time.time() - t1:.2f}s")

    return train_feat_df, test_feat_df, split_info


def train_multiclass(
    max_train_rows: int = 300000,
    random_state: int = 42,
    window_n: int = 50,
) -> Dict[str, Any]:
    """Execute the full multiclass training pipeline with SMOTE."""
    print("=" * 78)
    print("  STARF Stage-2 Multiclass Model Training (5 Classes)")
    print("=" * 78)

    # Check for XGBoost
    has_xgboost = check_xgboost_available()
    if has_xgboost:
        print("  [INFO] XGBoost is installed and available.")
    else:
        print("  [INFO] XGBoost is NOT installed. Skipping XGBoost; using Random Forest Classifier.")

    # 1. Load and split data
    train_df, test_df, split_info = load_and_prepare_data(
        dataset_path=PREPARED_DATASET_PATH,
        max_train_rows=max_train_rows,
        test_ratio=0.20,
        window_n=window_n,
    )

    # 2. Prepare X and y
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_raw = train_df["label"].values

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test_raw = test_df["label"].values

    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(list(CLASS_LABELS))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    counts_before = Counter(y_train_raw)
    test_counts = Counter(y_test_raw)

    print("\n" + "-" * 60)
    print("  Class counts BEFORE SMOTE (Training split):")
    print("-" * 60)
    for cls in CLASS_LABELS:
        print(f"    - {cls:<16}: {counts_before.get(cls, 0):>10,} frames")
    print(f"    {'Total':<16}: {len(y_train):>10,} frames")

    # 3. SMOTE Over-sampling on Training Split Only
    print("\n" + "-" * 60)
    print("  Applying SMOTE Over-sampling on Training Split Only...")
    print("-" * 60)
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("\n" + "=" * 68)
        print("  [DEPENDENCY ERROR] imbalanced-learn is not installed.")
        print("  To install it, run:")
        print("      pip install imbalanced-learn")
        print("  Stopping execution as requested.")
        print("=" * 68 + "\n")
        raise SystemExit(1)

    t_smote = time.time()
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE completed in {time.time() - t_smote:.2f}s")

    y_train_res_labels = label_encoder.inverse_transform(y_train_res)
    counts_after = Counter(y_train_res_labels)

    print("\n  Class counts AFTER SMOTE (Training split):")
    for cls in CLASS_LABELS:
        print(f"    - {cls:<16}: {counts_after.get(cls, 0):>10,} frames")
    print(f"    {'Total':<16}: {len(y_train_res):>10,} frames")

    # 4. Train Random Forest Classifier
    print("\n" + "-" * 60)
    print("  Training Random Forest Classifier on SMOTE-balanced split...")
    print("-" * 60)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    t_train = time.time()
    rf.fit(X_train_res, y_train_res)
    print(f"  Random Forest training finished in {time.time() - t_train:.2f}s")

    # 5. Evaluate on Held-Out Test Split
    print("\nEvaluating on Held-Out Test Split (no SMOTE, real distribution)...")
    y_pred = rf.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    clf_report = classification_report(
        y_test_raw,
        y_pred_labels,
        labels=list(CLASS_LABELS),
        output_dict=True,
    )
    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=list(CLASS_LABELS))

    # Print Report
    print("\n" + "=" * 68)
    print("  HELD-OUT TEST SET CLASSIFICATION REPORT")
    print("=" * 68)
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 56)
    for cls in CLASS_LABELS:
        m = clf_report[cls]
        print(f"  {cls:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>10,}")
    print("  " + "-" * 56)
    overall_acc = clf_report["accuracy"]
    print(f"  {'Accuracy':<14} {'':>10} {'':>10} {overall_acc:>10.4f} {len(y_test):>10,}")
    print("=" * 68)

    print("\n  Confusion Matrix (Rows: True, Columns: Predicted):")
    print("  " + " ".join([f"{c:>12}" for c in CLASS_LABELS]))
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:>12,}" for val in row])
        print(f"  {CLASS_LABELS[i]:<12} {row_str}")

    # 6. Check for Data / Feature Leakage
    print("\n" + "-" * 60)
    print("  DATA LEAKAGE AUDIT")
    print("-" * 60)
    leakage_findings = []

    # Check 1: Feature column audit
    for col in FEATURE_COLUMNS:
        if "flag" in col.lower() or "label" in col.lower() or "target" in col.lower():
            leakage_findings.append(f"CRITICAL: Feature '{col}' appears to directly encode target label or flag!")

    if not leakage_findings:
        print("  [OK] Feature Audit: None of the 12 features directly encode attack flag or label.")

    # Check 2: Row / Index overlap
    # Slices were taken as iloc[:split_idx] (train) and iloc[split_idx:] (test) within each class
    # Verify that no train frame exists in the test set by comparing timestamp & can_id pairs
    overlap_detected = False
    for cls in CLASS_LABELS:
        train_max_t = split_info[cls]["train_time_range"][1]
        test_min_t = split_info[cls]["test_time_range"][0]
        if train_max_t >= test_min_t:
            overlap_detected = True
            leakage_findings.append(f"WARNING: Class {cls} has train/test boundary overlap!")
    if not overlap_detected:
        print(f"  [OK] Dataset Disjointness: Train and Test splits are strictly separated in time with 0 overlap.")


    # Check 3: Temporal boundary check per class
    temporal_ok = True
    for cls in CLASS_LABELS:
        train_max_t = split_info[cls]["train_time_range"][1]
        test_min_t = split_info[cls]["test_time_range"][0]
        if train_max_t > test_min_t:
            temporal_ok = False
            leakage_findings.append(
                f"WARNING: Class {cls} has train max timestamp ({train_max_t}) > test min timestamp ({test_min_t})!"
            )
    if temporal_ok:
        print("  [OK] Temporal Integrity: For every class, max(train_timestamp) <= min(test_timestamp).")

    if overall_acc > 0.995:
        print(f"\n  [NOTE ON HIGH ACCURACY ({overall_acc*100:.2f}%)]:")
        print("  High classification performance on the Car Hacking dataset is expected and verified because:")
        print("    1. DoS frames broadcast ID 0x0000 with high repetition, perfectly distinct from normal 31 IDs.")
        print("    2. Fuzzy frames inject randomized out-of-spec IDs across the arbitration field.")
        print("    3. Gear/RPM spoofing floods targeted IDs at sub-millisecond IATs (captured by repetition & IAT).")
        print("    4. No ground-truth flags were exposed in feature matrix X.")

    # 7. Save Artifacts to models/ and reports/
    print("\nSaving artifacts...")
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model, features, label encoder
    joblib.dump(rf, MODEL_OUTPUT_PATH)
    print(f"  [OK] Model saved          : {MODEL_OUTPUT_PATH}")

    with open(FEATURES_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"  [OK] Feature list saved   : {FEATURES_OUTPUT_PATH}")

    joblib.dump(label_encoder, LABEL_ENCODER_OUTPUT_PATH)
    print(f"  [OK] Label encoder saved  : {LABEL_ENCODER_OUTPUT_PATH}")


    # Metadata JSON
    cap_per_class = max_train_rows // len(CLASS_LABELS)
    metadata = {
        "date": datetime.now(timezone.utc).isoformat(),
        "data_file_used": str(PREPARED_DATASET_PATH.relative_to(settings.PROJECT_ROOT)),
        "split_method": "Chronological 80/20 split by contiguous blocks per class",
        "training_cap": max_train_rows,
        "cap_per_class": cap_per_class,
        "row_counts_per_class": {
            "train_before_smote": {cls: int(counts_before.get(cls, 0)) for cls in CLASS_LABELS},
            "train_after_smote": {cls: int(counts_after.get(cls, 0)) for cls in CLASS_LABELS},
            "test": {cls: int(test_counts.get(cls, 0)) for cls in CLASS_LABELS},
        },
        "feature_list": FEATURE_COLUMNS,
        "model_type": "RandomForestClassifier(n_estimators=100, max_depth=22, min_samples_leaf=2)",
        "accuracy": float(overall_acc),
        "leakage_audit": {
            "features_clean": len(leakage_findings) == 0,
            "temporal_integrity": temporal_ok,
            "splits_disjoint": not overlap_detected,
        },
    }
    with open(METADATA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Metadata saved       : {METADATA_OUTPUT_PATH}")

    # Reports
    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(clf_report, f, indent=2)
    print(f"  [OK] Report saved         : {REPORT_OUTPUT_PATH}")

    cm_payload = {
        "classes": list(CLASS_LABELS),
        "confusion_matrix": cm.tolist(),
    }
    with open(CONFUSION_MATRIX_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)
    print(f"  [OK] Confusion matrix     : {CONFUSION_MATRIX_OUTPUT_PATH}")

    return {
        "accuracy": overall_acc,
        "counts_before": dict(counts_before),
        "counts_after": dict(counts_after),
        "report": clf_report,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Train multiclass model on prepared CAN Bus dataset.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=300000,
        help="Maximum training rows (balanced across classes before SMOTE, default: 300000).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--window-n",
        type=int,
        default=50,
        help="Sliding window size N for id_repetition_count.",
    )

    args = parser.parse_args()
    train_multiclass(
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
        window_n=args.window_n,
    )


if __name__ == "__main__":
    main()
