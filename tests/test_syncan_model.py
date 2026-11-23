"""Tests for SynCAN XGBoost model training, chronological split integrity, and metrics schema.

Covers:
  1. Chronological split validation: zero time overlap across train, val, test per file.
  2. Synthetic lightweight training run completion (tiny dataset, fast execution).
  3. Schema verification for reports/syncan_xgb_metrics.json (message and window level metrics).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from xgboost import XGBClassifier

from scripts.train_syncan_xgboost import (
    ATTACK_CLASSES,
    FEATURE_COLUMNS,
    NORMAL_FILES,
    PROCESSED_DIR,
    SCENARIO_FILES,
    SYNCAN_CLASSES,
    compute_message_metrics,
    compute_window_metrics_1s,
    extract_features_matrix,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_METRICS_PATH = _PROJECT_ROOT / "reports" / "syncan_xgb_metrics.json"
_MODEL_PATH = _PROJECT_ROOT / "models" / "syncan_xgb_model.pkl"
_META_PATH = _PROJECT_ROOT / "models" / "syncan_xgb_meta.json"


# ---------------------------------------------------------------------------
# 1. Chronological split integrity: zero overlap per file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not PROCESSED_DIR.exists() or not (PROCESSED_DIR / "test_flooding.parquet").exists(),
    reason="SynCAN processed parquet files not found",
)
@pytest.mark.parametrize("fname", SCENARIO_FILES + NORMAL_FILES)
def test_chronological_splits_no_overlap(fname: str):
    """Verify strictly monotonic, non-overlapping chronological 70/15/15 splits per file."""
    fpath = PROCESSED_DIR / fname
    assert fpath.exists(), f"Missing scenario file: {fpath}"

    ts = pq.read_table(fpath, columns=["timestamp"])["timestamp"].to_numpy()
    n = len(ts)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_ts = ts[:n_train]
    val_ts = ts[n_train : n_train + n_val]
    test_ts = ts[n_train + n_val :]

    assert len(train_ts) > 0
    assert len(val_ts) > 0
    assert len(test_ts) > 0

    max_train = float(train_ts.max())
    min_val = float(val_ts.min())
    max_val = float(val_ts.max())
    min_test = float(test_ts.min())

    # Strictly non-overlapping chronological boundaries
    assert max_train <= min_val, (
        f"{fname}: Train max timestamp ({max_train}) exceeds Val min timestamp ({min_val})"
    )
    assert max_val <= min_test, (
        f"{fname}: Val max timestamp ({max_val}) exceeds Test min timestamp ({min_test})"
    )


# ---------------------------------------------------------------------------
# 2. Synthetic lightweight training run completion
# ---------------------------------------------------------------------------

def test_synthetic_syncan_training_run():
    """Verify that feature extraction, XGBoost training, and window aggregation complete without error."""
    from sklearn.preprocessing import LabelEncoder

    rng = np.random.default_rng(42)
    n_samples = 300

    # Build synthetic dataframe with all required columns
    records = []
    classes_list = list(SYNCAN_CLASSES)
    for i in range(n_samples):
        cls = classes_list[i % len(classes_list)]
        is_attack = 1 if cls != "normal" else 0
        cid = f"id{(i % 10) + 1}"
        ts = 100.0 + (i * 0.02)  # 20ms apart

        rec = {
            "timestamp": ts,
            "can_id": cid,
            "s1": float(rng.uniform(0, 1)),
            "s2": float(rng.uniform(0, 1)),
            "s3": float(rng.uniform(0, 1)),
            "s4": float(rng.uniform(0, 1)),
            "s1_valid": 1,
            "s2_valid": 1,
            "s3_valid": 0,
            "s4_valid": 0,
            "label": is_attack,
            "attack_type": cls,
            "dt_same_id": 0.015,
            "bus_rate_50ms": 25,
            "bus_rate_1s": 500,
            "file_source": f"test_{cls}" if is_attack else "test_normal",
            "plateau_max_repeat": int(rng.integers(1, 10)),
        }
        for s in (1, 2, 3, 4):
            rec[f"s{s}_delta"] = 0.01
            rec[f"s{s}_mean_10"] = 0.5
            rec[f"s{s}_std_10"] = 0.1
            rec[f"s{s}_zscore"] = 0.2
            rec[f"s{s}_repeat_count"] = 1
        records.append(rec)

    synth_df = pd.DataFrame(records)

    # Feature extraction
    X = extract_features_matrix(synth_df)
    assert X.shape == (n_samples, len(FEATURE_COLUMNS))
    assert not np.isnan(X).any()

    # Fit small XGBoost model
    le = LabelEncoder().fit(list(SYNCAN_CLASSES))
    y = le.transform(synth_df["attack_type"].values)

    clf = XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )
    clf.fit(X, y)

    # Message metrics
    msg_m, y_pred = compute_message_metrics(clf, X, synth_df["attack_type"].values, le)
    assert 0.0 <= msg_m["accuracy"] <= 1.0
    assert "per_class" in msg_m
    assert "per_attack_recall" in msg_m

    # Window metrics
    win_m = compute_window_metrics_1s(synth_df, y_pred)
    assert win_m["total_windows"] > 0
    assert "per_attack" in win_m


# ---------------------------------------------------------------------------
# 3. Schema verification for reports/syncan_xgb_metrics.json
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _METRICS_PATH.exists(),
    reason="reports/syncan_xgb_metrics.json not yet generated",
)
def test_syncan_metrics_file_has_required_keys():
    """Verify reports/syncan_xgb_metrics.json contains all required fields and sub-objects."""
    with open(_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Top-level keys
    for key in ("date", "model_type", "evaluation_note", "training", "message_level", "window_level_1s"):
        assert key in data, f"Missing top-level key: {key}"

    # Required note content
    assert "window-tagged" in data["evaluation_note"].lower()

    # Training metadata
    training = data["training"]
    for k in ("candidate_train_rows", "subsampling_k", "final_train_rows", "val_rows", "test_rows", "best_params"):
        assert k in training, f"Missing training key: {k}"
    assert training["subsampling_k"] >= 1
    assert training["final_train_rows"] > 0

    # Message-level schema
    msg = data["message_level"]
    for k in ("accuracy", "macro_f1", "weighted_f1", "false_positive_rate_normal", "false_alarms_per_100k_normal", "per_class", "per_attack_recall", "confusion_matrix"):
        assert k in msg, f"Missing message_level key: {k}"

    for cls in SYNCAN_CLASSES:
        assert cls in msg["per_class"], f"Missing class in message per_class: {cls}"
        for metric in ("precision", "recall", "f1-score", "support"):
            assert metric in msg["per_class"][cls]

    for atk in ATTACK_CLASSES:
        assert atk in msg["per_attack_recall"], f"Missing attack in per_attack_recall: {atk}"

    # Window-level schema
    win = data["window_level_1s"]
    for k in ("total_windows", "normal_windows", "attack_windows", "macro_f1", "false_positive_rate_normal", "false_alarms_per_100k_normal", "per_attack"):
        assert k in win, f"Missing window_level_1s key: {k}"

    for atk in ATTACK_CLASSES:
        assert atk in win["per_attack"], f"Missing attack in window per_attack: {atk}"
        for metric in ("precision", "recall", "f1-score", "fpr", "support_windows"):
            assert metric in win["per_attack"][atk]
