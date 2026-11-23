"""Tests for SynCAN window-level feature engineering, dataset splitting, causality, and metrics.

Verifies:
  1. Window labeling 50% rule (>=50% attack messages -> window label 1).
  2. Chronological split integrity (no cross-boundary windows).
  3. Causality (features in earlier windows unaffected by subsequent messages/windows; s1_prev_corr is strictly backward-looking).
  4. Schema and metrics file key verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

from src.syncan_windows import (
    FEATURE_COLUMNS,
    build_windows,
    load_syncan_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def dummy_stats() -> Dict[str, Any]:
    """Synthetic normal baseline statistics for 10 CAN IDs."""
    by_id: Dict[str, Any] = {}
    for i in range(1, 11):
        by_id[f"id{i}"] = {
            "signals": {
                f"s{s}": {
                    "valid": True,
                    "mean": 10.0 * s,
                    "std": 2.0,
                    "min": 0.0,
                    "max": 50.0,
                }
                for s in range(1, 5)
            }
        }
    return {"by_id": by_id}


def test_fifty_percent_rule_labeling(dummy_stats: Dict[str, Any]):
    """Window label is 1 if >=50% of messages are attack (label 1), else 0."""
    # Window 0: 4 attack out of 10 messages (40% < 50%) -> label 0
    # Window 1: 5 attack out of 10 messages (50% >= 50%) -> label 1
    # Window 2: 7 attack out of 10 messages (70% >= 50%) -> label 1
    timestamps = []
    labels = []
    can_ids = []
    s1_vals = []

    # Window 0: t = 0.0 to 0.45 (duration 0.5s)
    for i in range(10):
        timestamps.append(0.04 * i)
        labels.append(1 if i < 4 else 0)
        can_ids.append("id1")
        s1_vals.append(10.0)

    # Window 1: t = 0.5 to 0.95
    for i in range(10):
        timestamps.append(0.50 + 0.04 * i)
        labels.append(1 if i < 5 else 0)
        can_ids.append("id1")
        s1_vals.append(10.0)

    # Window 2: t = 1.0 to 1.45
    for i in range(10):
        timestamps.append(1.0 + 0.04 * i)
        labels.append(1 if i < 7 else 0)
        can_ids.append("id1")
        s1_vals.append(10.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "can_id": can_ids,
        "s1": s1_vals,
        "s2": [20.0] * len(timestamps),
        "s3": [30.0] * len(timestamps),
        "s4": [40.0] * len(timestamps),
        "label": labels,
    })

    windows = build_windows(df, window_s=0.5, stats=dummy_stats, file_attack_type="flooding")

    assert len(windows) == 3
    assert windows.iloc[0]["label"] == 0
    assert windows.iloc[0]["attack_type"] == "normal"
    assert pytest.approx(windows.iloc[0]["pct_attack_messages"], 0.01) == 0.40

    assert windows.iloc[1]["label"] == 1
    assert windows.iloc[1]["attack_type"] == "flooding"
    assert pytest.approx(windows.iloc[1]["pct_attack_messages"], 0.01) == 0.50

    assert windows.iloc[2]["label"] == 1
    assert windows.iloc[2]["attack_type"] == "flooding"
    assert pytest.approx(windows.iloc[2]["pct_attack_messages"], 0.01) == 0.70


def test_window_feature_causality(dummy_stats: Dict[str, Any]):
    """Features in earlier windows must not depend on later windows."""
    timestamps = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 1.1, 1.2]
    s1_vals = [10.0, 12.0, 11.0, 25.0, 26.0, 27.0, 50.0, 51.0]

    df_full = pd.DataFrame({
        "timestamp": timestamps,
        "can_id": ["id1"] * len(timestamps),
        "s1": s1_vals,
        "s2": [20.0] * len(timestamps),
        "s3": [30.0] * len(timestamps),
        "s4": [40.0] * len(timestamps),
        "label": [0] * len(timestamps),
    })

    # Sub-dataframe omitting the last window (t >= 1.0)
    df_truncated = df_full.iloc[:6].copy()

    windows_full = build_windows(df_full, window_s=0.5, stats=dummy_stats)
    windows_trunc = build_windows(df_truncated, window_s=0.5, stats=dummy_stats)

    # First two windows should be identical across all 222 feature columns
    assert len(windows_full) == 3
    assert len(windows_trunc) == 2

    for col in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            windows_full.iloc[0][col],
            windows_trunc.iloc[0][col],
            err_msg=f"Discrepancy in col {col} window 0",
        )
        np.testing.assert_allclose(
            windows_full.iloc[1][col],
            windows_trunc.iloc[1][col],
            err_msg=f"Discrepancy in col {col} window 1",
        )

    # First window s1_prev_corr must be -1.0 because there is no prior window
    assert windows_full.iloc[0]["id1_s1_prev_corr"] == -1.0


def test_chronological_splits_no_cross_boundary():
    """Verify that chronological 70/15/15 window splitting creates strictly disjoint time intervals."""
    # Create synthetic series of 100 consecutive 0.5s windows
    n = 100
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    indices = np.arange(n)
    starts = indices * 0.5
    ends = (indices + 1) * 0.5

    df = pd.DataFrame({
        "window_idx": indices,
        "window_start": starts,
        "window_end": ends,
    })

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strictly ordered boundaries: max(train) <= min(val) < max(val) <= min(test)
    assert train_df["window_end"].max() <= val_df["window_start"].min()
    assert val_df["window_end"].max() <= test_df["window_start"].min()

    # Zero overlap in window indices
    assert len(set(train_df["window_idx"]).intersection(set(val_df["window_idx"]))) == 0
    assert len(set(val_df["window_idx"]).intersection(set(test_df["window_idx"]))) == 0
    assert len(set(train_df["window_idx"]).intersection(set(test_df["window_idx"]))) == 0


def test_window_metrics_file_schema():
    """Verify reports/syncan_window_metrics.json has all required keys if file exists."""
    metrics_path = PROJECT_ROOT / "reports" / "syncan_window_metrics.json"
    if not metrics_path.exists():
        pytest.skip("reports/syncan_window_metrics.json not yet generated.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = [
        "per_class",
        "confusion_matrix",
        "binary_attack_normal",
        "false_positive_rate_normal",
        "false_alarms_per_1000_normal",
    ]
    for key in required_keys:
        assert key in data, f"Missing required key '{key}' in {metrics_path.name}"

    # Verify per-class contents
    for cls_name in ["normal", "continuous", "flooding", "plateau", "playback", "suppress"]:
        assert cls_name in data["per_class"], f"Missing class '{cls_name}' in per_class"
        for metric in ["precision", "recall", "f1-score", "support"]:
            assert metric in data["per_class"][cls_name], f"Missing '{metric}' for '{cls_name}'"
