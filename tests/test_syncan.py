"""Tests for SynCAN dataset loader and causal feature extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.syncan_dataset import load_syncan, extract_attack_type_from_filename
from src.syncan_features import SynCANFeatureExtractor


EXPECTED_LOADER_COLUMNS = [
    "timestamp",
    "can_id",
    "s1",
    "s2",
    "s3",
    "s4",
    "s1_valid",
    "s2_valid",
    "s3_valid",
    "s4_valid",
    "label",
    "attack_type",
]


@pytest.fixture
def temp_csv_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for synthetic CSVs."""
    return tmp_path


def test_syncan_loader_schema_and_unit(temp_csv_dir: Path):
    """Test loader schema, timestamp conversion (ms to seconds), and column types."""
    # Synthetic file with header (like train_1 or test_flooding)
    csv_file = temp_csv_dir / "test_flooding.csv"
    csv_content = (
        "Label,Time,ID,Signal1,Signal2,Signal3,Signal4\n"
        "0,2000.0,id5,0.1,0.2,,\n"
        "1,2015.0,id5,0.15,0.25,,\n"
        "1,2030.0,id1,0.8,0.5,,\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df = load_syncan(csv_file)

    assert list(df.columns) == EXPECTED_LOADER_COLUMNS
    assert len(df) == 3
    # Time unit check: 2000.0 ms -> 2.000000 s
    assert np.isclose(df.iloc[0]["timestamp"], 2.0)
    assert np.isclose(df.iloc[1]["timestamp"], 2.015)
    # Attack type mapping: test_flooding -> 'flooding' for attack rows (label=1), 'normal' for label=0
    assert df.iloc[0]["attack_type"] == "normal"
    assert df.iloc[1]["attack_type"] == "flooding"
    assert df.iloc[2]["attack_type"] == "flooding"


def test_syncan_headerless_file_support(temp_csv_dir: Path):
    """Test loader support for headerless files with variable line widths (train_2..4)."""
    csv_file = temp_csv_dir / "train_2.csv"
    # id8 has 1 signal (4 columns), id1 has 2 signals (5 columns), id10 has 4 signals (7 columns)
    csv_content = (
        "0,15000.0,id8,0.179\n"
        "0,15015.0,id1,0.877,0.5\n"
        "0,15030.0,id10,0.29,0.77,0.32,0.43\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df = load_syncan(csv_file)

    assert len(df) == 3
    assert list(df.columns) == EXPECTED_LOADER_COLUMNS
    assert df.iloc[0]["can_id"] == "id8"
    assert df.iloc[1]["can_id"] == "id1"
    assert df.iloc[2]["can_id"] == "id10"
    assert df.iloc[0]["s1_valid"] == 1
    assert df.iloc[0]["s2_valid"] == 0


def test_syncan_nan_handling_and_validity_masks(temp_csv_dir: Path):
    """Test that missing signals are filled with -1.0 and validity masks correctly set."""
    csv_file = temp_csv_dir / "sample.csv"
    csv_content = (
        "Label,Time,ID,Signal1,Signal2,Signal3,Signal4\n"
        "0,1000.0,id4,0.42,,,\n"  # id4 has only 1 signal
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    df = load_syncan(csv_file)

    row = df.iloc[0]
    assert np.isclose(row["s1"], 0.42)
    assert row["s1_valid"] == 1

    assert np.isclose(row["s2"], -1.0)
    assert row["s2_valid"] == 0

    assert np.isclose(row["s3"], -1.0)
    assert row["s3_valid"] == 0

    assert np.isclose(row["s4"], -1.0)
    assert row["s4_valid"] == 0


def test_syncan_feature_causality(temp_csv_dir: Path):
    """Verify strictly causal feature extraction: row i must not change if subsequent rows are removed."""
    # Build 40 synthetic messages
    n_rows = 40
    times = [1000.0 + i * 2.5 for i in range(n_rows)]
    ids = ["id1", "id3", "id5", "id8"] * 10
    s1 = [0.1 * (i % 5) for i in range(n_rows)]
    s2 = [0.2 * (i % 3) for i in range(n_rows)]

    rows_str = ["Label,Time,ID,Signal1,Signal2,Signal3,Signal4"]
    for t, cid, v1, v2 in zip(times, ids, s1, s2):
        rows_str.append(f"0,{t},{cid},{v1},{v2},,")

    csv_file = temp_csv_dir / "causality_test.csv"
    csv_file.write_text("\n".join(rows_str), encoding="utf-8")

    df_full = load_syncan(csv_file)

    # Initialize extractor and fit dummy stats
    fe = SynCANFeatureExtractor()
    fe.fit(df_full, save_path=None)

    # Transform full 40 rows
    full_feat = fe.transform(df_full, reset_state=True)

    # Transform truncated slice (first 20 rows only)
    df_half = df_full.iloc[:20].copy()
    half_feat = fe.transform(df_half, reset_state=True)

    # Verify that row 0..19 in half_feat are exactly identical to row 0..19 in full_feat
    feature_cols = [
        "dt_same_id",
        "bus_rate_50ms",
        "bus_rate_1s",
        "s1_delta",
        "s1_mean_10",
        "s1_std_10",
        "s1_zscore",
        "s1_repeat_count",
        "plateau_max_repeat",
    ]

    for col in feature_cols:
        full_slice = full_feat[col].iloc[:20].to_numpy()
        half_slice = half_feat[col].to_numpy()
        assert np.allclose(full_slice, half_slice, equal_nan=True), f"Causality violation in {col}"


def test_syncan_id_wise_statistics(temp_csv_dir: Path):
    """Test normal baseline statistics fitting and z-score computation per ID."""
    csv_file = temp_csv_dir / "stats_test.csv"
    # Create 10 rows of id1 with constant signal1 = 2.0, signal2 = 4.0
    rows = ["Label,Time,ID,Signal1,Signal2,Signal3,Signal4"]
    for i in range(10):
        rows.append(f"0,{1000.0 + i * 15.0},id1,2.0,4.0,,")
    csv_file.write_text("\n".join(rows), encoding="utf-8")

    df = load_syncan(csv_file)
    fe = SynCANFeatureExtractor()
    stats_file = temp_csv_dir / "test_stats.json"
    fe.fit(df, save_path=stats_file)

    assert "id1" in fe.stats["by_id"]
    id1_info = fe.stats["by_id"]["id1"]
    assert np.isclose(id1_info["signals"]["s1"]["mean"], 2.0)
    assert np.isclose(id1_info["signals"]["s2"]["mean"], 4.0)
    assert np.isclose(id1_info["iat_mean"], 0.015)


def test_syncan_plateau_detection(temp_csv_dir: Path):
    """Test that repeated signal values correctly increment plateau repeat counters."""
    csv_file = temp_csv_dir / "plateau_test.csv"
    # 5 consecutive identical values for id1
    rows = ["Label,Time,ID,Signal1,Signal2,Signal3,Signal4"]
    for i in range(5):
        rows.append(f"0,{1000.0 + i * 15.0},id1,0.5,0.5,,")
    csv_file.write_text("\n".join(rows), encoding="utf-8")

    df = load_syncan(csv_file)
    fe = SynCANFeatureExtractor()
    fe.fit(df, save_path=None)
    feat = fe.transform(df, reset_state=True)

    # Repeats should increment from 1 up to 5
    assert list(feat["plateau_max_repeat"]) == [1, 2, 3, 4, 5]
