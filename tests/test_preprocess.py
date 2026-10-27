"""Unit tests for feature builder and repetition count in src/preprocess.py."""

import pandas as pd
import pytest

from src.preprocess import build_frame_features


def test_id_repetition_count_handmade_example():
    """Verify id_repetition_count on a small, hand-crafted CAN message sequence."""
    # Sequence of 6 frames with N=3:
    # Frame 0: 0x100 -> window=[0x100]               -> count = 1
    # Frame 1: 0x200 -> window=[0x100, 0x200]        -> count = 1
    # Frame 2: 0x100 -> window=[0x100, 0x200, 0x100] -> count = 2
    # Frame 3: 0x100 -> window=[0x200, 0x100, 0x100] -> count = 2 (first 0x100 was evicted)
    # Frame 4: 0x300 -> window=[0x100, 0x100, 0x300] -> count = 1
    # Frame 5: 0x100 -> window=[0x100, 0x300, 0x100] -> count = 2
    data = {
        "timestamp": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05],
        "can_id": [0x100, 0x200, 0x100, 0x100, 0x300, 0x100],
        "dlc": [8, 8, 8, 8, 8, 8],
        "d0": ["00", "01", "02", "03", "04", "05"],
        "d1": ["00"] * 6,
        "d2": ["00"] * 6,
        "d3": ["00"] * 6,
        "d4": ["00"] * 6,
        "d5": ["00"] * 6,
        "d6": ["00"] * 6,
        "d7": ["00"] * 6,
    }
    df = pd.DataFrame(data)

    feat_df = build_frame_features(df, window_n=3)

    assert "id_repetition_count" in feat_df.columns
    expected_rep_counts = [1, 1, 2, 2, 1, 2]
    assert list(feat_df["id_repetition_count"]) == expected_rep_counts


def test_inter_arrival_time_and_byte_parsing():
    """Verify IAT delta calculation and hex byte conversion."""
    data = {
        "timestamp": [10.0, 10.01, 10.05],
        "can_id": [0x120, 0x340, 0x120],
        "dlc": [5, 8, 2],
        "d0": ["ff", "00", "0a"],
        "d1": ["a0", "00", "0b"],
        "d2": ["01", "00", ""],
        "d3": ["02", "00", ""],
        "d4": ["03", "00", ""],
        "d5": ["", "00", ""],
        "d6": ["", "00", ""],
        "d7": ["", "00", ""],
    }
    df = pd.DataFrame(data)

    feat_df = build_frame_features(df, window_n=50)

    # First occurrence of 0x120: iat = 0.0
    # First occurrence of 0x340: iat = 0.0
    # Second occurrence of 0x120 at 10.05 (previous was 10.0): iat = 0.05
    assert feat_df["inter_arrival_time_delta"].iloc[0] == 0.0
    assert feat_df["inter_arrival_time_delta"].iloc[1] == 0.0
    assert pytest.approx(feat_df["inter_arrival_time_delta"].iloc[2], rel=1e-4) == 0.05

    # Check d0..d7 byte parsing (hex -> integer 0-255, empty -> 0)
    assert feat_df["d0"].iloc[0] == 255  # 'ff' -> 255
    assert feat_df["d1"].iloc[0] == 160  # 'a0' -> 160
    assert feat_df["d5"].iloc[0] == 0    # empty -> 0
    assert feat_df["dlc"].iloc[0] == 5
