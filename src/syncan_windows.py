"""Fixed-duration window-level feature engineering for the SynCAN benchmark dataset.

Transforms continuous CAN message streams into non-overlapping fixed time windows
(default: window_s = 0.5 seconds). Computes ID-specific timing, distribution, drift,
plateau, and correlation features alongside bus-level load and timing metrics.

All features are strictly causal with zero look-ahead across future windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

DEFAULT_STATS_PATH = Path(__file__).resolve().parent.parent / "models" / "syncan_stats.json"

CAN_IDS = [f"id{i}" for i in range(1, 11)]

# Pre-define feature names (22 per ID * 10 IDs + 2 bus features = 222 features)
FEATURE_COLUMNS: List[str] = ["bus_total_count", "bus_max_gap"]
for cid_num in range(1, 11):
    prefix = f"id{cid_num}_"
    FEATURE_COLUMNS.extend([
        f"{prefix}present",
        f"{prefix}count",
        f"{prefix}dt_mean",
        f"{prefix}dt_std",
    ])
    for s_i in range(1, 5):
        FEATURE_COLUMNS.extend([
            f"{prefix}s{s_i}_max_abs_z",
            f"{prefix}s{s_i}_mean_z",
            f"{prefix}s{s_i}_std_ratio",
            f"{prefix}s{s_i}_frac_repeat",
        ])
    FEATURE_COLUMNS.extend([
        f"{prefix}s1_slope",
        f"{prefix}s1_prev_corr",
    ])

FEATURE_NAME_TO_IDX = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}


def load_syncan_stats(stats_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load baseline normal statistics precomputed on train_normal."""
    p = Path(stats_path or DEFAULT_STATS_PATH)
    if not p.exists():
        raise FileNotFoundError(f"SynCAN baseline stats not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_windows(
    df: pd.DataFrame,
    window_s: float = 0.5,
    stats: Optional[Dict[str, Any]] = None,
    file_attack_type: Optional[str] = None,
) -> pd.DataFrame:
    """Build non-overlapping time window feature rows from time-ordered message dataframe.

    Args:
        df: Input DataFrame of CAN messages containing timestamp, can_id, s1..s4, label.
        window_s: Window duration in seconds (default: 0.5s).
        stats: Baseline statistics dict (loaded from models/syncan_stats.json if None).
        file_attack_type: Ground truth attack type for attack messages (e.g. 'flooding').
                          If None, inferred from df['attack_type'] if present.

    Returns:
        DataFrame with one row per non-overlapping time window containing 222 features,
        window metadata, window label, and attack_type.
    """
    if len(df) == 0:
        return pd.DataFrame()

    if stats is None:
        stats = load_syncan_stats()

    stats_by_id = stats.get("by_id", {})

    # Infer file attack type if not provided
    if file_attack_type is None:
        if "attack_type" in df.columns:
            non_normal = df.loc[df["attack_type"] != "normal", "attack_type"]
            file_attack_type = str(non_normal.iloc[0]) if len(non_normal) > 0 else "normal"
        else:
            file_attack_type = "normal"

    # Extract NumPy arrays for fast vectorized slicing
    ts = df["timestamp"].values.astype(np.float64)
    can_id_strs = df["can_id"].astype(str).values
    # Map 'id1'..'id10' to integer 1..10
    cid_ints = np.array([int(c.replace("id", "")) if c.startswith("id") else 0 for c in can_id_strs], dtype=np.int16)

    s1 = df["s1"].values.astype(np.float32)
    s2 = df["s2"].values.astype(np.float32)
    s3 = df["s3"].values.astype(np.float32)
    s4 = df["s4"].values.astype(np.float32)
    signals = [s1, s2, s3, s4]

    labels = df["label"].values.astype(np.int8) if "label" in df.columns else np.zeros(len(df), dtype=np.int8)

    # Compute non-overlapping window indices
    win_indices = np.floor(ts / window_s).astype(np.int64)
    unique_wins, win_starts = np.unique(win_indices, return_index=True)
    win_ends = np.append(win_starts[1:], len(df))

    num_windows = len(unique_wins)

    # Pre-allocate 2D NumPy array for all 222 features (default -1.0)
    feat_matrix = np.full((num_windows, len(FEATURE_COLUMNS)), -1.0, dtype=np.float32)

    # Metadata columns
    win_starts_t = np.empty(num_windows, dtype=np.float64)
    win_ends_t = np.empty(num_windows, dtype=np.float64)
    total_messages = np.empty(num_windows, dtype=np.int32)
    pct_attack_messages = np.empty(num_windows, dtype=np.float32)
    win_labels = np.zeros(num_windows, dtype=np.int8)
    win_attack_types = ["normal"] * num_windows

    # Causal state tracking previous window's signal 1 for each ID (for correlation)
    prev_s1_by_id: Dict[int, np.ndarray] = {}

    # Precompute norm_mean and norm_std arrays per ID (10, 4)
    norm_means = np.zeros((11, 5), dtype=np.float32)
    norm_stds = np.ones((11, 5), dtype=np.float32)
    sig_valids = np.zeros((11, 5), dtype=bool)

    for cid_num in range(1, 11):
        id_stat = stats_by_id.get(f"id{cid_num}", {}).get("signals", {})
        for s_i in range(1, 5):
            s_dict = id_stat.get(f"s{s_i}", {})
            if s_dict.get("valid", False):
                sig_valids[cid_num, s_i] = True
                norm_means[cid_num, s_i] = float(s_dict.get("mean", 0.0))
                std_v = float(s_dict.get("std", 1.0))
                norm_stds[cid_num, s_i] = std_v if std_v > 1e-6 else 1e-6

    # Iterate over windows
    for w_i in range(num_windows):
        st = win_starts[w_i]
        en = win_ends[w_i]
        w_idx = unique_wins[w_i]

        win_st = float(w_idx * window_s)
        win_en = float((w_idx + 1) * window_s)
        win_starts_t[w_i] = win_st
        win_ends_t[w_i] = win_en

        total_msg = en - st
        total_messages[w_i] = total_msg

        ts_slice = ts[st:en]
        cid_slice = cid_ints[st:en]
        label_slice = labels[st:en]

        # Bus-level metrics
        feat_matrix[w_i, 0] = float(total_msg)
        if total_msg >= 2:
            feat_matrix[w_i, 1] = float(np.max(np.diff(ts_slice)))

        # Window ground-truth label (50% rule)
        pct_atk = float(np.mean(label_slice == 1))
        pct_attack_messages[w_i] = pct_atk
        if pct_atk >= 0.50:
            win_labels[w_i] = 1
            win_attack_types[w_i] = file_attack_type

        # Per-ID features
        for cid_num in range(1, 11):
            matches = np.where(cid_slice == cid_num)[0]
            count_c = len(matches)

            base_offset = 2 + (cid_num - 1) * 22

            if count_c == 0:
                feat_matrix[w_i, base_offset] = 0.0  # present
                feat_matrix[w_i, base_offset + 1] = 0.0  # count
                continue

            feat_matrix[w_i, base_offset] = 1.0  # present
            feat_matrix[w_i, base_offset + 1] = float(count_c)  # count

            abs_indices = st + matches
            t_c = ts[abs_indices]

            # Timing
            if count_c >= 2:
                dt = np.diff(t_c)
                feat_matrix[w_i, base_offset + 2] = float(np.mean(dt))
                feat_matrix[w_i, base_offset + 3] = float(np.std(dt))

            # Signals 1..4
            s1_curr: Optional[np.ndarray] = None

            for s_i in range(1, 5):
                sig_offset = base_offset + 4 + (s_i - 1) * 4
                if not sig_valids[cid_num, s_i]:
                    continue

                s_sig = signals[s_i - 1][abs_indices]
                if s_i == 1:
                    s1_curr = s_sig

                n_mean = norm_means[cid_num, s_i]
                n_std = norm_stds[cid_num, s_i]

                # z-scores
                z = (s_sig - n_mean) / n_std
                feat_matrix[w_i, sig_offset] = float(np.max(np.abs(z)))
                feat_matrix[w_i, sig_offset + 1] = float(np.mean(z))

                # std ratio & frac repeat
                if count_c >= 2:
                    win_std = float(np.std(s_sig))
                    feat_matrix[w_i, sig_offset + 2] = float(win_std / n_std)
                    feat_matrix[w_i, sig_offset + 3] = float(np.mean(s_sig[1:] == s_sig[:-1]))
                else:
                    feat_matrix[w_i, sig_offset + 3] = 0.0

            # Absolute slope of signal 1 (drift rate)
            if count_c >= 2 and s1_curr is not None:
                t_diff = t_c - np.mean(t_c)
                s_diff = s1_curr - np.mean(s1_curr)
                t_var = float(np.sum(t_diff**2))
                if t_var > 1e-12:
                    feat_matrix[w_i, base_offset + 20] = abs(float(np.sum(t_diff * s_diff) / t_var))
                else:
                    feat_matrix[w_i, base_offset + 20] = 0.0

            # Correlation with same ID's previous window (playback detection)
            prev_s1 = prev_s1_by_id.get(cid_num)
            if prev_s1 is not None and s1_curr is not None and len(prev_s1) >= 2 and len(s1_curr) >= 2:
                min_len = min(len(prev_s1), len(s1_curr))
                s_prev_slice = prev_s1[:min_len]
                s_curr_slice = s1_curr[:min_len]
                std_prev = float(np.std(s_prev_slice))
                std_curr = float(np.std(s_curr_slice))

                if std_prev < 1e-8 or std_curr < 1e-8:
                    feat_matrix[w_i, base_offset + 21] = 1.0 if np.isclose(s_prev_slice[0], s_curr_slice[0]) else 0.0
                else:
                    r_val = float(np.corrcoef(s_prev_slice, s_curr_slice)[0, 1])
                    feat_matrix[w_i, base_offset + 21] = 0.0 if np.isnan(r_val) else r_val

            # Update causal state
            if s1_curr is not None and len(s1_curr) > 0:
                prev_s1_by_id[cid_num] = s1_curr.copy()

    # Build final DataFrame
    out_dict: Dict[str, Any] = {
        "window_idx": unique_wins,
        "window_start": win_starts_t,
        "window_end": win_ends_t,
        "total_messages": total_messages,
        "pct_attack_messages": pct_attack_messages,
    }
    for col_idx, col_name in enumerate(FEATURE_COLUMNS):
        out_dict[col_name] = feat_matrix[:, col_idx]

    out_dict["label"] = win_labels
    out_dict["attack_type"] = win_attack_types

    return pd.DataFrame(out_dict)
