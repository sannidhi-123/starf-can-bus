"""Causal feature engineering for the SynCAN benchmark dataset.

Computes strictly causal, look-ahead-free features for CAN intrusion detection:
- dt_same_id: Inter-arrival time between consecutive messages of the same CAN ID.
- bus_rate_50ms, bus_rate_1s: Total bus message frequency in trailing 50 ms and 1 s windows.
- s1_delta..s4_delta: First-difference of each signal from its previous transmission for that ID.
- s1_mean_10..s4_mean_10, s1_std_10..s4_std_10: Rolling 10-message mean and standard deviation per signal per ID.
- s1_zscore..s4_zscore: Standardized deviation of signals against normal training baseline stats.
- s1_repeat_count..s4_repeat_count, plateau_max_repeat: Consecutive repeated signal values for plateau attack detection.

All state resets per file to ensure zero leakage across independent capture sessions.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


DEFAULT_STATS_PATH = Path(__file__).resolve().parent.parent / "models" / "syncan_stats.json"


class SynCANFeatureExtractor:
    """Causal window feature extractor for SynCAN message streams."""

    def __init__(self, stats: Optional[Dict[str, Any]] = None):
        """Initialize extractor with optional normal baseline statistics."""
        self.stats: Dict[str, Any] = stats or {}
        self.reset()

    def reset(self) -> None:
        """Reset all streaming state (sliding windows, caches, repeat counters)."""
        # Last timestamp seen per CAN ID
        self._last_ts_per_id: Dict[str, float] = {}
        # Last signal values seen per CAN ID: dict[str, list[float]]
        self._last_signals_per_id: Dict[str, List[float]] = {}
        # Sliding buffer of last 10 signal values per CAN ID: dict[str, list[deque]]
        self._signal_window_per_id: Dict[str, List[deque]] = {}
        # Repeated value counter per CAN ID: dict[str, list[int]]
        self._repeat_count_per_id: Dict[str, List[int]] = {}
        # Previous global timestamp seen
        self._last_global_ts: Optional[float] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        save_path: Optional[Union[str, Path]] = DEFAULT_STATS_PATH,
    ) -> SynCANFeatureExtractor:
        """Compute normal baseline statistics per ID and signal from training data.

        Args:
            train_df: DataFrame of normal training messages containing can_id, s1..s4, s1_valid..s4_valid.
            save_path: Optional path to save precomputed statistics JSON.

        Returns:
            self with updated baseline statistics.
        """
        stats: Dict[str, Any] = {
            "meta": {
                "total_rows": int(len(train_df)),
                "unique_ids": sorted(train_df["can_id"].unique().tolist()),
            },
            "by_id": {},
        }

        # Group by CAN ID and compute mean, std, min, max, count for valid signals
        for cid, grp in train_df.groupby("can_id"):
            cid_str = str(cid)
            cid_stats: Dict[str, Any] = {
                "count": int(len(grp)),
                "signals": {},
            }

            # Inter-arrival times in normal traffic (ignore inter-file / inter-session jumps > 1s)
            if "timestamp" in grp.columns:
                iats = grp["timestamp"].diff().dropna()
                positive_iats = iats[(iats > 0) & (iats < 1.0)]
                if len(positive_iats) > 0:
                    cid_stats["iat_mean"] = float(positive_iats.mean())
                    cid_stats["iat_std"] = float(positive_iats.std()) if len(positive_iats) > 1 else 0.0
                    cid_stats["iat_min"] = float(positive_iats.min())
                    cid_stats["iat_max"] = float(positive_iats.max())
                else:
                    cid_stats["iat_mean"] = 0.015
                    cid_stats["iat_std"] = 0.0
            else:
                cid_stats["iat_mean"] = 0.015
                cid_stats["iat_std"] = 0.0

            for i in range(1, 5):
                s_col = f"s{i}"
                v_col = f"s{i}_valid"
                if s_col in grp.columns and v_col in grp.columns:
                    valid_vals = grp.loc[grp[v_col] == 1, s_col].to_numpy()
                    if len(valid_vals) > 0:
                        s_mean = float(np.mean(valid_vals))
                        s_std = float(np.std(valid_vals, ddof=1)) if len(valid_vals) > 1 else 1.0
                        # Avoid division by zero for constant signals
                        if s_std < 1e-6:
                            s_std = 1.0

                        cid_stats["signals"][f"s{i}"] = {
                            "valid": True,
                            "count": int(len(valid_vals)),
                            "mean": s_mean,
                            "std": s_std,
                            "min": float(np.min(valid_vals)),
                            "max": float(np.max(valid_vals)),
                        }
                    else:
                        cid_stats["signals"][f"s{i}"] = {
                            "valid": False,
                            "count": 0,
                            "mean": -1.0,
                            "std": 1.0,
                            "min": -1.0,
                            "max": -1.0,
                        }

            stats["by_id"][cid_str] = cid_stats

        self.stats = stats

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)

        return self

    def load_stats(self, stats_path: Union[str, Path] = DEFAULT_STATS_PATH) -> SynCANFeatureExtractor:
        """Load baseline statistics from a JSON file."""
        p = Path(stats_path)
        if not p.exists():
            raise FileNotFoundError(f"Statistics file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        reset_state: bool = True,
    ) -> pd.DataFrame:
        """Extract causal window features for each message in the DataFrame.

        Args:
            df: Input DataFrame containing timestamp, can_id, s1..s4, s1_valid..s4_valid.
            reset_state: If True, clears all sliding buffers before processing.

        Returns:
            New DataFrame with input columns plus engineered features.
        """
        if reset_state:
            self.reset()

        n = len(df)
        if n == 0:
            return df.copy()

        # Extract arrays for high-performance looping
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        can_ids = df["can_id"].astype(str).to_numpy()

        s_vals = [df[f"s{i}"].to_numpy(dtype=np.float32) for i in range(1, 5)]
        s_valids = [df[f"s{i}_valid"].to_numpy(dtype=np.int32) for i in range(1, 5)]

        # Pre-allocate output arrays
        dt_same_id = np.zeros(n, dtype=np.float32)
        bus_rate_50ms = np.zeros(n, dtype=np.int32)
        bus_rate_1s = np.zeros(n, dtype=np.int32)

        s_deltas = [np.zeros(n, dtype=np.float32) for _ in range(4)]
        s_means = [np.full(n, -1.0, dtype=np.float32) for _ in range(4)]
        s_stds = [np.zeros(n, dtype=np.float32) for _ in range(4)]
        s_zscores = [np.zeros(n, dtype=np.float32) for _ in range(4)]
        s_repeats = [np.zeros(n, dtype=np.int32) for _ in range(4)]
        plateau_max_repeat = np.zeros(n, dtype=np.int32)

        # Baseline stats lookup cache
        by_id_stats = self.stats.get("by_id", {})

        # Sliding window two-pointer indexes for bus-wide rate calculation
        left_50ms = 0
        left_1s = 0

        # Primary causal sequential pass
        for i in range(n):
            t = timestamps[i]
            cid = can_ids[i]

            # Detect backward time jumps across files
            if self._last_global_ts is not None and t < self._last_global_ts:
                self.reset()
                left_50ms = i
                left_1s = i
            self._last_global_ts = t

            # 1. Bus message rate in trailing 50 ms and 1 s (two-pointer O(N))
            t_50ms_cut = t - 0.050
            while left_50ms < i and timestamps[left_50ms] < t_50ms_cut:
                left_50ms += 1
            bus_rate_50ms[i] = i - left_50ms + 1

            t_1s_cut = t - 1.000
            while left_1s < i and timestamps[left_1s] < t_1s_cut:
                left_1s += 1
            bus_rate_1s[i] = i - left_1s + 1

            # 2. Inter-arrival time for same CAN ID
            if cid in self._last_ts_per_id:
                dt_same_id[i] = float(t - self._last_ts_per_id[cid])
            else:
                # Default expected period from normal baseline or 0.015 s
                expected_iat = by_id_stats.get(cid, {}).get("iat_mean", 0.015)
                dt_same_id[i] = float(expected_iat)
            self._last_ts_per_id[cid] = t

            # Initialize per-ID state if first time seeing this ID
            if cid not in self._signal_window_per_id:
                self._signal_window_per_id[cid] = [deque(maxlen=10) for _ in range(4)]
                self._last_signals_per_id[cid] = [-1.0, -1.0, -1.0, -1.0]
                self._repeat_count_per_id[cid] = [0, 0, 0, 0]

            curr_windows = self._signal_window_per_id[cid]
            prev_signals = self._last_signals_per_id[cid]
            repeat_counts = self._repeat_count_per_id[cid]
            cid_sig_stats = by_id_stats.get(cid, {}).get("signals", {})

            max_rep_this_msg = 0

            # 3. Process signals 1 through 4
            for s_idx in range(4):
                val = s_vals[s_idx][i]
                is_valid = s_valids[s_idx][i] == 1

                if is_valid:
                    # Rolling window of last 10
                    dq = curr_windows[s_idx]
                    dq.append(val)
                    w_len = len(dq)
                    w_mean = sum(dq) / w_len
                    s_means[s_idx][i] = w_mean
                    if w_len > 1:
                        var = sum((x - w_mean) ** 2 for x in dq) / (w_len - 1)
                        s_stds[s_idx][i] = math.sqrt(var)
                    else:
                        s_stds[s_idx][i] = 0.0

                    # Delta from previous message of this ID
                    if len(dq) > 1:
                        s_deltas[s_idx][i] = val - prev_signals[s_idx]
                    else:
                        s_deltas[s_idx][i] = 0.0

                    # Plateau repeated-value detection
                    if len(dq) > 1 and abs(val - prev_signals[s_idx]) < 1e-7:
                        repeat_counts[s_idx] += 1
                    else:
                        repeat_counts[s_idx] = 1

                    s_repeats[s_idx][i] = repeat_counts[s_idx]
                    if repeat_counts[s_idx] > max_rep_this_msg:
                        max_rep_this_msg = repeat_counts[s_idx]

                    prev_signals[s_idx] = val

                    # Z-score against normal training baseline stats
                    stat_info = cid_sig_stats.get(f"s{s_idx + 1}")
                    if stat_info and stat_info.get("valid", False):
                        mean_stat = stat_info["mean"]
                        std_stat = stat_info["std"]
                        s_zscores[s_idx][i] = (val - mean_stat) / std_stat
                    else:
                        s_zscores[s_idx][i] = 0.0
                else:
                    # Inactive signal for this ID
                    s_deltas[s_idx][i] = 0.0
                    s_means[s_idx][i] = -1.0
                    s_stds[s_idx][i] = 0.0
                    s_zscores[s_idx][i] = 0.0
                    s_repeats[s_idx][i] = 0

            plateau_max_repeat[i] = max_rep_this_msg

        # Assemble result DataFrame preserving original columns
        out = df.copy()
        out["dt_same_id"] = dt_same_id
        out["bus_rate_50ms"] = bus_rate_50ms
        out["bus_rate_1s"] = bus_rate_1s

        for s_idx in range(4):
            num = s_idx + 1
            out[f"s{num}_delta"] = s_deltas[s_idx]
            out[f"s{num}_mean_10"] = s_means[s_idx]
            out[f"s{num}_std_10"] = s_stds[s_idx]
            out[f"s{num}_zscore"] = s_zscores[s_idx]
            out[f"s{num}_repeat_count"] = s_repeats[s_idx]

        out["plateau_max_repeat"] = plateau_max_repeat

        return out
