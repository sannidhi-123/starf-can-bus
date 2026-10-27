"""Rule-based Intrusion Detection Engine for CAN Bus.

This module provides the RuleEngine class that processes CAN bus frames
in time order and evaluates deterministic heuristic/statistical rules:
  1. Invalid ID: Frame CAN ID not present in normal traffic whitelist.
  2. Frequency: Messages per CAN ID in a sliding window exceeding learned thresholds.
  3. Timing: Inter-arrival time (IAT) per CAN ID below learned minimum thresholds.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models" / "rules_config.json"


class RuleEngine:
    """Deterministic Rule-based Intrusion Detection Engine for CAN bus traffic.

    Attributes:
        config (dict): Full dictionary configuration containing rules, thresholds,
                       and whitelists.
        whitelist (set): Set of integer CAN IDs seen in normal traffic.
        window_sec (float): Sliding window duration in seconds for frequency checks.
        freq_thresholds (dict): Maximum allowed frame count per window for each CAN ID.
        default_freq_threshold (int): Fallback threshold for unknown / unmodeled IDs.
        timing_thresholds (dict): Minimum allowed inter-arrival time (seconds) per CAN ID.
        default_min_iat (float): Fallback minimum IAT (seconds).
        enabled_rules (set): Set of rule identifiers to evaluate.
    """

    RULE_INVALID_ID = "INVALID_ID"
    RULE_FREQUENCY = "FREQUENCY"
    RULE_TIMING = "TIMING"

    ALL_RULES = (RULE_INVALID_ID, RULE_FREQUENCY, RULE_TIMING)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RuleEngine with an optional configuration dict.

        If config is None, an uninitialized engine is created (call fit() or load_config()).
        """
        self.config: Dict[str, Any] = {}
        self.whitelist: Set[int] = set()
        self.enabled_rules: Set[str] = set(self.ALL_RULES)

        # Frequency configuration
        self.window_sec: float = 0.5
        self.freq_thresholds: Dict[int, int] = {}
        self.default_freq_threshold: int = 100

        # Timing configuration
        self.timing_thresholds: Dict[int, float] = {}
        self.default_min_iat: float = 0.001

        # Runtime state for streaming evaluation
        self._window_history: Dict[int, deque] = {}
        self._last_timestamp: Dict[int, float] = {}
        self._last_seen_time: Optional[float] = None

        if config is not None:
            self._apply_config(config)

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply a configuration dictionary to the engine state."""
        self.config = config

        # Whitelist
        raw_whitelist = config.get("whitelist", [])
        self.whitelist = {int(x) for x in raw_whitelist}

        # Enabled rules
        raw_rules = config.get("enabled_rules", self.ALL_RULES)
        self.enabled_rules = set(raw_rules)

        # Frequency settings
        freq_cfg = config.get("frequency", {})
        self.window_sec = float(freq_cfg.get("window_sec", 0.5))
        self.default_freq_threshold = int(freq_cfg.get("default_threshold", 100))
        raw_freq_thresh = freq_cfg.get("thresholds", {})
        self.freq_thresholds = {int(k): int(v) for k, v in raw_freq_thresh.items()}

        # Timing settings
        timing_cfg = config.get("timing", {})
        self.default_min_iat = float(timing_cfg.get("default_min_iat", 0.001))
        raw_timing_thresh = timing_cfg.get("thresholds", {})
        self.timing_thresholds = {int(k): float(v) for k, v in raw_timing_thresh.items()}

        self.reset()

    def reset(self) -> None:
        """Reset internal streaming states (sliding windows and last seen timestamps)."""
        self._window_history.clear()
        self._last_timestamp.clear()
        self._last_seen_time = None

    def fit(
        self,
        normal_df: pd.DataFrame,
        save_path: Optional[Union[Path, str]] = DEFAULT_CONFIG_PATH,
        window_sec: float = 0.5,
        freq_percentile: float = 99.9,
        freq_margin: float = 1.25,
        timing_percentile: float = 0.1,
        timing_margin: float = 0.70,
    ) -> RuleEngine:
        """Learn rule parameters and thresholds from normal training traffic only.

        Args:
            normal_df: DataFrame containing at least 'can_id' and 'timestamp' columns.
            save_path: Optional file path to save the resulting rules_config.json.
            window_sec: Sliding window size (seconds) for frequency tracking.
            freq_percentile: Percentile of normal window counts to use as base threshold.
            freq_margin: Multiplier applied to frequency percentile (e.g. 1.25x).
            timing_percentile: Lower percentile of normal IATs to base minimum threshold on.
            timing_margin: Safety factor multiplied by timing percentile (e.g. 0.70x).

        Returns:
            self with updated configuration and thresholds.
        """
        if normal_df.empty:
            raise ValueError("normal_df cannot be empty for fitting RuleEngine")

        # 1. Whitelist from normal CAN IDs
        unique_ids = sorted(int(x) for x in normal_df["can_id"].unique())
        whitelist_set = set(unique_ids)

        # 2. Timing (Inter-Arrival Time per ID)
        iats_per_id: Dict[int, List[float]] = {}
        # Group by source_file if available to avoid computing IAT across unrelated recording sessions
        group_col = "source_file" if "source_file" in normal_df.columns else None

        if group_col:
            groups = [g for _, g in normal_df.groupby(group_col, sort=False)]
        else:
            groups = [normal_df]

        for grp in groups:
            for cid, cgrp in grp.groupby("can_id"):
                diffs = cgrp["timestamp"].diff().dropna()
                diffs = diffs[diffs > 0]
                if len(diffs) > 0:
                    cid_int = int(cid)
                    if cid_int not in iats_per_id:
                        iats_per_id[cid_int] = []
                    iats_per_id[cid_int].extend(diffs.tolist())

        timing_thresholds: Dict[int, float] = {}
        for cid, diff_list in iats_per_id.items():
            if len(diff_list) >= 5:
                val = float(np.percentile(diff_list, timing_percentile))
                timing_thresholds[cid] = max(float(val * timing_margin), 0.0005)
            elif diff_list:
                timing_thresholds[cid] = max(float(min(diff_list) * timing_margin), 0.0005)

        default_min_iat = min(timing_thresholds.values()) if timing_thresholds else 0.001

        # 3. Frequency (sliding window message counts per ID)
        freq_per_id: Dict[int, List[int]] = {}
        for grp in groups:
            windows: Dict[int, deque] = {}
            for row in grp.itertuples():
                cid = int(row.can_id)
                t = float(row.timestamp)
                if cid not in windows:
                    windows[cid] = deque()
                    if cid not in freq_per_id:
                        freq_per_id[cid] = []
                q = windows[cid]
                q.append(t)
                cutoff = t - window_sec
                while q and q[0] < cutoff:
                    q.popleft()
                freq_per_id[cid].append(len(q))

        freq_thresholds: Dict[int, int] = {}
        for cid, counts in freq_per_id.items():
            if counts:
                p_val = float(np.percentile(counts, freq_percentile))
                freq_thresholds[cid] = int(np.ceil(p_val * freq_margin))

        default_max_freq = max(freq_thresholds.values()) if freq_thresholds else 100

        config = {
            "enabled_rules": list(self.ALL_RULES),
            "whitelist": unique_ids,
            "frequency": {
                "window_sec": window_sec,
                "percentile": freq_percentile,
                "margin": freq_margin,
                "default_threshold": default_max_freq,
                "thresholds": {str(k): v for k, v in freq_thresholds.items()},
            },
            "timing": {
                "percentile": timing_percentile,
                "margin": timing_margin,
                "default_min_iat": default_min_iat,
                "thresholds": {str(k): v for k, v in timing_thresholds.items()},
            },
        }

        self._apply_config(config)

        if save_path:
            self.save_config(save_path)

        return self

    def save_config(self, save_path: Union[Path, str]) -> None:
        """Save engine configuration to JSON."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def load_config(self, config_path: Union[Path, str]) -> RuleEngine:
        """Load engine configuration from JSON."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Rules config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self._apply_config(config)
        return self

    def process_frame(
        self,
        timestamp: float,
        can_id: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Process a single CAN bus frame in time order.

        Args:
            timestamp: Frame arrival timestamp in seconds (float).
            can_id: CAN arbitration ID (integer).
            **kwargs: Extra attributes (ignored or available for custom extensions).

        Returns:
            List of triggered rule dictionaries. If no rules are triggered, returns [].
            Each dict has:
              - 'rule_id': str
              - 'reason': str
              - 'evidence': dict
        """
        ts = float(timestamp)
        cid = int(can_id)
        triggered_rules: List[Dict[str, Any]] = []

        # Detect backward time jumps (e.g. crossing recording files/sessions) and auto-reset state
        if self._last_seen_time is not None and ts < self._last_seen_time:
            self.reset()
        self._last_seen_time = ts

        # -------------------------------------------------------------
        # Rule 1: Invalid ID
        # -------------------------------------------------------------
        if self.RULE_INVALID_ID in self.enabled_rules:
            if cid not in self.whitelist:
                triggered_rules.append(
                    {
                        "rule_id": self.RULE_INVALID_ID,
                        "reason": f"CAN ID 0x{cid:04x} ({cid}) is not in the whitelist of known normal CAN IDs.",
                        "evidence": {
                            "can_id": cid,
                            "can_id_hex": f"0x{cid:04x}",
                            "whitelist_size": len(self.whitelist),
                        },
                    }
                )

        # -------------------------------------------------------------
        # Rule 2: Frequency (Sliding Window message count)
        # -------------------------------------------------------------
        if self.RULE_FREQUENCY in self.enabled_rules:
            if cid not in self._window_history:
                self._window_history[cid] = deque()
            q = self._window_history[cid]
            q.append(ts)
            cutoff = ts - self.window_sec
            while q and q[0] < cutoff:
                q.popleft()

            count = len(q)
            limit = self.freq_thresholds.get(cid, self.default_freq_threshold)
            if count > limit:
                triggered_rules.append(
                    {
                        "rule_id": self.RULE_FREQUENCY,
                        "reason": (
                            f"Frequency for CAN ID 0x{cid:04x} exceeded threshold: "
                            f"{count} messages in {self.window_sec}s window (limit: {limit})."
                        ),
                        "evidence": {
                            "can_id": cid,
                            "can_id_hex": f"0x{cid:04x}",
                            "count": count,
                            "threshold": limit,
                            "window_sec": self.window_sec,
                        },
                    }
                )

        # -------------------------------------------------------------
        # Rule 3: Timing (Inter-arrival time below minimum threshold)
        # -------------------------------------------------------------
        if self.RULE_TIMING in self.enabled_rules:
            if cid in self._last_timestamp:
                iat = ts - self._last_timestamp[cid]
                min_iat = self.timing_thresholds.get(cid, self.default_min_iat)
                if 0 <= iat < min_iat:
                    triggered_rules.append(
                        {
                            "rule_id": self.RULE_TIMING,
                            "reason": (
                                f"Inter-arrival time for CAN ID 0x{cid:04x} ({iat:.6f}s) "
                                f"is below learned minimum threshold ({min_iat:.6f}s)."
                            ),
                            "evidence": {
                                "can_id": cid,
                                "can_id_hex": f"0x{cid:04x}",
                                "iat": float(iat),
                                "min_iat_threshold": float(min_iat),
                            },
                        }
                    )
            self._last_timestamp[cid] = ts

        return triggered_rules

    def process_dataframe(
        self,
        df: pd.DataFrame,
        reset_state: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Process an entire DataFrame of frames in time order.

        Args:
            df: DataFrame containing at least 'timestamp' and 'can_id'.
            reset_state: Whether to reset streaming state before processing.

        Returns:
            List where element i is the list of triggered rules for row i.
        """
        if reset_state:
            self.reset()

        results: List[List[Dict[str, Any]]] = []
        for row in df.itertuples():
            res = self.process_frame(timestamp=row.timestamp, can_id=row.can_id)
            results.append(res)
        return results

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate rules against a labeled DataFrame and calculate metrics.

        Expects column 'label' where 'Normal' is negative and any other label is positive/attack.

        Returns:
            Dictionary containing confusion matrix counts, precision, recall, FPR,
            and breakdown of triggers per rule.
        """
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain 'label' column for evaluation")

        self.reset()

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        rule_counts: Dict[str, int] = {r: 0 for r in self.ALL_RULES}

        # Process per source_file if available to preserve streaming continuity
        groups = (
            [g for _, g in df.groupby("source_file", sort=False)]
            if "source_file" in df.columns
            else [df]
        )

        for grp in groups:
            self.reset()
            for row in grp.itertuples():
                is_attack = row.label != "Normal"
                triggered = self.process_frame(timestamp=row.timestamp, can_id=row.can_id)
                has_anomaly = len(triggered) > 0

                for t in triggered:
                    rule_id = t["rule_id"]
                    rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

                if is_attack and has_anomaly:
                    tp += 1
                elif is_attack and not has_anomaly:
                    fn += 1
                elif not is_attack and has_anomaly:
                    fp += 1
                else:
                    tn += 1

        total = tp + fp + tn + fn
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        accuracy = float((tp + tn) / total) if total else 0.0

        return {
            "total_frames": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "accuracy": accuracy,
            "rule_counts": rule_counts,
        }


def main():
    """CLI to fit and evaluate RuleEngine on data/prepared.parquet."""
    import argparse

    parser = argparse.ArgumentParser(description="Fit and evaluate RuleEngine on CAN Bus data.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "prepared.parquet",
        help="Path to prepared dataset parquet file.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Output path for learned rules_config.json.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data used for training (default: 0.8).",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset {args.dataset} not found. Run src/dataset_prepare.py first.")
        return

    print("Loading prepared dataset...")
    df = pd.read_parquet(args.dataset)

    # Contiguous split per source file
    train_list, test_list = [], []
    for _, group in df.groupby("source_file", sort=False):
        n = len(group)
        n_train = int(n * args.train_ratio)
        train_list.append(group.iloc[:n_train])
        test_list.append(group.iloc[n_train:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    normal_train = train_df[train_df["label"] == "Normal"]
    print(f"Normal training split: {len(normal_train):,} rows")
    print(f"Held-out test split   : {len(test_df):,} rows")

    # Fit RuleEngine on Normal training split only
    print("Fitting RuleEngine on Normal training split...")
    engine = RuleEngine()
    engine.fit(normal_train, save_path=args.config_out)
    print(f"Saved learned rules config to: {args.config_out}")

    # Evaluate on held-out split
    print("\nEvaluating RuleEngine on held-out test split...")
    metrics = engine.evaluate(test_df)

    print("\n" + "=" * 60)
    print("  RULE ENGINE HELD-OUT EVALUATION METRICS")
    print("=" * 60)
    print(f"  Total Held-out Frames : {metrics['total_frames']:,}")
    print(f"  True Positives  (TP)  : {metrics['tp']:,}")
    print(f"  False Positives (FP)  : {metrics['fp']:,}")
    print(f"  True Negatives  (TN)  : {metrics['tn']:,}")
    print(f"  False Negatives (FN)  : {metrics['fn']:,}")
    print("  --------------------------------------------------")
    print(f"  Precision             : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall                : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  False Positive Rate   : {metrics['fpr']:.4f} ({metrics['fpr']*100:.4f}%)")
    print(f"  Accuracy              : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("  --------------------------------------------------")
    print("  Rule Trigger Breakdown:")
    for rule_id, count in metrics["rule_counts"].items():
        print(f"    - {rule_id:<20}: {count:,} triggers")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
