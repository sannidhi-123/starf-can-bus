"""Prepares SynCAN dataset into canonical Parquet files with causal window features.

Builds:
1. datasets/syncan/processed/train_normal.parquet (500,000 rows from train_1 and train_2)
2. datasets/syncan/processed/test_<scenario>.parquet (for each test file)
Prints row and class counts for each file.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.syncan_dataset import load_syncan
from src.syncan_features import SynCANFeatureExtractor


def prepare_syncan(
    train_rows: int = 500_000,
    max_test_rows: int | None = None,
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Prepare and save train and test Parquet files with causal features."""
    base_raw = raw_dir or (PROJECT_ROOT / "datasets" / "syncan" / "raw")
    base_out = output_dir or (PROJECT_ROOT / "datasets" / "syncan" / "processed")
    base_out.mkdir(parents=True, exist_ok=True)

    fe = SynCANFeatureExtractor()
    stats_path = PROJECT_ROOT / "models" / "syncan_stats.json"

    summary: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # 1. Prepare Normal Training Parquet
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("1. Preparing datasets/syncan/processed/train_normal.parquet")
    print("=" * 70)

    train_1_path = base_raw / "train" / "train_1.csv"
    train_2_path = base_raw / "train" / "train_2.csv"

    rows_per_file = train_rows // 2
    t0 = time.time()
    df_t1 = load_syncan(train_1_path, max_rows=rows_per_file)
    df_t2 = load_syncan(train_2_path, max_rows=rows_per_file)

    train_raw_df = pd.concat([df_t1, df_t2], ignore_index=True)
    print(f"Loaded {len(train_raw_df):,} raw training rows ({len(df_t1):,} from train_1, {len(df_t2):,} from train_2).")

    # Fit feature extractor baseline stats if not already saved
    if not stats_path.exists():
        print(f"Fitting baseline statistics to {stats_path}...")
        fe.fit(train_raw_df, save_path=stats_path)
    else:
        fe.load_stats(stats_path)

    # Transform features (reset state per training segment)
    train_feat_1 = fe.transform(df_t1, reset_state=True)
    train_feat_2 = fe.transform(df_t2, reset_state=True)
    train_feat_df = pd.concat([train_feat_1, train_feat_2], ignore_index=True)

    out_train_path = base_out / "train_normal.parquet"
    train_feat_df.to_parquet(out_train_path, index=False)
    elapsed_train = time.time() - t0

    train_label_counts = train_feat_df["label"].value_counts().to_dict()
    train_attack_counts = train_feat_df["attack_type"].value_counts().to_dict()

    summary["train_normal.parquet"] = {
        "rows": len(train_feat_df),
        "label_counts": train_label_counts,
        "attack_type_counts": train_attack_counts,
        "output_path": str(out_train_path),
        "file_size_mb": round(out_train_path.stat().st_size / (1024 * 1024), 2),
    }

    print(f"Saved: {out_train_path} ({summary['train_normal.parquet']['file_size_mb']} MB in {elapsed_train:.1f}s)")
    print(f"  Rows: {len(train_feat_df):,}")
    print(f"  Label distribution: {train_label_counts}")
    print(f"  Attack types: {train_attack_counts}")

    # -------------------------------------------------------------------------
    # 2. Prepare Test Parquets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Preparing Test Parquets (causal window features, state reset per file)")
    print("=" * 70)

    test_files = sorted((base_raw / "test").glob("test_*.csv"))
    if not test_files:
        print(f"Warning: No test files found under {base_raw / 'test'}")

    for test_csv in test_files:
        t_start = time.time()
        file_stem = test_csv.stem
        out_name = f"{file_stem}.parquet"
        out_path = base_out / out_name

        print(f"\nProcessing {test_csv.name}...")
        test_df = load_syncan(test_csv, max_rows=max_test_rows)
        test_feat = fe.transform(test_df, reset_state=True)
        test_feat.to_parquet(out_path, index=False)

        elapsed = time.time() - t_start
        size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)
        lbl_counts = test_feat["label"].value_counts().to_dict()
        atk_counts = test_feat["attack_type"].value_counts().to_dict()

        summary[out_name] = {
            "rows": len(test_feat),
            "label_counts": lbl_counts,
            "attack_type_counts": atk_counts,
            "output_path": str(out_path),
            "file_size_mb": size_mb,
        }

        print(f"Saved {out_name} ({size_mb} MB in {elapsed:.1f}s)")
        print(f"  Rows: {len(test_feat):,}")
        print(f"  Label distribution (0=Normal, 1=Attack): {lbl_counts}")
        print(f"  Attack type distribution: {atk_counts}")

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SynCAN Parquet Preparation Complete - Summary")
    print("=" * 70)
    print(f"{'File Name':28s} {'Rows':>12s} {'Size':>10s} {'Class Distribution'}")
    print("-" * 70)
    for fname, info in summary.items():
        dist_str = f"Normal={info['attack_type_counts'].get('normal', 0):,}"
        attacks = [f"{k}={v:,}" for k, v in info['attack_type_counts'].items() if k != 'normal']
        if attacks:
            dist_str += " | " + ", ".join(attacks)
        print(f"{fname:28s} {info['rows']:12,d} {info['file_size_mb']:>8.1f}MB  {dist_str}")
    print("=" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare SynCAN canonical Parquet files.")
    parser.add_argument(
        "--train-rows",
        type=int,
        default=500_000,
        help="Total rows to subsample for train_normal.parquet (default: 500000)",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional row limit per test file (default: None for full file)",
    )
    args = parser.parse_args()

    prepare_syncan(train_rows=args.train_rows, max_test_rows=args.max_test_rows)


if __name__ == "__main__":
    main()
