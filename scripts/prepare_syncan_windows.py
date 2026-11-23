"""Build window-level parquet files for all SynCAN processed files.

Reads each message-level parquet, calls build_windows(), and saves the
result to datasets/syncan/processed/windows_<stem>.parquet.

Memory note: processes one file at a time (never loads two full files
simultaneously). Each parquet is ~300 MB on disk; loading it consumes
roughly 600-900 MB RAM. The window result is ~5-20 MB.

Usage:
    venv\\Scripts\\python scripts\\prepare_syncan_windows.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to sys.path so src imports work when run from any dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.syncan_windows import build_windows, load_syncan_stats  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "syncan" / "processed"

# Files to process: (source parquet stem, attack_type for attack messages)
# train_normal uses all-normal labels; test files carry mixed labels
FILES_TO_PROCESS = [
    ("train_normal", "normal"),
    ("test_normal", "normal"),
    ("test_continuous", "continuous"),
    ("test_flooding", "flooding"),
    ("test_plateau", "plateau"),
    ("test_playback", "playback"),
    ("test_suppress", "suppress"),
]


def process_file(stem: str, attack_type: str, stats: dict, window_s: float = 0.5) -> None:
    src_path = PROCESSED_DIR / f"{stem}.parquet"
    dst_path = PROCESSED_DIR / f"windows_{stem}.parquet"

    if not src_path.exists():
        print(f"  [SKIP] {src_path.name} not found — skipping")
        return

    print(f"\n--- Processing {stem} ---")
    t0 = time.time()

    df = pd.read_parquet(src_path)
    print(f"  Loaded {len(df):,} messages in {time.time() - t0:.1f}s")

    t1 = time.time()
    windows_df = build_windows(df, window_s=window_s, stats=stats, file_attack_type=attack_type)
    elapsed = time.time() - t1

    total_windows = len(windows_df)
    if total_windows == 0:
        print(f"  [WARN] 0 windows produced — skipping save")
        return

    # Print stats
    label_counts = windows_df["label"].value_counts().sort_index().to_dict()
    n_attack = label_counts.get(1, 0)
    n_normal = label_counts.get(0, 0)
    attack_pct = 100.0 * n_attack / total_windows if total_windows > 0 else 0.0

    print(f"  Windows    : {total_windows:,}")
    print(f"  Normal (0) : {n_normal:,}")
    print(f"  Attack (1) : {n_attack:,}  ({attack_pct:.1f}%)")
    print(f"  Build time : {elapsed:.1f}s")

    windows_df.to_parquet(dst_path, index=False)
    print(f"  Saved      : {dst_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("=" * 60)
    print("SynCAN Window Preparation Script")
    print("=" * 60)

    # Load stats once — shared across all files
    stats = load_syncan_stats()
    print(f"Loaded normal-train statistics from models/syncan_stats.json")
    print(f"Output directory: {PROCESSED_DIR.relative_to(PROJECT_ROOT)}")

    overall_t0 = time.time()

    for stem, attack_type in FILES_TO_PROCESS:
        process_file(stem, attack_type, stats)

    total_elapsed = time.time() - overall_t0
    print(f"\n{'=' * 60}")
    print(f"All files done in {total_elapsed:.1f}s total")
    print("=" * 60)


if __name__ == "__main__":
    main()
