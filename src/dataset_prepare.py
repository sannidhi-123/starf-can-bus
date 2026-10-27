"""Dataset preparation script for STARF CAN Bus IDS.

Reads raw Car Hacking dataset files from data/car_hacking/, unifies them into
a single standardized schema, samples contiguous time blocks per class, and
exports to data/prepared.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


EXPECTED_CLASSES = ("Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM")

SCHEMA_COLUMNS = [
    "timestamp",
    "can_id",
    "dlc",
    "d0",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "label",
    "source_file",
]


def parse_attack_file(
    file_path: Path,
    attack_label: str,
    max_attack_rows: Optional[int],
    max_normal_rows: Optional[int],
) -> List[dict]:
    """Parse a CSV attack file (DoS, Fuzzy, Gear, RPM).

    Within the file:
      - Flag 'T' indicates an injected attack -> labeled as attack_label.
      - Flag 'R' indicates regular traffic -> labeled as 'Normal'.
    If the flag does not exist or is unrecognized, raises an error and stops.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required attack file not found: {file_path}")

    source_name = file_path.name
    rows: List[dict] = []
    attack_count = 0
    normal_count = 0

    attack_limit = max_attack_rows if (max_attack_rows and max_attack_rows > 0) else None
    normal_limit = max_normal_rows if (max_normal_rows and max_normal_rows > 0) else None

    # Track contiguous collection: we collect sequentially from the file
    print(f"  Reading {source_name}...")
    t0 = time.time()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split(",")
            if len(parts) < 4:
                raise ValueError(
                    f"Malformed line {line_idx} in {source_name}: expected at least 4 fields, got {len(parts)}: {line_str}"
                )

            flag = parts[-1].strip()
            if flag not in ("R", "T"):
                raise ValueError(
                    f"Flag column does not exist or has invalid value '{flag}' in {source_name} at line {line_idx}. Stopping."
                )

            # Determine label
            if flag == "T":
                if attack_limit is not None and attack_count >= attack_limit:
                    # Attack quota filled; check if we still need normal rows
                    if normal_limit is not None and normal_count >= normal_limit:
                        break
                    continue
                label = attack_label
                attack_count += 1
            else:  # flag == 'R'
                if normal_limit is not None and normal_count >= normal_limit:
                    # Normal quota filled; check if we still need attack rows
                    if attack_limit is not None and attack_count >= attack_limit:
                        break
                    continue
                label = "Normal"
                normal_count += 1

            ts = float(parts[0])
            can_id = int(parts[1], 16)
            dlc = int(parts[2])
            raw_bytes = parts[3:-1]

            # Pad or slice data bytes d0..d7
            data_bytes = (raw_bytes + [""] * 8)[:8]

            rows.append(
                {
                    "timestamp": ts,
                    "can_id": can_id,
                    "dlc": dlc,
                    "d0": data_bytes[0],
                    "d1": data_bytes[1],
                    "d2": data_bytes[2],
                    "d3": data_bytes[3],
                    "d4": data_bytes[4],
                    "d5": data_bytes[5],
                    "d6": data_bytes[6],
                    "d7": data_bytes[7],
                    "label": label,
                    "source_file": source_name,
                }
            )

            # Early exit if both limits are satisfied
            if (
                attack_limit is not None
                and attack_count >= attack_limit
                and normal_limit is not None
                and normal_count >= normal_limit
            ):
                break

    elapsed = time.time() - t0
    print(
        f"    -> Parsed {len(rows):,} rows ({attack_count:,} {attack_label}, {normal_count:,} Normal) in {elapsed:.1f}s"
    )
    return rows


def parse_normal_file(
    file_path: Path,
    max_rows: Optional[int],
) -> List[dict]:
    """Parse normal_run_data.txt.

    Format:
      Timestamp: 1479121434.850202        ID: 0350    000    DLC: 8    05 28 84 66 6d 00 00 a2
    All frames in this file are pure Normal traffic.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required normal traffic file not found: {file_path}")

    source_name = file_path.name
    rows: List[dict] = []
    limit = max_rows if (max_rows and max_rows > 0) else None

    print(f"  Reading {source_name}...")
    t0 = time.time()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            tokens = line_str.split()
            # Expect tokens like: ['Timestamp:', '<ts>', 'ID:', '<hex_id>', '000', 'DLC:', '<dlc>', '<d0>', ...]
            if len(tokens) < 7:
                # Incomplete or trailing lines are safely skipped
                continue

            try:
                ts = float(tokens[1])
                can_id = int(tokens[3], 16)
                dlc = int(tokens[6])
                raw_bytes = tokens[7:]
            except (ValueError, IndexError):
                continue

            data_bytes = (raw_bytes + [""] * 8)[:8]

            rows.append(
                {
                    "timestamp": ts,
                    "can_id": can_id,
                    "dlc": dlc,
                    "d0": data_bytes[0],
                    "d1": data_bytes[1],
                    "d2": data_bytes[2],
                    "d3": data_bytes[3],
                    "d4": data_bytes[4],
                    "d5": data_bytes[5],
                    "d6": data_bytes[6],
                    "d7": data_bytes[7],
                    "label": "Normal",
                    "source_file": source_name,
                }
            )

            if limit is not None and len(rows) >= limit:
                break

    elapsed = time.time() - t0
    print(f"    -> Parsed {len(rows):,} rows (all Normal) in {elapsed:.1f}s")
    return rows


def prepare_dataset(
    car_hacking_dir: Path,
    output_path: Path,
    max_rows_per_class: int = 200000,
) -> pd.DataFrame:
    """Read all Car Hacking files, standardize schema, sample contiguous blocks, and save."""
    print("=" * 68)
    print("  STARF Dataset Preparation Pipeline")
    print(f"  Input directory     : {car_hacking_dir}")
    print(f"  Output path         : {output_path}")
    print(f"  Max rows per class  : {max_rows_per_class:,}" if max_rows_per_class > 0 else "  Max rows per class  : Unlimited (all rows)")
    print("=" * 68)

    # Locate files
    dos_file = car_hacking_dir / "DoS_dataset.csv"
    fuzzy_file = car_hacking_dir / "Fuzzy_dataset.csv"
    gear_file = car_hacking_dir / "gear_dataset.csv"
    rpm_file = car_hacking_dir / "RPM_dataset.csv"

    # normal_run_data can be directly inside or inside a subdirectory
    normal_candidates = [
        car_hacking_dir / "normal_run_data" / "normal_run_data.txt",
        car_hacking_dir / "normal_run_data.txt",
    ]
    normal_file = next((p for p in normal_candidates if p.exists()), None)
    if not normal_file:
        raise FileNotFoundError(
            f"normal_run_data.txt not found in {car_hacking_dir} or {car_hacking_dir / 'normal_run_data'}"
        )

    # Determine per-file normal budget when max_rows_per_class is set
    # 5 files contribute to Normal: normal_run_data + 4 attack files
    if max_rows_per_class > 0:
        num_sources = 5
        base_normal_per_source = max_rows_per_class // num_sources
        remainder = max_rows_per_class % num_sources
        normal_file_quota = base_normal_per_source + remainder
        attack_file_normal_quota = base_normal_per_source
        attack_file_attack_quota = max_rows_per_class
    else:
        normal_file_quota = None
        attack_file_normal_quota = None
        attack_file_attack_quota = None

    all_rows: List[dict] = []

    # 1. Parse normal traffic run
    normal_rows = parse_normal_file(normal_file, max_rows=normal_file_quota)
    all_rows.extend(normal_rows)

    # 2. Parse attack files
    attack_configs = [
        (dos_file, "DoS"),
        (fuzzy_file, "Fuzzy"),
        (gear_file, "Spoof_Gear"),
        (rpm_file, "Spoof_RPM"),
    ]

    for fpath, label in attack_configs:
        file_rows = parse_attack_file(
            file_path=fpath,
            attack_label=label,
            max_attack_rows=attack_file_attack_quota,
            max_normal_rows=attack_file_normal_quota,
        )
        all_rows.extend(file_rows)

    # Build DataFrame
    print("\n  Constructing DataFrame...")
    df = pd.DataFrame(all_rows, columns=SCHEMA_COLUMNS)

    # Ensure correct data types
    df["timestamp"] = df["timestamp"].astype("float64")
    df["can_id"] = df["can_id"].astype("int64")
    df["dlc"] = df["dlc"].astype("int64")
    for b in ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]:
        df[b] = df[b].astype("string")
    df["label"] = df["label"].astype("string")
    df["source_file"] = df["source_file"].astype("string")

    # 4. Fail loudly if any expected class is missing or empty
    class_counts = df["label"].value_counts().to_dict()
    for exp_cls in EXPECTED_CLASSES:
        count = class_counts.get(exp_cls, 0)
        if count == 0:
            raise RuntimeError(
                f"FATAL: Class '{exp_cls}' is empty in the prepared dataset! Expected all 5 classes: {EXPECTED_CLASSES}"
            )

    # Save to parquet or fallback to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving output to {output_path}...")
    t_save = time.time()
    saved_as_parquet = False
    try:
        df.to_parquet(output_path, index=False, engine="pyarrow")
        saved_as_parquet = True
    except Exception as exc:
        csv_fallback = output_path.with_suffix(".csv")
        print(f"  [WARN] Parquet export failed ({exc}). Falling back to CSV: {csv_fallback}")
        df.to_csv(csv_fallback, index=False)
        output_path = csv_fallback

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    format_name = "Parquet" if saved_as_parquet else "CSV"
    print(f"  Output saved successfully: {file_size_mb:.2f} MB ({format_name}) in {time.time() - t_save:.1f}s")

    # 5. Print count tables
    print("\n" + "=" * 68)
    print("  PREPARED DATASET SUMMARY")
    print("=" * 68)
    print(f"  Total Rows: {len(df):,}\n")

    print("  Per-Class Distribution:")
    print("  --------------------------------------------------")
    print(f"  {'Class':<16} {'Count':>12} {'Percent':>10}")
    print("  --------------------------------------------------")
    for cls in EXPECTED_CLASSES:
        cnt = class_counts.get(cls, 0)
        pct = (cnt / len(df)) * 100 if len(df) else 0.0
        print(f"  {cls:<16} {cnt:>12,} {pct:>9.2f}%")
    print("  --------------------------------------------------")
    print(f"  {'Total':<16} {len(df):>12,} {'100.00%':>10}\n")

    print("  Per-File Distribution:")
    print("  --------------------------------------------------")
    print(f"  {'Source File':<24} {'Count':>12} {'Percent':>10}")
    print("  --------------------------------------------------")
    file_counts = df["source_file"].value_counts().to_dict()
    for s_file, cnt in file_counts.items():
        pct = (cnt / len(df)) * 100 if len(df) else 0.0
        print(f"  {s_file:<24} {cnt:>12,} {pct:>9.2f}%")
    print("  --------------------------------------------------")
    print(f"  {'Total':<24} {len(df):>12,} {'100.00%':>10}")
    print("=" * 68 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare Car Hacking dataset for STARF CAN Bus IDS.")
    parser.add_argument(
        "--car-hacking-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "car_hacking",
        help="Directory containing the raw Car Hacking files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "prepared.parquet",
        help="Path for the output parquet file.",
    )
    parser.add_argument(
        "--max-rows-per-class",
        type=int,
        default=200000,
        help="Maximum rows to sample per class using contiguous time blocks (default: 200000). Set to 0 for unlimited.",
    )

    args = parser.parse_args()
    try:
        prepare_dataset(
            car_hacking_dir=args.car_hacking_dir,
            output_path=args.output,
            max_rows_per_class=args.max_rows_per_class,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
