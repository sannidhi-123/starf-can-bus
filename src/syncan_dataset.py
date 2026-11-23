"""Dataset loader for the SynCAN (Synthetic CAN Bus) benchmark.

Provides chunked reading and unified normalization of SynCAN CSV files:
- Header detection: automatically handles files with headers (train_1, test_*)
  and files without headers (train_2, train_3, train_4) where lines have variable columns.
- Time column (ms) converted to timestamp (seconds).
- Arbitrary signal counts mapped to s1..s4 with NaN filled with -1.0.
- Signal validity masks s1_valid..s4_valid.
- Binary labels (0 = Normal, 1 = Attack).
- attack_type mapped from file naming and message labels.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import pandas as pd


# Known SynCAN signal counts per ID (from benchmark specification)
SYNCAN_SIGNAL_COUNTS = {
    "id1": 2,
    "id2": 3,
    "id3": 2,
    "id4": 1,
    "id5": 2,
    "id6": 2,
    "id7": 2,
    "id8": 1,
    "id9": 1,
    "id10": 4,
}

CANONICAL_COLUMNS = ["Label", "Time", "ID", "s1", "s2", "s3", "s4"]


def extract_attack_type_from_filename(file_path: Union[str, Path]) -> str:
    """Extract standard SynCAN attack type identifier from a file path."""
    name = Path(file_path).stem.lower()
    for at in ["flooding", "plateau", "continuous", "playback", "suppress"]:
        if at in name:
            return at
    if "normal" in name or "train" in name:
        return "normal"
    return "other"


def has_header_row(file_path: Union[str, Path]) -> bool:
    """Check whether the first line of the CSV file contains header names."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().lower()
    return "time" in first_line or "label" in first_line


def load_syncan(
    path: Union[str, Path],
    max_rows: Optional[int] = None,
    chunksize: int = 100_000,
    label_conditioned_attack_type: bool = True,
) -> pd.DataFrame:
    """Load a SynCAN CSV file using chunked streaming.

    Args:
        path: Path to the SynCAN CSV file.
        max_rows: Maximum total rows to read (reads entire file if None).
        chunksize: Number of rows per chunk for low-memory processing.
        label_conditioned_attack_type: If True, rows where label == 0 are
            labeled 'normal', and only attack intervals (label == 1) receive
            the attack type from the file name.

    Returns:
        DataFrame containing:
          - timestamp: float in seconds (Time / 1000.0)
          - can_id: str ('id1'..'id10')
          - s1, s2, s3, s4: float (NaN filled with -1.0)
          - s1_valid, s2_valid, s3_valid, s4_valid: int (1 if signal is valid for ID, 0 if NaN)
          - label: int (0 = Normal, 1 = Attack)
          - attack_type: str ('normal', 'flooding', 'plateau', etc.)
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"SynCAN file not found: {file_path}")

    base_attack_type = extract_attack_type_from_filename(file_path)
    has_header = has_header_row(file_path)

    chunks: list[pd.DataFrame] = []
    total_read = 0

    # For files without header, supply explicit column names to handle variable column count per line
    if has_header:
        reader = pd.read_csv(
            file_path,
            chunksize=chunksize,
            dtype={"ID": "string"},
            engine="c",
        )
    else:
        reader = pd.read_csv(
            file_path,
            names=CANONICAL_COLUMNS,
            header=None,
            chunksize=chunksize,
            dtype={"ID": "string"},
            engine="c",
        )

    for chunk in reader:
        if max_rows is not None and total_read >= max_rows:
            break

        if max_rows is not None and total_read + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - total_read]

        # Standardize column names
        # Raw files vary: Signal1 vs Signal1_of_ID vs s1
        col_map = {}
        for col in chunk.columns:
            clow = str(col).lower().strip()
            if clow == "time":
                col_map[col] = "Time"
            elif clow in ("id", "can_id"):
                col_map[col] = "ID"
            elif clow == "label":
                col_map[col] = "Label"
            elif clow in ("signal1", "signal1_of_id", "s1"):
                col_map[col] = "s1"
            elif clow in ("signal2", "signal2_of_id", "s2"):
                col_map[col] = "s2"
            elif clow in ("signal3", "signal3_of_id", "s3"):
                col_map[col] = "s3"
            elif clow in ("signal4", "signal4_of_id", "s4"):
                col_map[col] = "s4"

        chunk = chunk.rename(columns=col_map)

        # Validate mandatory columns
        if "Time" not in chunk.columns:
            raise ValueError(f"SynCAN file {file_path.name} missing 'Time' column")
        if "ID" not in chunk.columns:
            raise ValueError(f"SynCAN file {file_path.name} missing 'ID' column")

        # 1. Timestamp (convert ms to seconds)
        timestamp = (chunk["Time"].astype(np.float64) / 1000.0).to_numpy()

        # 2. CAN ID
        can_id = chunk["ID"].astype(str).str.strip().to_numpy()

        # 3. Label
        if "Label" in chunk.columns:
            label = chunk["Label"].fillna(0).astype(int).to_numpy()
        else:
            label = np.zeros(len(chunk), dtype=int)

        # 4. Signals and Validity Masks
        signal_dict = {}
        for i in range(1, 5):
            col_name = f"s{i}"
            if col_name in chunk.columns:
                vals = pd.to_numeric(chunk[col_name], errors="coerce")
                valid_mask = (~vals.isna()).astype(int).to_numpy()
                filled_vals = vals.fillna(-1.0).astype(np.float32).to_numpy()
            else:
                valid_mask = np.zeros(len(chunk), dtype=int)
                filled_vals = np.full(len(chunk), -1.0, dtype=np.float32)

            signal_dict[f"s{i}"] = filled_vals
            signal_dict[f"s{i}_valid"] = valid_mask

        # 5. Attack Type
        if label_conditioned_attack_type:
            # If label == 1, assign file's attack type; if 0, assign 'normal'
            attack_types = np.where(label == 1, base_attack_type, "normal")
        else:
            attack_types = np.full(len(chunk), base_attack_type, dtype=object)

        processed_chunk = pd.DataFrame(
            {
                "timestamp": timestamp,
                "can_id": can_id,
                "s1": signal_dict["s1"],
                "s2": signal_dict["s2"],
                "s3": signal_dict["s3"],
                "s4": signal_dict["s4"],
                "s1_valid": signal_dict["s1_valid"],
                "s2_valid": signal_dict["s2_valid"],
                "s3_valid": signal_dict["s3_valid"],
                "s4_valid": signal_dict["s4_valid"],
                "label": label,
                "attack_type": attack_types,
            }
        )

        chunks.append(processed_chunk)
        total_read += len(processed_chunk)

    if not chunks:
        return pd.DataFrame(
            columns=[
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
        )

    return pd.concat(chunks, ignore_index=True)
