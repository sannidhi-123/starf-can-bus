from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gc
import json
import os
import numpy as np
import pandas as pd
import torch


# ══════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER — Frame-Level Feature Extraction
# ══════════════════════════════════════════════════════════════════════════
def build_frame_features(
    df: pd.DataFrame,
    window_n: int = 50,
    can_id_col: str = "can_id",
    timestamp_col: str = "timestamp",
    dlc_col: str = "dlc",
    byte_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build engineered frame-level features from sequential CAN bus messages.

    Exact Feature Definitions:
    --------------------------
    1. id_repetition_count (int):
       - Definition: The number of times the current frame's CAN ID appears within
         a sliding FIFO window of the last N frames (including the current frame
         and up to N - 1 immediately preceding frames in temporal sequence):
             repetition_count = sum(1 for frame in window if frame.can_id == current.can_id)
       - Window size parameter: Specified by `window_n` (default: N = 50).
       - Boundary behavior: For the initial k < N frames in the stream, the window
         contains all k frames observed so far.
       - Value range: [1, N]. Higher values indicate heavy burst/flooding activity
         characteristic of DoS and spoofing attacks.

    2. inter_arrival_time_delta (float):
       - Definition: The elapsed time in seconds between the arrival of the current
         frame and the arrival of the most recent preceding frame with the exact same
         CAN ID:
             delta = timestamp_current - timestamp_previous_for_same_id
       - Boundary behavior: If a CAN ID is observed for the first time in the stream,
         delta is assigned 0.0.
       - Time reset: If a backward time jump is detected (e.g. crossing recording
         sessions), previous timestamp history is reset and delta is set to 0.0.
       - Value range: [0.0, +inf). Unusually small values indicate abnormally high
         transmission rates for periodic or injected IDs.

    3. dlc (int):
       - Definition: Data Length Code, representing the number of payload data bytes
         transmitted in the CAN arbitration frame.
       - Value range: [0, 8].

    4. d0, d1, d2, d3, d4, d5, d6, d7 (int):
       - Definition: The 8 individual CAN payload data bytes, each represented as
         an integer in [0, 255].
       - If input byte values are hexadecimal strings (e.g. '05', 'fe', '00'),
         they are parsed as base-16 integers: int(byte_str, 16).
       - If DLC < 8 or bytes are empty/missing/NaN, they are zero-padded to 0.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing CAN bus frames in temporal order.
    window_n : int, default=50
        Sliding window size N (number of frames) for `id_repetition_count`.
    can_id_col : str, default='can_id'
        Column name for the CAN identifier (supports hex strings or integers).
    timestamp_col : str, default='timestamp'
        Column name for arrival timestamp (in seconds, float).
    dlc_col : str, default='dlc'
        Column name for Data Length Code.
    byte_cols : Sequence[str], optional
        List of 8 column names for data bytes. Defaults to ['d0', 'd1', ..., 'd7'].

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the engineered feature columns:
        ['id_repetition_count', 'inter_arrival_time_delta', 'dlc',
         'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']
        along with any target metadata columns ('label', 'source_file') if present in `df`.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id_repetition_count",
                "inter_arrival_time_delta",
                "dlc",
                "d0",
                "d1",
                "d2",
                "d3",
                "d4",
                "d5",
                "d6",
                "d7",
            ]
        )

    if byte_cols is None:
        byte_cols = [f"d{i}" for i in range(8)]

    # Validate required columns
    required_cols = [can_id_col, timestamp_col, dlc_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame missing required columns: {missing}")

    # Helper to parse CAN ID to int
    def _parse_id(val: Any) -> int:
        if isinstance(val, (int, np.integer)):
            return int(val)
        val_str = str(val).strip().lower()
        if val_str.startswith("0x"):
            val_str = val_str[2:]
        return int(val_str, 16) if val_str else 0

    # Helper to parse byte value to int (0-255)
    def _parse_byte(val: Any) -> int:
        if pd.isna(val) or val == "" or val is None:
            return 0
        if isinstance(val, (int, np.integer)):
            return int(val) & 0xFF
        val_str = str(val).strip().lower()
        if not val_str:
            return 0
        try:
            return int(val_str, 16) if len(val_str) <= 2 else int(val_str)
        except ValueError:
            return 0

    n_rows = len(df)
    rep_counts = np.zeros(n_rows, dtype=np.int32)
    iat_deltas = np.zeros(n_rows, dtype=np.float64)

    window: deque = deque()
    counts: Counter = Counter()
    last_timestamps: Dict[int, float] = {}
    last_t: Optional[float] = None

    can_ids = df[can_id_col].values
    timestamps = df[timestamp_col].astype(float).values

    for idx in range(n_rows):
        cid = _parse_id(can_ids[idx])
        t = timestamps[idx]

        # Reset state on backward time jumps (e.g. crossing recording files)
        if last_t is not None and t < last_t:
            window.clear()
            counts.clear()
            last_timestamps.clear()
        last_t = t

        # 1. Sliding window repetition count (last N frames)
        if len(window) == window_n:
            old_cid = window.popleft()
            counts[old_cid] -= 1
        window.append(cid)
        counts[cid] += 1
        rep_counts[idx] = counts[cid]

        # 2. Inter-arrival time delta (per CAN ID)
        if cid in last_timestamps:
            iat_deltas[idx] = t - last_timestamps[cid]
        else:
            iat_deltas[idx] = 0.0
        last_timestamps[cid] = t

    # 3. DLC
    dlc_vals = df[dlc_col].fillna(8).astype(int).values

    # 4. Data bytes d0..d7
    parsed_bytes: Dict[str, np.ndarray] = {}
    for i, col in enumerate(byte_cols):
        if col in df.columns:
            parsed_bytes[f"d{i}"] = np.array([_parse_byte(v) for v in df[col]], dtype=np.int16)
        else:
            parsed_bytes[f"d{i}"] = np.zeros(n_rows, dtype=np.int16)

    # Build output feature DataFrame
    result_dict = {
        "id_repetition_count": rep_counts,
        "inter_arrival_time_delta": iat_deltas,
        "dlc": dlc_vals,
    }
    for i in range(8):
        result_dict[f"d{i}"] = parsed_bytes[f"d{i}"]

    # Preserve metadata columns if present
    for meta_col in ["timestamp", "can_id", "label", "source_file"]:
        if meta_col in df.columns and meta_col not in result_dict:
            result_dict[meta_col] = df[meta_col].values

    return pd.DataFrame(result_dict, index=df.index)


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
NORMAL_CSV      = DATA_DIR / "normal_traffic_augmented.csv"
VOCAB_PATH      = DATA_DIR / "vocab.json"
SEQUENCES_PATH  = DATA_DIR / "sequences.pt"

# ── Config ─────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 64       # each sequence = 64 CAN messages
STEP_SIZE       = 32       # sliding step (50% overlap for more sequences)
CHUNK_SIZE      = 500_000  # rows to read at a time
MAX_SEQUENCES   = 500_000  # cap to avoid RAM issues

# ── Special Tokens ─────────────────────────────────────────────────────────
PAD_TOKEN   = "[PAD]"    # padding short sequences
MASK_TOKEN  = "[MASK]"   # used in training (Phase 4)
UNK_TOKEN   = "[UNK]"    # unknown CAN IDs

# ══════════════════════════════════════════════════════════════════════════
# STEP 3.1 — BUILD VOCABULARY
# ══════════════════════════════════════════════════════════════════════════
def build_vocabulary():
    print("\n" + "="*50)
    print("STEP 3.1 — Building CAN ID Vocabulary")
    print("="*50)

    unique_ids = set()
    chunk_num = 0

    print("📖 Scanning all CAN IDs from dataset...")

    for chunk in pd.read_csv(NORMAL_CSV, chunksize=CHUNK_SIZE):
        # Clean the ID column
        chunk['ID'] = chunk['ID'].astype(str).str.strip().str.lower()
        unique_ids.update(chunk['ID'].unique())
        chunk_num += 1

        if chunk_num % 10 == 0:
            print(f"  Scanned {chunk_num * CHUNK_SIZE:,} rows — "
                  f"{len(unique_ids)} unique IDs found so far...")

        del chunk
        gc.collect()

    print(f"\n✅ Scan complete!")
    print(f"📊 Total unique CAN IDs found: {len(unique_ids)}")

    # Build vocab — reserve 0,1,2 for special tokens
    vocab = {
        PAD_TOKEN:  0,
        MASK_TOKEN: 1,
        UNK_TOKEN:  2,
    }

    # Assign integer to every unique CAN ID starting from 3
    for idx, can_id in enumerate(sorted(unique_ids), start=3):
        vocab[can_id] = idx

    print(f"📋 Vocabulary size: {len(vocab)} entries")
    print(f"   → [PAD]  = 0")
    print(f"   → [MASK] = 1")
    print(f"   → [UNK]  = 2")
    print(f"   → CAN IDs = 3 to {len(vocab)-1}")

    # Save vocab to JSON
    with open(VOCAB_PATH, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"\n💾 Saved vocab.json → {VOCAB_PATH}")
    return vocab


# ══════════════════════════════════════════════════════════════════════════
# STEP 3.2 — CREATE SLIDING WINDOW SEQUENCES
# ══════════════════════════════════════════════════════════════════════════
def build_sequences(vocab):
    print("\n" + "="*50)
    print("STEP 3.2 — Creating Sliding Window Sequences")
    print("="*50)
    print(f"⚙️  Window size : {WINDOW_SIZE}")
    print(f"⚙️  Step size   : {STEP_SIZE}")
    print(f"⚙️  Max seqs    : {MAX_SEQUENCES:,}")

    all_sequences = []
    token_buffer  = []   # holds tokenized IDs across chunks
    total_seqs    = 0
    chunk_num     = 0

    for chunk in pd.read_csv(NORMAL_CSV, chunksize=CHUNK_SIZE):
        chunk_num += 1

        # Tokenize: convert each CAN ID hex → integer
        chunk['ID'] = chunk['ID'].astype(str).str.strip().str.lower()
        tokens = chunk['ID'].map(
            lambda x: vocab.get(x, vocab[UNK_TOKEN])
        ).tolist()

        # Add to buffer
        token_buffer.extend(tokens)

        # Slide window over buffer
        while len(token_buffer) >= WINDOW_SIZE:
            window = token_buffer[:WINDOW_SIZE]
            all_sequences.append(window)
            total_seqs += 1

            # Slide forward by STEP_SIZE
            token_buffer = token_buffer[STEP_SIZE:]

            # Stop if we have enough sequences
            if total_seqs >= MAX_SEQUENCES:
                break

        if chunk_num % 10 == 0:
            print(f"  Chunks processed: {chunk_num} — "
                  f"Sequences so far: {total_seqs:,}")

        del chunk, tokens
        gc.collect()

        if total_seqs >= MAX_SEQUENCES:
            print(f"\n⚠️  Reached max sequence cap: {MAX_SEQUENCES:,}")
            break

    print(f"\n✅ Sequence creation complete!")
    print(f"📊 Total sequences created : {total_seqs:,}")
    print(f"📐 Each sequence length    : {WINDOW_SIZE} tokens")

    # Convert to PyTorch tensor and save
    print("\n💾 Converting to PyTorch tensor...")
    tensor = torch.tensor(all_sequences, dtype=torch.long)
    print(f"📐 Tensor shape: {tensor.shape}")
    # Shape should be (total_seqs, WINDOW_SIZE) e.g. (500000, 64)

    torch.save(tensor, SEQUENCES_PATH)
    print(f"💾 Saved sequences.pt → {SEQUENCES_PATH}")

    return tensor


# ══════════════════════════════════════════════════════════════════════════
# MAIN — Run Both Steps
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🚀 Starting Phase 3: Tokenization & Sequence Engineering")

    # Step 3.1
    vocab = build_vocabulary()

    # Step 3.2
    tensor = build_sequences(vocab)

    print("\n" + "="*50)
    print("🎉 Phase 3 Complete!")
    print("="*50)
    print(f"✅ vocab.json     → {VOCAB_PATH}")
    print(f"✅ sequences.pt   → {SEQUENCES_PATH}")
    print(f"📐 Tensor shape   → {tensor.shape}")
    print(f"📋 Vocab size     → {len(json.load(open(VOCAB_PATH)))}")
    print("\n➡️  Ready for Phase 4: MLM Dataset Preparation")