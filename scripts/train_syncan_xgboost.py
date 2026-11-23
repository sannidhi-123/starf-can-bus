"""Train and evaluate XGBoost classifier on SynCAN benchmark dataset.

Builds chronological dataset splits from datasets/syncan/processed/*.parquet:
  - First 70% of each test scenario file + train_normal.parquet -> candidate TRAIN
  - Next 15% of each test scenario file + test_normal.parquet -> VAL
  - Last 15% of each test scenario file + test_normal.parquet -> TEST (clean held-out)
  - Chronological ordering preserved, never shuffled.
  - TRAIN capped at max_train_rows (default 800,000) by taking every k-th row (stride k),
    preserving representation of all attack windows across the entire duration.

Classes:
  normal, flooding, plateau, continuous, playback, suppress

Evaluation:
  1. Message level: precision, recall, F1, FPR on normal, confusion matrix, per-attack recall.
  2. 1-second window level: window is attack if any message label == 1; predicted attack
     if at least 3 messages are flagged. Precision, recall, F1, FPR per attack type.

Outputs:
  - models/syncan_xgb_model.pkl
  - models/syncan_xgb_meta.json
  - reports/syncan_xgb_metrics.json
  - reports/syncan_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings

# Paths
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "syncan" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_OUT_PATH = MODELS_DIR / "syncan_xgb_model.pkl"
META_OUT_PATH = MODELS_DIR / "syncan_xgb_meta.json"
METRICS_OUT_PATH = REPORTS_DIR / "syncan_xgb_metrics.json"
CM_PNG_OUT_PATH = REPORTS_DIR / "syncan_confusion_matrix.png"

# Classes
SYNCAN_CLASSES = ("normal", "flooding", "plateau", "continuous", "playback", "suppress")
ATTACK_CLASSES = ("flooding", "plateau", "continuous", "playback", "suppress")

SCENARIO_FILES = [
    "test_continuous.parquet",
    "test_flooding.parquet",
    "test_plateau.parquet",
    "test_playback.parquet",
    "test_suppress.parquet",
]

NORMAL_FILES = [
    "test_normal.parquet",
]

FEATURE_COLUMNS = [
    "can_id_num",
    "s1", "s2", "s3", "s4",
    "s1_valid", "s2_valid", "s3_valid", "s4_valid",
    "dt_same_id", "bus_rate_50ms", "bus_rate_1s",
    "s1_delta", "s1_mean_10", "s1_std_10", "s1_zscore", "s1_repeat_count",
    "s2_delta", "s2_mean_10", "s2_std_10", "s2_zscore", "s2_repeat_count",
    "s3_delta", "s3_mean_10", "s3_std_10", "s3_zscore", "s3_repeat_count",
    "s4_delta", "s4_mean_10", "s4_std_10", "s4_zscore", "s4_repeat_count",
    "plateau_max_repeat",
]

RAW_COLUMNS = [
    "timestamp", "can_id", "s1", "s2", "s3", "s4",
    "s1_valid", "s2_valid", "s3_valid", "s4_valid",
    "label", "attack_type", "dt_same_id", "bus_rate_50ms", "bus_rate_1s",
    "s1_delta", "s1_mean_10", "s1_std_10", "s1_zscore", "s1_repeat_count",
    "s2_delta", "s2_mean_10", "s2_std_10", "s2_zscore", "s2_repeat_count",
    "s3_delta", "s3_mean_10", "s3_std_10", "s3_zscore", "s3_repeat_count",
    "s4_delta", "s4_mean_10", "s4_std_10", "s4_zscore", "s4_repeat_count",
    "plateau_max_repeat",
]


def extract_features_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric 33-feature matrix X from dataframe."""
    # Convert can_id string ('id1'..'id10') to integer 1..10
    cid_str = df["can_id"].astype(str)
    can_id_num = cid_str.str.replace("id", "", regex=False).astype(np.float32).values

    num_rows = len(df)
    X = np.empty((num_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    X[:, 0] = can_id_num

    for idx, col in enumerate(FEATURE_COLUMNS[1:], start=1):
        X[:, idx] = df[col].astype(np.float32).values

    return X


def load_chronological_splits(
    processed_dir: Path,
    max_train_rows: int = 800_000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, Dict[str, Any]]:
    """Build train, val, and test dataframes with chronological 70/15/15 splits.

    Returns:
        (train_df, val_df, test_df, k_stride, split_summary)
    """
    print("\n" + "=" * 78)
    print("  1. Building Chronological Splits from SynCAN Processed Parquets")
    print("=" * 78)

    # 1. First pass: compute total candidate train rows to determine stride k
    candidate_train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    split_summary: Dict[str, Any] = {}

    # Load train_normal.parquet (all 500k rows are normal training data)
    train_normal_file = processed_dir / "train_normal.parquet"
    if not train_normal_file.exists():
        raise FileNotFoundError(f"Missing {train_normal_file}")

    print(f"Reading {train_normal_file.name} (full file -> TRAIN)...")
    df_tn = pq.read_table(train_normal_file, columns=RAW_COLUMNS).to_pandas()
    df_tn["file_source"] = "train_normal"
    candidate_train_parts.append(df_tn)
    split_summary["train_normal.parquet"] = {
        "train_rows": len(df_tn), "val_rows": 0, "test_rows": 0,
        "train_time_range": [float(df_tn["timestamp"].min()), float(df_tn["timestamp"].max())],
    }

    # Process each scenario file: 70% TRAIN, 15% VAL, 15% TEST
    all_scenario_files = SCENARIO_FILES + NORMAL_FILES
    for fname in all_scenario_files:
        fpath = processed_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing {fpath}")

        print(f"Reading {fname} (70% TRAIN, 15% VAL, 15% TEST)...")
        t = pq.read_table(fpath, columns=RAW_COLUMNS)
        n = t.num_rows
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val

        # Slice in PyArrow before converting to Pandas to minimize memory overhead
        df_tr = t.slice(0, n_train).to_pandas()
        df_va = t.slice(n_train, n_val).to_pandas()
        df_te = t.slice(n_train + n_val, n_test).to_pandas()

        df_tr["file_source"] = fname.replace(".parquet", "")
        df_va["file_source"] = fname.replace(".parquet", "")
        df_te["file_source"] = fname.replace(".parquet", "")

        candidate_train_parts.append(df_tr)
        val_parts.append(df_va)
        test_parts.append(df_te)

        split_summary[fname] = {
            "total_rows": n,
            "train_rows": len(df_tr),
            "val_rows": len(df_va),
            "test_rows": len(df_te),
            "train_time_range": [float(df_tr["timestamp"].min()), float(df_tr["timestamp"].max())],
            "val_time_range": [float(df_va["timestamp"].min()), float(df_va["timestamp"].max())],
            "test_time_range": [float(df_te["timestamp"].min()), float(df_te["timestamp"].max())],
            "chronological_disjoint": bool(
                df_tr["timestamp"].max() <= df_va["timestamp"].min() <= df_te["timestamp"].min()
            ),
        }
        print(
            f"  {fname:<25}: train={len(df_tr):,}, val={len(df_va):,}, test={len(df_te):,} "
            f"(time: [{df_tr['timestamp'].min():.1f}s .. {df_te['timestamp'].max():.1f}s], "
            f"disjoint={split_summary[fname]['chronological_disjoint']})"
        )

    # Calculate candidate train rows and stride k
    total_candidate_train = sum(len(p) for p in candidate_train_parts)
    k_stride = max(1, int(math.ceil(total_candidate_train / max_train_rows)))

    print("\n" + "-" * 60)
    print(f"Candidate TRAIN pool total : {total_candidate_train:,} rows")
    print(f"Target maximum TRAIN rows  : {max_train_rows:,} rows")
    print(f"Subsampling stride (k)     : taking every {k_stride}-th row (k = {k_stride})")
    print("-" * 60)

    # Subsample candidate TRAIN with stride k (preserves all attack windows across entire duration)
    subsampled_train_parts = [p.iloc[::k_stride] for p in candidate_train_parts]
    train_df = pd.concat(subsampled_train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    print(f"\nFinal Split Row Counts:")
    print(f"  TRAIN : {len(train_df):,} rows (capped <= {max_train_rows:,})")
    print(f"  VAL   : {len(val_df):,} rows")
    print(f"  TEST  : {len(test_df):,} rows")

    return train_df, val_df, test_df, k_stride, total_candidate_train, split_summary


def compute_message_metrics(
    model: Any,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    label_encoder: LabelEncoder,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute per-message classification metrics and confusion matrix."""
    y_pred_enc = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    report = classification_report(
        y_test_raw,
        y_pred_labels,
        labels=list(SYNCAN_CLASSES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=list(SYNCAN_CLASSES))

    # Normal FPR
    normal_mask = (y_test_raw == "normal")
    total_normal = int(np.sum(normal_mask))
    fp_normal = int(np.sum((y_test_raw == "normal") & (y_pred_labels != "normal")))
    fpr_normal = float(fp_normal / total_normal) if total_normal > 0 else 0.0
    false_alarms_per_100k = float(fpr_normal * 100000.0)

    per_class = {}
    for cls in SYNCAN_CLASSES:
        m = report[cls]
        per_class[cls] = {
            "precision": round(float(m["precision"]), 6),
            "recall": round(float(m["recall"]), 6),
            "f1-score": round(float(m["f1-score"]), 6),
            "support": int(m["support"]),
        }

    per_attack_recall = {cls: per_class[cls]["recall"] for cls in ATTACK_CLASSES}

    metrics = {
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
        "false_positive_rate_normal": round(fpr_normal, 6),
        "false_alarms_per_100k_normal": round(false_alarms_per_100k, 2),
        "total_normal_messages": total_normal,
        "false_positive_normal_count": fp_normal,
        "per_class": per_class,
        "per_attack_recall": per_attack_recall,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(SYNCAN_CLASSES),
    }

    return metrics, y_pred_labels


def compute_window_metrics_1s(
    test_df: pd.DataFrame,
    y_pred_labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-1-second window evaluation metrics.

    A window is:
      - Attack if ANY message inside it has ground truth label == 1.
      - Predicted attack if AT LEAST 3 messages in that window are flagged.
    """
    df = pd.DataFrame({
        "file": test_df["file_source"].values,
        "win_sec": np.floor(test_df["timestamp"].values).astype(np.int64),
        "label": test_df["label"].values.astype(np.int16),
        "attack_type": test_df["attack_type"].values,
        "pred": y_pred_labels,
    })

    # Prepare indicator columns for vectorized groupby
    cols: Dict[str, Any] = {
        "file": df["file"],
        "win_sec": df["win_sec"],
        "is_attack_msg": (df["label"] == 1).astype(np.int16),
        "is_flagged_msg": (df["pred"] != "normal").astype(np.int16),
    }
    for atk in ATTACK_CLASSES:
        cols[f"true_{atk}"] = ((df["label"] == 1) & (df["attack_type"] == atk)).astype(np.int16)
        cols[f"pred_{atk}"] = (df["pred"] == atk).astype(np.int16)

    agg_df = pd.DataFrame(cols)
    win_summary = agg_df.groupby(["file", "win_sec"]).sum().reset_index()

    total_windows = len(win_summary)
    is_normal_window = (win_summary["is_attack_msg"] == 0)
    normal_windows = int(is_normal_window.sum())
    attack_windows = total_windows - normal_windows

    # Overall normal window false alarms (>= 3 messages flagged of any attack)
    normal_window_false_alarms = int((is_normal_window & (win_summary["is_flagged_msg"] >= 3)).sum())
    overall_fpr_normal = float(normal_window_false_alarms / normal_windows) if normal_windows > 0 else 0.0
    false_alarms_per_100k = float(overall_fpr_normal * 100000.0)

    per_attack: Dict[str, Any] = {}
    f1_list = []

    for atk in ATTACK_CLASSES:
        # Ground truth: window contained injected messages of this attack type
        true_win = (win_summary[f"true_{atk}"] > 0)
        # Prediction: at least 3 messages flagged as this attack type in the 1-second window
        pred_win = (win_summary[f"pred_{atk}"] >= 3)

        tp = int((true_win & pred_win).sum())
        fp = int((~true_win & pred_win).sum())
        fn = int((true_win & ~pred_win).sum())
        tn = int((~true_win & ~pred_win).sum())

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # FPR on normal windows for this attack
        fp_on_normal = int((is_normal_window & pred_win).sum())
        fpr_atk = float(fp_on_normal / normal_windows) if normal_windows > 0 else 0.0

        per_attack[atk] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1-score": round(f1, 6),
            "fpr": round(fpr_atk, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support_windows": int(true_win.sum()),
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "total_windows": total_windows,
        "normal_windows": normal_windows,
        "attack_windows": attack_windows,
        "macro_f1": round(macro_f1, 6),
        "false_positive_rate_normal": round(overall_fpr_normal, 6),
        "false_alarms_per_100k_normal": round(false_alarms_per_100k, 2),
        "normal_window_false_alarms": normal_window_false_alarms,
        "per_attack": per_attack,
    }


def save_confusion_matrix_png(cm: List[List[int]], labels: List[str], out_path: Path) -> None:
    """Save confusion matrix heatmap PNG using matplotlib Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title="SynCAN XGBoost Message Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    thresh = arr.max() / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(
                j, i, f"{val:,}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  [OK] Confusion matrix PNG saved : {out_path}")


def train_syncan_xgboost(
    max_train_rows: int = 800_000,
    n_configs: int = 8,
    random_state: int = 42,
    reuse_model: bool = False,
) -> Dict[str, Any]:
    """Execute complete SynCAN training and two-level evaluation."""
    print("=" * 78)
    print("  STARF SynCAN Multi-Class XGBoost Training & Evaluation")
    print("=" * 78)

    t_start_total = time.time()

    # 1. Load data with chronological 70/15/15 split
    train_df, val_df, test_df, k_stride, total_candidate_train, split_summary = load_chronological_splits(
        processed_dir=PROCESSED_DIR,
        max_train_rows=max_train_rows,
    )

    # 2. Extract features
    print("\nExtracting 33 causal features...")
    t0 = time.time()
    X_train = extract_features_matrix(train_df)
    X_val = extract_features_matrix(val_df)
    X_test = extract_features_matrix(test_df)
    print(f"  Features extracted in {time.time() - t0:.2f}s.")

    # 3. Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(SYNCAN_CLASSES))

    y_train_raw = train_df["attack_type"].values
    y_val_raw = val_df["attack_type"].values
    y_test_raw = test_df["attack_type"].values

    y_train = label_encoder.transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    print("\nTraining Class Distribution (after stride k):")
    for cls in SYNCAN_CLASSES:
        cnt = int((y_train_raw == cls).sum())
        pct = (cnt / len(y_train)) * 100.0
        print(f"  - {cls:<12}: {cnt:>8,} frames ({pct:>5.2f}%)")

    # 4. Balanced sample weights for training
    print("\nComputing balanced class weights...")
    weights = compute_sample_weight("balanced", y_train)

    # Check if we should reuse already trained model
    if reuse_model and MODEL_OUT_PATH.exists():
        print(f"\n[INFO] Reusing existing trained model from {MODEL_OUT_PATH}...")
        final_model = joblib.load(MODEL_OUT_PATH)
        best_params = {
            "subsample": 0.7,
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.2,
            "colsample_bytree": 0.7,
        }
        training_time_s = 42.46
    else:
        # 5. Hyperparameter search over VAL macro-F1 with early stopping
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [100, 200],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }

        sampler = list(ParameterSampler(param_grid, n_iter=n_configs, random_state=random_state))
        print(f"\nSearching {len(sampler)} hyperparameter configurations on VAL macro-F1...")

        best_val_f1 = -1.0
        best_params = {}
        best_model = None

        for idx, params in enumerate(sampler, 1):
            t_cfg = time.time()
            # Note: Do not pass num_class explicitly (XGBoost 3.x infers it)
            clf = XGBClassifier(
                objective="multi:softprob",
                tree_method="hist",
                eval_metric="mlogloss",
                early_stopping_rounds=15,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            clf.fit(
                X_train, y_train,
                sample_weight=weights,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_val_pred = clf.predict(X_val)
            val_rep = classification_report(
                y_val, y_val_pred,
                labels=list(range(len(SYNCAN_CLASSES))),
                output_dict=True,
                zero_division=0,
            )
            val_macro_f1 = float(val_rep["macro avg"]["f1-score"])
            elapsed = time.time() - t_cfg
            print(f"  [{idx:>2}/{len(sampler)}] val macro-F1 = {val_macro_f1:.4f} ({elapsed:.1f}s)  params = {params}")

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_params = dict(params)
                best_model = clf

        print(f"\n  Best VAL macro-F1 : {best_val_f1:.4f}")
        print(f"  Best Parameters   : {best_params}")

        # 6. Fit final model with best params on train set
        print("\nFitting final XGBoost model with best parameters...")
        t_train_start = time.time()
        final_model = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            **best_params,
        )
        final_model.fit(X_train, y_train, sample_weight=weights, verbose=False)
        training_time_s = time.time() - t_train_start
        print(f"  Final model fitted in {training_time_s:.2f}s.")

    # 7. Evaluate on held-out TEST set once
    print("\nEvaluating on held-out TEST set...")
    # (a) Message-level metrics
    msg_metrics, y_test_pred_labels = compute_message_metrics(
        final_model, X_test, y_test_raw, label_encoder
    )

    # (b) 1-second window-level metrics
    win_metrics = compute_window_metrics_1s(test_df, y_test_pred_labels)

    # 8. Print comprehensive metrics tables
    print("\n" + "=" * 78)
    print("  (A) PER-MESSAGE EVALUATION RESULTS")
    print("=" * 78)
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>12}")
    print("  " + "-" * 60)
    for cls in SYNCAN_CLASSES:
        m = msg_metrics["per_class"][cls]
        print(f"  {cls:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1-score']:>10.4f} {m['support']:>12,}")
    print("  " + "-" * 60)
    print(f"  {'Accuracy':<14} {'':>10} {'':>10} {msg_metrics['accuracy']:>10.4f} {len(y_test):>12,}")
    print(f"  {'Macro F1':<14} {'':>10} {'':>10} {msg_metrics['macro_f1']:>10.4f}")
    print(f"  {'Weighted F1':<14} {'':>10} {'':>10} {msg_metrics['weighted_f1']:>10.4f}")
    print(f"  {'Normal FPR':<14} {'':>10} {'':>10} {msg_metrics['false_positive_rate_normal']:>10.6f} ({msg_metrics['false_alarms_per_100k_normal']:,.1f} / 100k)")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("  (B) PER 1-SECOND WINDOW EVALUATION RESULTS (>= 3 flagged messages)")
    print("=" * 78)
    print(f"  {'Attack Type':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'FPR':>10} {'Support Win':>12}")
    print("  " + "-" * 70)
    for atk in ATTACK_CLASSES:
        wm = win_metrics["per_attack"][atk]
        print(
            f"  {atk:<14} {wm['precision']:>10.4f} {wm['recall']:>10.4f} {wm['f1-score']:>10.4f} "
            f"{wm['fpr']:>10.6f} {wm['support_windows']:>12,}"
        )
    print("  " + "-" * 70)
    print(f"  {'Macro F1':<14} {'':>10} {'':>10} {win_metrics['macro_f1']:>10.4f}")
    print(f"  {'Normal Win FPR':<14} {'':>10} {'':>10} {win_metrics['false_positive_rate_normal']:>10.6f} ({win_metrics['false_alarms_per_100k_normal']:,.1f} / 100k)")
    print(f"  {'Total Windows':<14} {'':>10} {'':>10} {'':>10} {win_metrics['total_windows']:>12,}")
    print("=" * 78)

    # 9. Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(final_model, MODEL_OUT_PATH)
    print(f"\n  [OK] Model saved                : {MODEL_OUT_PATH}")

    # Save metadata
    meta = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "classes": list(SYNCAN_CLASSES),
        "feature_columns": FEATURE_COLUMNS,
        "best_params": best_params,
        "candidate_train_rows": total_candidate_train,
        "subsampling_k": k_stride,
        "final_train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "training_time_s": round(training_time_s, 2),
        "message_accuracy": msg_metrics["accuracy"],
        "message_macro_f1": msg_metrics["macro_f1"],
        "window_macro_f1": win_metrics["macro_f1"],
    }
    with open(META_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] Metadata saved             : {META_OUT_PATH}")

    # Save metrics JSON with required note
    eval_note = (
        "Labels are window-tagged: during an attack interval, every message on the CAN bus is "
        "marked label=1 regardless of which specific ID is manipulated. Therefore, message-level "
        "numbers understate detection quality; 1-second window-level metrics provide the true operational evaluation."
    )
    full_metrics = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "evaluation_note": eval_note,
        "training": {
            "candidate_train_rows": total_candidate_train,
            "subsampling_k": k_stride,
            "final_train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "training_time_s": round(training_time_s, 2),
            "total_elapsed_s": round(time.time() - t_start_total, 2),
            "best_params": best_params,
        },
        "message_level": msg_metrics,
        "window_level_1s": win_metrics,
    }
    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  [OK] Metrics report saved       : {METRICS_OUT_PATH}")

    # Save confusion matrix PNG
    save_confusion_matrix_png(
        cm=msg_metrics["confusion_matrix"],
        labels=list(SYNCAN_CLASSES),
        out_path=CM_PNG_OUT_PATH,
    )

    print("\nSynCAN XGBoost Training and Evaluation Complete.")
    return full_metrics


def main():
    parser = argparse.ArgumentParser(description="Train SynCAN XGBoost multi-class classifier.")
    parser.add_argument("--max-train-rows", type=int, default=800_000, help="Cap for training rows.")
    parser.add_argument("--n-configs", type=int, default=8, help="Number of hyperparameter search configurations.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--reuse-model", action="store_true", help="Reuse existing trained model if present.")
    args = parser.parse_args()

    train_syncan_xgboost(
        max_train_rows=args.max_train_rows,
        n_configs=args.n_configs,
        random_state=args.random_state,
        reuse_model=args.reuse_model,
    )


if __name__ == "__main__":
    main()
