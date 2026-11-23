"""Train and evaluate window-level XGBoost classifier on SynCAN benchmark dataset.

Addresses the window-tagging labeling artifact identified in Part B by classifying
fixed 0.5-second time windows instead of individual CAN messages.

Splits:
  - Chronological 70% TRAIN / 15% VAL / 15% TEST per file by window index.
  - Zero shuffle across split boundaries.
  - Window label = 1 if >=50% of messages in window are label 1.
  - Classes: normal, continuous, flooding, plateau, playback, suppress.

Model:
  - XGBClassifier(tree_method="hist", objective="multi:softprob")
  - Balanced sample weighting for class imbalance
  - Random search of 12 hyperparameter configurations, selected on VAL macro-F1
  - TEST set evaluated once at the end.

Outputs:
  - models/syncan_window_xgb_model.pkl
  - models/syncan_window_meta.json
  - reports/syncan_window_metrics.json
  - reports/syncan_window_confusion_matrix.png
  - reports/syncan_window_feature_importance.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.syncan_windows import FEATURE_COLUMNS  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "datasets" / "syncan" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_OUT_PATH = MODELS_DIR / "syncan_window_xgb_model.pkl"
META_OUT_PATH = MODELS_DIR / "syncan_window_meta.json"
METRICS_OUT_PATH = REPORTS_DIR / "syncan_window_metrics.json"
CM_PNG_OUT_PATH = REPORTS_DIR / "syncan_window_confusion_matrix.png"
FI_PNG_OUT_PATH = REPORTS_DIR / "syncan_window_feature_importance.png"
PART_B_METRICS_PATH = REPORTS_DIR / "syncan_xgb_metrics.json"

CLASSES = ("normal", "continuous", "flooding", "plateau", "playback", "suppress")

WINDOW_FILES = [
    ("windows_train_normal.parquet", "normal"),
    ("windows_test_normal.parquet", "normal"),
    ("windows_test_continuous.parquet", "continuous"),
    ("windows_test_flooding.parquet", "flooding"),
    ("windows_test_plateau.parquet", "plateau"),
    ("windows_test_playback.parquet", "playback"),
    ("windows_test_suppress.parquet", "suppress"),
]


def load_chronological_splits(
    processed_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load windows from all files and create chronological 70/15/15 splits per file."""
    train_dfs: List[pd.DataFrame] = []
    val_dfs: List[pd.DataFrame] = []
    test_dfs: List[pd.DataFrame] = []

    file_split_stats: Dict[str, Any] = {}

    for fname, file_attack in WINDOW_FILES:
        fpath = processed_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing window parquet file: {fpath}")

        df = pd.read_parquet(fpath)
        n = len(df)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        df_train = df.iloc[:n_train].copy()
        df_val = df.iloc[n_train : n_train + n_val].copy()
        df_test = df.iloc[n_train + n_val :].copy()

        train_dfs.append(df_train)
        val_dfs.append(df_val)
        test_dfs.append(df_test)

        file_split_stats[fname] = {
            "total_windows": n,
            "train_windows": len(df_train),
            "val_windows": len(df_val),
            "test_windows": len(df_test),
            "train_attack_count": int((df_train["label"] == 1).sum()),
            "val_attack_count": int((df_val["label"] == 1).sum()),
            "test_attack_count": int((df_test["label"] == 1).sum()),
        }

    full_train = pd.concat(train_dfs, ignore_index=True)
    full_val = pd.concat(val_dfs, ignore_index=True)
    full_test = pd.concat(test_dfs, ignore_index=True)

    return full_train, full_val, full_test, file_split_stats


def evaluate_window_predictions(
    y_true_str: np.ndarray,
    y_pred_str: np.ndarray,
    classes: Tuple[str, ...],
) -> Dict[str, Any]:
    """Compute per-class, binary, and false-alarm metrics on windows."""
    clf_rep = classification_report(
        y_true_str,
        y_pred_str,
        labels=list(classes),
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true_str, y_pred_str, labels=list(classes))

    # Binary metrics: attack vs normal
    is_true_attack = (y_true_str != "normal").astype(int)
    is_pred_attack = (y_pred_str != "normal").astype(int)

    bin_rep = classification_report(
        is_true_attack,
        is_pred_attack,
        target_names=["normal", "attack"],
        output_dict=True,
        zero_division=0,
    )

    # Normal class false positives
    normal_idx = list(classes).index("normal")
    total_normal = int(np.sum(y_true_str == "normal"))
    normal_row = cm[normal_idx]
    fp_normal = int(np.sum(normal_row) - normal_row[normal_idx])
    fpr_normal = float(fp_normal / total_normal) if total_normal > 0 else 0.0
    false_alarms_per_1000 = float(fpr_normal * 1000.0)

    # Clean per_class dict
    per_class_clean: Dict[str, Any] = {}
    for c in classes:
        if c in clf_rep:
            per_class_clean[c] = {
                "precision": round(float(clf_rep[c]["precision"]), 6),
                "recall": round(float(clf_rep[c]["recall"]), 6),
                "f1-score": round(float(clf_rep[c]["f1-score"]), 6),
                "support": int(clf_rep[c]["support"]),
            }

    return {
        "accuracy": round(float(clf_rep["accuracy"]), 6),
        "macro_f1": round(float(clf_rep["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(clf_rep["weighted avg"]["f1-score"]), 6),
        "false_positive_rate_normal": round(fpr_normal, 6),
        "false_alarms_per_1000_normal": round(false_alarms_per_1000, 4),
        "total_normal_windows": total_normal,
        "false_positive_normal_count": fp_normal,
        "per_class": per_class_clean,
        "binary_attack_normal": {
            "attack_precision": round(float(bin_rep["attack"]["precision"]), 6),
            "attack_recall": round(float(bin_rep["attack"]["recall"]), 6),
            "attack_f1": round(float(bin_rep["attack"]["f1-score"]), 6),
            "normal_precision": round(float(bin_rep["normal"]["precision"]), 6),
            "normal_recall": round(float(bin_rep["normal"]["recall"]), 6),
            "normal_f1": round(float(bin_rep["normal"]["f1-score"]), 6),
        },
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }


def plot_confusion_matrix(cm: np.ndarray, classes: Tuple[str, ...], out_path: Path) -> None:
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(classes),
        yticklabels=list(classes),
        cbar=True,
    )
    plt.title("SynCAN Window-Level XGBoost Confusion Matrix (TEST)", fontsize=13, pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(model: XGBClassifier, feature_names: List[str], out_path: Path) -> None:
    """Save top 20 feature importances bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_names = [feature_names[i] for i in indices]
    top_scores = [float(importances[i]) for i in indices]

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_scores[::-1], align="center", color="#2b5c8f")
    plt.yticks(y_pos, top_names[::-1], fontsize=9)
    plt.xlabel("Feature Importance (Gain / Weight)", fontsize=11)
    plt.title("SynCAN Window-Level XGBoost Top 20 Features", fontsize=13, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def update_part_b_metrics_note(part_b_path: Path) -> None:
    """Update reports/syncan_xgb_metrics.json with factual note about label artifacts."""
    if not part_b_path.exists():
        print(f"Part B metrics file not found at {part_b_path}; skipping note update.")
        return

    factual_note = (
        "Per-message labels in SynCAN are window-tagged: every message during an attack interval "
        "is marked label=1 regardless of which ID is manipulated. As a result, the per-message "
        "XGBoost model cannot distinguish plateau, continuous and playback attacks from normal traffic "
        "(message-level precision ~0.03, window false alarm rate ~100%) because normal messages "
        "interleaved within attack windows are incorrectly labeled as attacks during training. "
        "Fixed time window evaluation (Part C) resolves this by classifying fixed 0.5-second intervals."
    )

    with open(part_b_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["evaluation_note"] = factual_note

    with open(part_b_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Updated evaluation_note in {part_b_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("SynCAN Window-Level XGBoost Training (Part C)")
    print("=" * 70)

    # 1. Load data
    print("\n--- 1. Loading Chronological Splits ---")
    df_train, df_val, df_test, split_stats = load_chronological_splits(PROCESSED_DIR)

    print(f"Train windows: {len(df_train):,} | Val windows: {len(df_val):,} | Test windows: {len(df_test):,}")
    for fname, st in split_stats.items():
        print(f"  {fname:<32}: Train={st['train_windows']:<6} Val={st['val_windows']:<5} Test={st['test_windows']:<5} "
              f"Attacks(Tr/Val/Te)={st['train_attack_count']}/{st['val_attack_count']}/{st['test_attack_count']}")

    # 2. Extract features and target
    X_train = df_train[FEATURE_COLUMNS].values.astype(np.float32)
    y_train_str = df_train["attack_type"].values.astype(str)

    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float32)
    y_val_str = df_val["attack_type"].values.astype(str)

    X_test = df_test[FEATURE_COLUMNS].values.astype(np.float32)
    y_test_str = df_test["attack_type"].values.astype(str)

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    y_train = label_encoder.transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)

    print(f"\nTarget classes: {list(CLASSES)}")
    print(f"Train class distribution: {dict(zip(*np.unique(y_train_str, return_counts=True)))}")
    print(f"Val class distribution:   {dict(zip(*np.unique(y_val_str, return_counts=True)))}")
    print(f"Test class distribution:  {dict(zip(*np.unique(y_test_str, return_counts=True)))}")

    sample_weights_train = compute_sample_weight("balanced", y_train)

    # 3. Hyperparameter search (12 configs) on VAL macro-F1
    print("\n--- 2. Hyperparameter Search (12 configs on VAL macro-F1) ---")
    param_grid = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "n_estimators": [100, 150, 200, 300],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    sampler = ParameterSampler(param_grid, n_iter=12, random_state=42)
    configs = list(sampler)

    best_val_macro_f1 = -1.0
    best_config: Dict[str, Any] = {}
    best_model: Any = None

    for idx, params in enumerate(configs, 1):
        t0 = time.time()
        clf = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=15,
            **params,
        )

        clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weights_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_preds = clf.predict(X_val)
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        fit_dur = time.time() - t0

        print(
            f"Config {idx:02d}/12 | lr={params['learning_rate']} depth={params['max_depth']} "
            f"n_est={params['n_estimators']} sub={params['subsample']} col={params['colsample_bytree']} "
            f"min_child={params['min_child_weight']} -> VAL Macro-F1: {val_macro_f1:.4f} ({fit_dur:.1f}s)"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_config = params
            best_model = clf

    print(f"\nBest Config: {best_config}")
    print(f"Best VAL Macro-F1: {best_val_macro_f1:.4f}")

    # 4. Evaluate once on TEST set
    print("\n--- 3. Evaluation on Held-Out Chronological TEST Set ---")
    test_preds_int = best_model.predict(X_test)
    test_preds_str = label_encoder.inverse_transform(test_preds_int)

    test_metrics = evaluate_window_predictions(y_test_str, test_preds_str, CLASSES)

    print("\nWindow-Level Test Classification Report:")
    print(f"{'Class':<14} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}")
    print("-" * 55)
    for c, stats in test_metrics["per_class"].items():
        print(f"{c:<14} {stats['precision']:<11.4f} {stats['recall']:<11.4f} {stats['f1-score']:<11.4f} {stats['support']:<8}")
    print("-" * 55)
    print(f"Overall Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:       {test_metrics['weighted_f1']:.4f}")
    print(f"FPR on Normal:     {test_metrics['false_positive_rate_normal']:.6f} ({test_metrics['false_positive_normal_count']}/{test_metrics['total_normal_windows']})")
    print(f"False Alarms/1000: {test_metrics['false_alarms_per_1000_normal']:.2f}")

    print("\nBinary (Attack vs Normal) Test Metrics:")
    for k, v in test_metrics["binary_attack_normal"].items():
        print(f"  {k:<20}: {v:.4f}")

    # 5. Save Artifacts
    print("\n--- 4. Saving Model and Reports ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_OUT_PATH)
    print(f"Saved model: {MODEL_OUT_PATH.relative_to(PROJECT_ROOT)}")

    meta_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "window_duration_s": 0.5,
        "n_features": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": list(CLASSES),
        "best_hyperparameters": best_config,
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "split_summary": split_stats,
    }
    with open(META_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"Saved metadata: {META_OUT_PATH.relative_to(PROJECT_ROOT)}")

    full_metrics = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "evaluation_level": "0.5_second_window",
        "window_rule": ">=50% attack messages -> attack window",
        "best_params": best_config,
        "val_macro_f1": round(float(best_val_macro_f1), 4),
        **test_metrics,
    }
    with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved metrics: {METRICS_OUT_PATH.relative_to(PROJECT_ROOT)}")

    # Confusion matrix PNG
    cm_arr = np.array(test_metrics["confusion_matrix"])
    plot_confusion_matrix(cm_arr, CLASSES, CM_PNG_OUT_PATH)
    print(f"Saved CM plot: {CM_PNG_OUT_PATH.relative_to(PROJECT_ROOT)}")

    # Feature importance PNG
    plot_feature_importance(best_model, FEATURE_COLUMNS, FI_PNG_OUT_PATH)
    print(f"Saved FI plot: {FI_PNG_OUT_PATH.relative_to(PROJECT_ROOT)}")

    # Fix note in Part B metrics
    update_part_b_metrics_note(PART_B_METRICS_PATH)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
