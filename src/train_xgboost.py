"""Train XGBoost multiclass attack classifier for STARF CAN Bus IDS.

Reuses the SAME data loading, feature extraction, and chronological split as
src/train_multiclass.py (Random Forest).  Uses sample_weight for class balance
instead of SMOTE so XGBoost handles imbalance natively.

Outputs
-------
  models/xgb_model.pkl          – trained XGBClassifier (joblib)
  models/xgb_meta.json          – hyperparameters and training metadata
  reports/xgb_metrics.json      – accuracy, per-class P/R/F1, macro-F1,
                                   confusion matrix, latency, feature importances
  reports/xgb_confusion_matrix.png
  reports/rf_metrics.json       – RF metrics in the SAME schema (no re-training;
                                   loads existing multiclass_model.pkl and
                                   re-formats existing classification report)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight

from backend.config import settings
from src.train_multiclass import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    PREPARED_DATASET_PATH,
    load_and_prepare_data,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_ROOT = settings.PROJECT_ROOT
XGB_MODEL_PATH   = _ROOT / "models"  / "xgb_model.pkl"
XGB_META_PATH    = _ROOT / "models"  / "xgb_meta.json"
XGB_METRICS_PATH = _ROOT / "reports" / "xgb_metrics.json"
XGB_CM_PNG       = _ROOT / "reports" / "xgb_confusion_matrix.png"
RF_METRICS_PATH  = _ROOT / "reports" / "rf_metrics.json"

# Existing RF artifacts (produced by train_multiclass.py)
RF_MODEL_PATH    = _ROOT / "models"  / "multiclass_model.pkl"
RF_LABEL_ENC_PATH = _ROOT / "models" / "multiclass_label_encoder.pkl"
RF_REPORT_PATH   = _ROOT / "reports" / "multiclass_classification_report.json"
RF_CM_PATH       = _ROOT / "reports" / "multiclass_confusion_matrix.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measure_latency(model, X_sample: np.ndarray, n_calls: int = 100) -> float:
    """Return mean latency in ms over n_calls predict_proba calls."""
    # warm-up
    for _ in range(3):
        model.predict_proba(X_sample)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.predict_proba(X_sample)
    return (time.perf_counter() - t0) / n_calls * 1000.0


def _build_metrics_dict(
    model,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    label_encoder,
    best_params: Dict[str, Any],
    training_time_s: float,
    model_type: str,
) -> Dict[str, Any]:
    """Evaluate model on X_test, measure latency, return unified metrics dict."""
    y_pred_enc = model.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    report = classification_report(
        y_test_raw,
        y_pred_labels,
        labels=list(CLASS_LABELS),
        output_dict=True,
    )
    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=list(CLASS_LABELS))

    overall_acc = float(report["accuracy"])
    macro_f1    = float(report["macro avg"]["f1-score"])

    per_class: Dict[str, Any] = {}
    for cls in CLASS_LABELS:
        m = report[cls]
        per_class[cls] = {
            "precision": round(float(m["precision"]), 6),
            "recall":    round(float(m["recall"]),    6),
            "f1-score":  round(float(m["f1-score"]),  6),
            "support":   int(m["support"]),
        }

    # Latency
    lat_batch1   = _measure_latency(model, X_test[:1],    n_calls=100)
    lat_batch1k  = _measure_latency(model, X_test[:1000], n_calls=100)

    # Feature importances (available for RF and XGB)
    feat_imp: Dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        for name, imp in zip(FEATURE_COLUMNS, model.feature_importances_):
            feat_imp[name] = round(float(imp), 6)

    return {
        "model_type":              model_type,
        "date":                    datetime.now(timezone.utc).isoformat(),
        "accuracy":                round(overall_acc, 6),
        "macro_f1":                round(macro_f1, 6),
        "per_class":               per_class,
        "confusion_matrix":        cm.tolist(),
        "confusion_matrix_labels": list(CLASS_LABELS),
        "latency_ms_batch1":       round(lat_batch1,  3),
        "latency_ms_batch1000":    round(lat_batch1k, 3),
        "feature_importances":     feat_imp,
        "best_params":             best_params,
        "training_time_s":         round(training_time_s, 1),
    }


def _save_confusion_matrix_png(cm: list, labels: list, out_path: Path) -> None:
    """Save a confusion-matrix heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title="XGBoost Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = arr.max() / 2.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, f"{arr[i, j]:,}",
                ha="center", va="center",
                color="white" if arr[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  [OK] Confusion matrix PNG : {out_path}")


# ---------------------------------------------------------------------------
# RF metrics in unified format (no re-training)
# ---------------------------------------------------------------------------

def build_rf_metrics() -> Dict[str, Any]:
    """
    Load the existing RF model + existing classification report and
    re-format them into the unified metrics schema.
    """
    print("\n--- Building RF metrics (unified format, no re-training) ---")
    rf_model = joblib.load(RF_MODEL_PATH)
    label_enc = joblib.load(RF_LABEL_ENC_PATH)

    with open(RF_REPORT_PATH, "r", encoding="utf-8") as f:
        rf_report = json.load(f)

    with open(RF_CM_PATH, "r", encoding="utf-8") as f:
        rf_cm_data = json.load(f)

    overall_acc = float(rf_report["accuracy"])
    macro_f1    = float(rf_report["macro avg"]["f1-score"])

    per_class: Dict[str, Any] = {}
    for cls in CLASS_LABELS:
        m = rf_report[cls]
        per_class[cls] = {
            "precision": round(float(m["precision"]), 6),
            "recall":    round(float(m["recall"]),    6),
            "f1-score":  round(float(m["f1-score"]),  6),
            "support":   int(m["support"]),
        }

    # We only have the test *labels* from the CM; to measure latency we need X.
    # Load a small sample from prepared.parquet (test portion is big; sample 1k rows).
    import pandas as pd
    from src.preprocess import build_frame_features

    prepared_path = _ROOT / "data" / "prepared.parquet"
    df = pd.read_parquet(prepared_path)
    sample = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    feat_df = build_frame_features(sample)
    X_sample = feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    lat_batch1  = _measure_latency(rf_model, X_sample[:1],    n_calls=100)
    lat_batch1k = _measure_latency(rf_model, X_sample[:1000], n_calls=100)

    feat_imp: Dict[str, float] = {}
    if hasattr(rf_model, "feature_importances_"):
        for name, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_):
            feat_imp[name] = round(float(imp), 6)

    return {
        "model_type":              "RandomForestClassifier",
        "date":                    datetime.now(timezone.utc).isoformat(),
        "accuracy":                round(overall_acc, 6),
        "macro_f1":                round(macro_f1, 6),
        "per_class":               per_class,
        "confusion_matrix":        rf_cm_data["confusion_matrix"],
        "confusion_matrix_labels": rf_cm_data["classes"],
        "latency_ms_batch1":       round(lat_batch1,  3),
        "latency_ms_batch1000":    round(lat_batch1k, 3),
        "feature_importances":     feat_imp,
        "best_params":             {
            "n_estimators": 100,
            "max_depth": 22,
            "min_samples_leaf": 2,
        },
        "training_time_s":         None,  # not re-measured
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_xgboost(
    max_train_rows: int = 300_000,
    random_state: int = 42,
    window_n: int = 50,
    n_search_configs: int = 12,
    val_fraction: float = 0.20,
) -> Dict[str, Any]:
    """Train XGBClassifier via random hyper-parameter search + early stopping."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed.  Run:  pip install xgboost>=2.0.0"
        ) from exc

    print("=" * 78)
    print("  STARF Stage-2 XGBoost Classifier Training (5 Classes)")
    print("=" * 78)

    # ------------------------------------------------------------------ #
    # 1. Load & split data (identical to RF pipeline)
    # ------------------------------------------------------------------ #
    train_feat_df, test_feat_df, split_info = load_and_prepare_data(
        dataset_path=PREPARED_DATASET_PATH,
        max_train_rows=max_train_rows,
        test_ratio=0.20,
        window_n=window_n,
    )

    # Prepare arrays
    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    label_enc.fit(list(CLASS_LABELS))

    X_full  = train_feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_full  = label_enc.transform(train_feat_df["label"].values)
    y_full_raw = train_feat_df["label"].values

    X_test  = test_feat_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test  = label_enc.transform(test_feat_df["label"].values)
    y_test_raw = test_feat_df["label"].values

    # ------------------------------------------------------------------ #
    # 2. Chronological val split from training data
    #    (last val_fraction rows of the concatenated, time-ordered block)
    # ------------------------------------------------------------------ #
    n_val = int(len(X_full) * val_fraction)
    n_tr  = len(X_full) - n_val

    X_tr, y_tr, y_tr_raw = X_full[:n_tr],  y_full[:n_tr],  y_full_raw[:n_tr]
    X_val, y_val         = X_full[n_tr:], y_full[n_tr:]

    print(f"\n  Train rows : {len(X_tr):,}")
    print(f"  Val rows   : {len(X_val):,}")
    print(f"  Test rows  : {len(X_test):,}")

    # Sample weights for training (balanced class weighting)
    weights = compute_sample_weight("balanced", y_tr)

    # ------------------------------------------------------------------ #
    # 3. Random hyper-parameter search
    # ------------------------------------------------------------------ #
    param_grid = {
        "max_depth":        [4, 6, 8, 10],
        "learning_rate":    [0.05, 0.1, 0.2],
        "n_estimators":     [100, 200, 300],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    sampler = list(
        ParameterSampler(param_grid, n_iter=n_search_configs, random_state=random_state)
    )

    print(f"\n  Searching {len(sampler)} hyper-parameter configurations ...\n")

    best_val_f1   = -1.0
    best_params: Dict[str, Any] = {}
    best_model    = None

    for idx, params in enumerate(sampler, 1):
        t0 = time.perf_counter()
        # Build a fixed-estimator version for search (no early stopping to avoid
        # label-range issues when a class is sparse in the val subset)
        search_params = dict(params)
        mdl = XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            **search_params,
        )
        mdl.fit(X_tr, y_tr, sample_weight=weights, verbose=False)
        y_val_pred = mdl.predict(X_val)
        val_rep    = classification_report(
            y_val, y_val_pred,
            labels=list(range(len(CLASS_LABELS))),
            output_dict=True,
            zero_division=0,
        )
        val_macro_f1 = float(val_rep["macro avg"]["f1-score"])
        elapsed      = time.perf_counter() - t0
        print(
            f"  [{idx:>2}/{len(sampler)}]  val macro-F1={val_macro_f1:.4f}  "
            f"({elapsed:.1f}s)  params={search_params}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_params = search_params
            best_model  = mdl

    print(f"\n  Best val macro-F1 = {best_val_f1:.4f}")
    print(f"  Best params       = {best_params}")

    # ------------------------------------------------------------------ #
    # 4. Final evaluation on HELD-OUT TEST SET (used exactly once)
    # ------------------------------------------------------------------ #
    print("\n--- Final evaluation on held-out test set ---")
    t_train_start = time.time()
    # Refit best model on the FULL training set (tr + val) with best params
    final_model = XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    full_weights = compute_sample_weight("balanced", y_full)
    final_model.fit(X_full, y_full, sample_weight=full_weights, verbose=False)
    training_time_s = time.time() - t_train_start

    xgb_metrics = _build_metrics_dict(
        model=final_model,
        X_test=X_test,
        y_test_raw=y_test_raw,
        label_encoder=label_enc,
        best_params=best_params,
        training_time_s=training_time_s,
        model_type="XGBClassifier",
    )

    # Print evaluation table
    print("\n" + "=" * 68)
    print("  XGBoost TEST SET RESULTS")
    print("=" * 68)
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 56)
    for cls in CLASS_LABELS:
        m = xgb_metrics["per_class"][cls]
        print(
            f"  {cls:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1-score']:>10.4f} {m['support']:>10,}"
        )
    print("  " + "-" * 56)
    print(f"  {'Accuracy':<14} {'':>10} {'':>10} {xgb_metrics['accuracy']:>10.4f} {len(y_test):>10,}")
    print(f"  {'Macro F1':<14} {'':>10} {'':>10} {xgb_metrics['macro_f1']:>10.4f}")
    print("=" * 68)
    print(f"\n  Latency  batch=1   : {xgb_metrics['latency_ms_batch1']:.3f} ms")
    print(f"  Latency  batch=1000: {xgb_metrics['latency_ms_batch1000']:.3f} ms")
    print(f"  Training time      : {xgb_metrics['training_time_s']} s")

    # ------------------------------------------------------------------ #
    # 5. Save artifacts
    # ------------------------------------------------------------------ #
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    XGB_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(final_model, XGB_MODEL_PATH)
    print(f"\n  [OK] XGB model saved      : {XGB_MODEL_PATH}")

    # Meta JSON (mirrors multiclass_meta.json structure)
    meta = {
        "date": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBClassifier",
        "feature_list": FEATURE_COLUMNS,
        "class_labels": list(CLASS_LABELS),
        "best_params": best_params,
        "accuracy": xgb_metrics["accuracy"],
        "macro_f1": xgb_metrics["macro_f1"],
        "split_method": "Chronological 80/20 per class + 20% val from train",
        "training_rows": len(X_full),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
    }
    with open(XGB_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [OK] XGB meta saved       : {XGB_META_PATH}")

    # Metrics JSON
    with open(XGB_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(xgb_metrics, f, indent=2)
    print(f"  [OK] XGB metrics saved    : {XGB_METRICS_PATH}")

    # Confusion matrix PNG
    _save_confusion_matrix_png(
        cm=xgb_metrics["confusion_matrix"],
        labels=list(CLASS_LABELS),
        out_path=XGB_CM_PNG,
    )

    # ------------------------------------------------------------------ #
    # 6. Build RF metrics in unified format (no re-training)
    # ------------------------------------------------------------------ #
    rf_metrics = build_rf_metrics()
    with open(RF_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(rf_metrics, f, indent=2)
    print(f"  [OK] RF metrics saved     : {RF_METRICS_PATH}")

    print("\n  Done.")
    return xgb_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost for STARF CAN Bus IDS.")
    parser.add_argument("--max-train-rows", type=int, default=300_000)
    parser.add_argument("--random-state",   type=int, default=42)
    parser.add_argument("--window-n",       type=int, default=50)
    parser.add_argument("--n-configs",      type=int, default=12,
                        help="Number of random hyper-parameter configs to try")
    args = parser.parse_args()
    train_xgboost(
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
        window_n=args.window_n,
        n_search_configs=args.n_configs,
    )


if __name__ == "__main__":
    main()
