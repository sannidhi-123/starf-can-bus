"""Held-out Evaluation for CAN Bus IDS Pipeline.

Evaluates three configurations on the chronological 80/20 held-out test block
from data/prepared.parquet (last 20% of each class in timestamp order):
  1. Rules only (any rule fired = attack)
  2. ML only (Random Forest on all frames, no rules)
  3. Hybrid (rules first, then ML classification)

Outputs:
  - reports/hybrid_evaluation.json
  - reports/hybrid_evaluation.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from backend.config import settings
from src.pipeline import HybridPipeline

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = settings.PROJECT_ROOT
DATA_PATH = PROJECT_ROOT / "data" / "prepared.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
JSON_OUTPUT_PATH = REPORTS_DIR / "hybrid_evaluation.json"
MD_OUTPUT_PATH = REPORTS_DIR / "hybrid_evaluation.md"

CLASS_LABELS = ["Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]
ATTACK_CLASSES = ["DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]
MAX_TEST_PER_CLASS = 20000  # Cap per class to keep evaluation fast & deterministic
TEST_SPLIT_RATIO = 0.20     # Same chronological 80/20 split as src/train_multiclass.py


def compute_binary_metrics(y_true_bin: List[int], y_pred_bin: List[int]) -> Dict[str, float]:
    """Compute binary intrusion detection metrics (0: Normal, 1: Attack)."""
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "fpr": round(fpr, 5),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


CONFUSION_OUTCOMES = ["Normal", "Rule violation", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]


def compute_confusion_matrix(y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix with true class as rows and predicted outcome as columns."""
    cm: Dict[str, Dict[str, int]] = {}
    for true_cls in CLASS_LABELS:
        cm[true_cls] = {col: 0 for col in CONFUSION_OUTCOMES}
        indices = [i for i, y in enumerate(y_true) if y == true_cls]
        for i in indices:
            pred = y_pred[i]
            if pred in cm[true_cls]:
                cm[true_cls][pred] += 1
            else:
                pass
    return cm


def format_confusion_matrix_md(cm: Dict[str, Dict[str, int]]) -> str:
    """Format 5x6 confusion matrix as a markdown table."""
    header = "| True Class | " + " | ".join(CONFUSION_OUTCOMES) + " | Total |"
    divider = "|:---| " + " | ".join(["---:"] * len(CONFUSION_OUTCOMES)) + " | ---:|"
    rows = [header, divider]
    for true_cls in CLASS_LABELS:
        counts = [f"{cm[true_cls][col]:,}" for col in CONFUSION_OUTCOMES]
        total = f"{sum(cm[true_cls].values()):,}"
        rows.append(f"| **{true_cls}** | " + " | ".join(counts) + f" | {total} |")
    return "\n".join(rows)


def evaluate() -> Dict[str, Any]:
    print("=" * 70)
    print("  CAN BUS IDS — HELD-OUT CHRONOLOGICAL EVALUATION")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Prepared dataset not found at: {DATA_PATH}")

    t_start = time.time()
    print(f"\n1. Loading prepared dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Total dataset size: {len(df):,} frames")

    # Verify expected classes
    present_classes = sorted(df["label"].unique())
    print(f"   Classes present: {present_classes}")
    for cls in CLASS_LABELS:
        if cls not in present_classes:
            raise ValueError(f"Required class '{cls}' missing from dataset!")

    print(f"\n2. Applying chronological 80/20 held-out split per class:")
    print(f"   Per-class test cap: up to {MAX_TEST_PER_CLASS:,} contiguous frames")

    test_subsets: Dict[str, pd.DataFrame] = {}
    for cls in CLASS_LABELS:
        cls_df = df[df["label"] == cls].sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(cls_df) * (1.0 - TEST_SPLIT_RATIO))
        test_block = cls_df.iloc[split_idx:]
        capped_block = test_block.iloc[:MAX_TEST_PER_CLASS].reset_index(drop=True)
        test_subsets[cls] = capped_block
        print(f"   - {cls:<12}: total={len(cls_df):>7,}, test_20%={len(test_block):>6,}, capped_eval={len(capped_block):>6,}")

    total_eval_frames = sum(len(s) for s in test_subsets.values())
    print(f"   Total evaluation frames across 5 classes: {total_eval_frames:,}")

    # Load HybridPipeline
    print("\n3. Initializing HybridPipeline (loading model and rules_config.json)...")
    pipeline = HybridPipeline()
    feature_names = pipeline.feature_names
    print(f"   Features ({len(feature_names)}): {feature_names}")
    print(f"   Rules whitelist size: {len(pipeline.whitelist)} CAN IDs")

    # Containers for results
    class_eval: Dict[str, Dict[str, Any]] = {}

    y_true_all: List[str] = []
    rules_pred_bin_all: List[int] = []
    rules_pred_cls_all: List[str] = []
    ml_pred_bin_all: List[int] = []
    ml_pred_cls_all: List[str] = []
    hybrid_pred_bin_all: List[int] = []
    hybrid_pred_cls_all: List[str] = []
    hybrid_stage_all: List[str] = []

    print("\n4. Running sequential evaluation in timestamp order per class...")

    for cls in CLASS_LABELS:
        sub_df = test_subsets[cls]
        n_frames = len(sub_df)
        t_cls = time.time()

        # Each class evaluated with fresh StreamState and fresh RuleEngine
        state = pipeline.new_stream_state()
        engine = pipeline.new_rule_state()

        timestamps = sub_df["timestamp"].to_numpy(dtype=np.float64)
        can_ids = sub_df["can_id"].to_numpy(dtype=np.int64)
        dlcs = sub_df["dlc"].to_numpy(dtype=np.int64)

        byte_cols = [f"d{i}" for i in range(8)]
        raw_bytes = np.zeros((n_frames, 8), dtype=np.int64)
        for i, c in enumerate(byte_cols):
            raw_bytes[:, i] = [HybridPipeline._parse_byte_val(v) for v in sub_df[c]]

        rule_fired = np.zeros(n_frames, dtype=bool)
        X = np.zeros((n_frames, len(feature_names)), dtype=np.float32)

        for i in range(n_frames):
            ts = timestamps[i]
            cid = int(can_ids[i])
            # 1. Rule check
            trig = engine.process_frame(ts, cid)
            rule_fired[i] = len(trig) > 0
            # 2. State update & feature building
            rep, iat = state.update(cid, ts)
            X[i] = pipeline._build_feature_row(cid, rep, iat, dlcs[i], raw_bytes[i])

        # Batch predict ML model
        probs = pipeline._model.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        ml_pred_classes = [pipeline._classes[idx] for idx in top_indices]

        # Hybrid decisions
        hybrid_stages = []
        hybrid_bin = []
        hybrid_cls = []

        rules_bin = []
        rules_cls = []
        ml_bin = []

        for i in range(n_frames):
            is_rule_hit = rule_fired[i]
            m_cls = ml_pred_classes[i]

            # Rules only: any rule fired = attack (1), else normal (0)
            rules_bin.append(1 if is_rule_hit else 0)
            rules_cls.append("Rule violation" if is_rule_hit else "Normal")

            # ML only: attack if ml_pred != Normal
            ml_bin.append(1 if m_cls != "Normal" else 0)

            # Hybrid: rules first, then ML
            if is_rule_hit:
                hybrid_stages.append("rule")
                hybrid_bin.append(1)
                hybrid_cls.append("Rule violation")
            else:
                hybrid_stages.append("ml")
                hybrid_bin.append(1 if m_cls != "Normal" else 0)
                hybrid_cls.append(m_cls)

        # Store for global metrics
        y_true_all.extend([cls] * n_frames)
        rules_pred_bin_all.extend(rules_bin)
        rules_pred_cls_all.extend(rules_cls)
        ml_pred_bin_all.extend(ml_bin)
        ml_pred_cls_all.extend(ml_pred_classes)
        hybrid_pred_bin_all.extend(hybrid_bin)
        hybrid_pred_cls_all.extend(hybrid_cls)
        hybrid_stage_all.extend(hybrid_stages)

        # Per-class summary counts
        rule_hits_count = int(np.sum(rules_bin))
        hybrid_rule_decisions = int(sum(1 for s in hybrid_stages if s == "rule"))
        hybrid_ml_decisions = int(sum(1 for s in hybrid_stages if s == "ml"))
        hybrid_attacks_detected = int(sum(hybrid_bin))
        ml_attacks_detected = int(sum(ml_bin))

        class_eval[cls] = {
            "total_frames": n_frames,
            "rules_only": {
                "detected_attacks": rule_hits_count,
                "detected_normal": n_frames - rule_hits_count,
            },
            "ml_only": {
                "detected_attacks": ml_attacks_detected,
                "detected_normal": n_frames - ml_attacks_detected,
                "predicted_as_own_class": int(sum(1 for p in ml_pred_classes if p == cls)),
            },
            "hybrid": {
                "decided_by_rule": hybrid_rule_decisions,
                "decided_by_ml": hybrid_ml_decisions,
                "detected_attacks": hybrid_attacks_detected,
                "detected_normal": n_frames - hybrid_attacks_detected,
            },
        }

        print(
            f"   Processed {cls:<12} ({n_frames:,} frames in {time.time()-t_cls:.2f}s) | "
            f"Rules={rule_hits_count:,} | ML={ml_attacks_detected:,} | Hybrid=(rule:{hybrid_rule_decisions:,}, ml:{hybrid_ml_decisions:,})"
        )

    # -----------------------------------------------------------------------
    # Binary Evaluation (Normal vs Any Attack)
    # -----------------------------------------------------------------------
    y_true_bin_all = [0 if y == "Normal" else 1 for y in y_true_all]

    binary_metrics = {
        "rules_only": compute_binary_metrics(y_true_bin_all, rules_pred_bin_all),
        "ml_only": compute_binary_metrics(y_true_bin_all, ml_pred_bin_all),
        "hybrid": compute_binary_metrics(y_true_bin_all, hybrid_pred_bin_all),
    }

    # -----------------------------------------------------------------------
    # Per-Class Detection Rate (Any Attack)
    # -----------------------------------------------------------------------
    def get_per_class_table(pred_bin: List[int]) -> Dict[str, Dict[str, float]]:
        res = {}
        normal_idx = [i for i, y in enumerate(y_true_all) if y == "Normal"]
        attack_idx = [i for i, y in enumerate(y_true_all) if y != "Normal"]

        # Normal class
        # Positive = Normal (0 in bin), Negative = Attack (1 in bin)
        normal_tp = sum(1 for i in normal_idx if pred_bin[i] == 0)
        normal_fp = sum(1 for i in attack_idx if pred_bin[i] == 0)
        normal_fn = sum(1 for i in normal_idx if pred_bin[i] == 1)

        norm_prec = normal_tp / (normal_tp + normal_fp) if (normal_tp + normal_fp) > 0 else 0.0
        norm_rec = normal_tp / (normal_tp + normal_fn) if (normal_tp + normal_fn) > 0 else 0.0
        norm_f1 = 2 * norm_prec * norm_rec / (norm_prec + norm_rec) if (norm_prec + norm_rec) > 0 else 0.0

        res["Normal"] = {
            "precision": round(norm_prec, 4),
            "recall": round(norm_rec, 4),
            "f1": round(norm_f1, 4),
            "support": len(normal_idx),
        }

        # Normal false alarms
        fp_norm = normal_fn

        for atk in ATTACK_CLASSES:
            atk_idx = [i for i, y in enumerate(y_true_all) if y == atk]
            tp = sum(1 for i in atk_idx if pred_bin[i] == 1)
            fn = sum(1 for i in atk_idx if pred_bin[i] == 0)
            rec = tp / len(atk_idx) if len(atk_idx) > 0 else 0.0
            prec = tp / (tp + fp_norm) if (tp + fp_norm) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            res[atk] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": len(atk_idx),
            }

        return res

    per_class_metrics = {
        "rules_only": get_per_class_table(rules_pred_bin_all),
        "ml_only": get_per_class_table(ml_pred_bin_all),
        "hybrid": get_per_class_table(hybrid_pred_bin_all),
    }

    # -----------------------------------------------------------------------
    # Confusion Matrices (Rows: True Class, Columns: Predicted Outcome)
    # Outcomes: Normal, Rule violation, DoS, Fuzzy, Spoof_Gear, Spoof_RPM
    # -----------------------------------------------------------------------
    cm_rules = compute_confusion_matrix(y_true_all, rules_pred_cls_all)
    cm_ml = compute_confusion_matrix(y_true_all, ml_pred_cls_all)
    cm_hybrid = compute_confusion_matrix(y_true_all, hybrid_pred_cls_all)

    # -----------------------------------------------------------------------
    # ML-Only Fine-Grained 5-Class Multiclass Metrics
    # -----------------------------------------------------------------------
    from sklearn.metrics import classification_report
    ml_multiclass_report = classification_report(
        y_true_all, ml_pred_cls_all, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )

    # -----------------------------------------------------------------------
    # Build Final JSON Summary
    # -----------------------------------------------------------------------
    output_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_test_per_class": MAX_TEST_PER_CLASS,
        "total_test_frames": total_eval_frames,
        "class_breakdown": class_eval,
        "binary_metrics": binary_metrics,
        "per_class_detection_metrics": per_class_metrics,
        "confusion_matrices": {
            "rules_only": cm_rules,
            "ml_only": cm_ml,
            "hybrid": cm_hybrid,
        },
        "ml_fine_grained_multiclass": {
            cls: {
                "precision": round(ml_multiclass_report[cls]["precision"], 4),
                "recall": round(ml_multiclass_report[cls]["recall"], 4),
                "f1": round(ml_multiclass_report[cls]["f1-score"], 4),
                "support": int(ml_multiclass_report[cls]["support"]),
            }
            for cls in CLASS_LABELS
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)
    print(f"\n[Saved] JSON results saved to: {JSON_OUTPUT_PATH}")

    # -----------------------------------------------------------------------
    # Build Markdown Report Table
    # -----------------------------------------------------------------------
    md_content = f"""# CAN Bus IDS — Held-Out Evaluation Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Dataset:** `data/prepared.parquet`  
**Evaluation Protocol:** Chronological 80/20 held-out test block per class (last 20% in timestamp order), contiguous stream processing with isolated state per class.  
**Per-Class Evaluation Cap:** {MAX_TEST_PER_CLASS:,} frames (Total: {total_eval_frames:,} frames across 5 classes).  

---

## 1. Binary Detection Performance (Normal vs Any Attack)

| Configuration | Precision | Recall | F1-Score | False Positive Rate (FPR) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
|---|---|---|---|---|---|---|---|---|
| **Rules Only** | {binary_metrics['rules_only']['precision']*100:.2f}% | {binary_metrics['rules_only']['recall']*100:.2f}% | {binary_metrics['rules_only']['f1']*100:.2f}% | {binary_metrics['rules_only']['fpr']*100:.2f}% | {binary_metrics['rules_only']['tp']:,} | {binary_metrics['rules_only']['fp']:,} | {binary_metrics['rules_only']['fn']:,} | {binary_metrics['rules_only']['tn']:,} |
| **ML Only** | {binary_metrics['ml_only']['precision']*100:.2f}% | {binary_metrics['ml_only']['recall']*100:.2f}% | {binary_metrics['ml_only']['f1']*100:.2f}% | {binary_metrics['ml_only']['fpr']*100:.2f}% | {binary_metrics['ml_only']['tp']:,} | {binary_metrics['ml_only']['fp']:,} | {binary_metrics['ml_only']['fn']:,} | {binary_metrics['ml_only']['tn']:,} |
| **Hybrid (Rules + ML)** | **{binary_metrics['hybrid']['precision']*100:.2f}%** | **{binary_metrics['hybrid']['recall']*100:.2f}%** | **{binary_metrics['hybrid']['f1']*100:.2f}%** | **{binary_metrics['hybrid']['fpr']*100:.2f}%** | **{binary_metrics['hybrid']['tp']:,}** | **{binary_metrics['hybrid']['fp']:,}** | **{binary_metrics['hybrid']['fn']:,}** | **{binary_metrics['hybrid']['tn']:,}** |

---

## 2. Per-Class Detection Rate (Any Attack)

> **Note:** The rule stage does not assign attack types; all rule-stage hits are flagged as generic "Rule violation". Specific attack types (DoS, Fuzzy, Spoof_Gear, Spoof_RPM) are classified exclusively when unflagged frames reach the ML stage. The tables below evaluate detection rate for any attack versus Normal.

### Configuration 1: Rules Only
*Deterministic heuristics (INVALID_ID, FREQUENCY, TIMING) learned strictly from training Normal traffic.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Detected as Attack |
|---|---|---|---|---|---|
| **Normal** | {per_class_metrics['rules_only']['Normal']['support']:,} | {per_class_metrics['rules_only']['Normal']['precision']*100:.2f}% | {per_class_metrics['rules_only']['Normal']['recall']*100:.2f}% | {per_class_metrics['rules_only']['Normal']['f1']*100:.2f}% | {class_eval['Normal']['rules_only']['detected_attacks']:,} (FP) |
| **DoS** | {per_class_metrics['rules_only']['DoS']['support']:,} | {per_class_metrics['rules_only']['DoS']['precision']*100:.2f}% | {per_class_metrics['rules_only']['DoS']['recall']*100:.2f}% | {per_class_metrics['rules_only']['DoS']['f1']*100:.2f}% | {class_eval['DoS']['rules_only']['detected_attacks']:,} |
| **Fuzzy** | {per_class_metrics['rules_only']['Fuzzy']['support']:,} | {per_class_metrics['rules_only']['Fuzzy']['precision']*100:.2f}% | {per_class_metrics['rules_only']['Fuzzy']['recall']*100:.2f}% | {per_class_metrics['rules_only']['Fuzzy']['f1']*100:.2f}% | {class_eval['Fuzzy']['rules_only']['detected_attacks']:,} |
| **Spoof_Gear** | {per_class_metrics['rules_only']['Spoof_Gear']['support']:,} | {per_class_metrics['rules_only']['Spoof_Gear']['precision']*100:.2f}% | {per_class_metrics['rules_only']['Spoof_Gear']['recall']*100:.2f}% | {per_class_metrics['rules_only']['Spoof_Gear']['f1']*100:.2f}% | {class_eval['Spoof_Gear']['rules_only']['detected_attacks']:,} |
| **Spoof_RPM** | {per_class_metrics['rules_only']['Spoof_RPM']['support']:,} | {per_class_metrics['rules_only']['Spoof_RPM']['precision']*100:.2f}% | {per_class_metrics['rules_only']['Spoof_RPM']['recall']*100:.2f}% | {per_class_metrics['rules_only']['Spoof_RPM']['f1']*100:.2f}% | {class_eval['Spoof_RPM']['rules_only']['detected_attacks']:,} |

#### Confusion Matrix (Rules Only)
{format_confusion_matrix_md(cm_rules)}

### Configuration 2: ML Only
*Random Forest classifier evaluated directly on all frames without rule filtering.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Detected as Attack |
|---|---|---|---|---|---|
| **Normal** | {per_class_metrics['ml_only']['Normal']['support']:,} | {per_class_metrics['ml_only']['Normal']['precision']*100:.2f}% | {per_class_metrics['ml_only']['Normal']['recall']*100:.2f}% | {per_class_metrics['ml_only']['Normal']['f1']*100:.2f}% | {class_eval['Normal']['ml_only']['detected_attacks']:,} (FP) |
| **DoS** | {per_class_metrics['ml_only']['DoS']['support']:,} | {per_class_metrics['ml_only']['DoS']['precision']*100:.2f}% | {per_class_metrics['ml_only']['DoS']['recall']*100:.2f}% | {per_class_metrics['ml_only']['DoS']['f1']*100:.2f}% | {class_eval['DoS']['ml_only']['detected_attacks']:,} |
| **Fuzzy** | {per_class_metrics['ml_only']['Fuzzy']['support']:,} | {per_class_metrics['ml_only']['Fuzzy']['precision']*100:.2f}% | {per_class_metrics['ml_only']['Fuzzy']['recall']*100:.2f}% | {per_class_metrics['ml_only']['Fuzzy']['f1']*100:.2f}% | {class_eval['Fuzzy']['ml_only']['detected_attacks']:,} |
| **Spoof_Gear** | {per_class_metrics['ml_only']['Spoof_Gear']['support']:,} | {per_class_metrics['ml_only']['Spoof_Gear']['precision']*100:.2f}% | {per_class_metrics['ml_only']['Spoof_Gear']['recall']*100:.2f}% | {per_class_metrics['ml_only']['Spoof_Gear']['f1']*100:.2f}% | {class_eval['Spoof_Gear']['ml_only']['detected_attacks']:,} |
| **Spoof_RPM** | {per_class_metrics['ml_only']['Spoof_RPM']['support']:,} | {per_class_metrics['ml_only']['Spoof_RPM']['precision']*100:.2f}% | {per_class_metrics['ml_only']['Spoof_RPM']['recall']*100:.2f}% | {per_class_metrics['ml_only']['Spoof_RPM']['f1']*100:.2f}% | {class_eval['Spoof_RPM']['ml_only']['detected_attacks']:,} |

#### Confusion Matrix (ML Only)
{format_confusion_matrix_md(cm_ml)}

### Configuration 3: Hybrid (Rules First, Then ML)
*Deterministic rules catch rapid floods and unknown IDs instantly; unflagged frames are classified by Random Forest.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Decided by Rule | Decided by ML | Total Detected |
|---|---|---|---|---|---|---|---|
| **Normal** | {per_class_metrics['hybrid']['Normal']['support']:,} | {per_class_metrics['hybrid']['Normal']['precision']*100:.2f}% | {per_class_metrics['hybrid']['Normal']['recall']*100:.2f}% | {per_class_metrics['hybrid']['Normal']['f1']*100:.2f}% | {class_eval['Normal']['hybrid']['decided_by_rule']:,} | {class_eval['Normal']['hybrid']['decided_by_ml']:,} | {class_eval['Normal']['hybrid']['detected_normal']:,} (Normal) |
| **DoS** | {per_class_metrics['hybrid']['DoS']['support']:,} | {per_class_metrics['hybrid']['DoS']['precision']*100:.2f}% | {per_class_metrics['hybrid']['DoS']['recall']*100:.2f}% | {per_class_metrics['hybrid']['DoS']['f1']*100:.2f}% | {class_eval['DoS']['hybrid']['decided_by_rule']:,} | {class_eval['DoS']['hybrid']['decided_by_ml']:,} | {class_eval['DoS']['hybrid']['detected_attacks']:,} (Attack) |
| **Fuzzy** | {per_class_metrics['hybrid']['Fuzzy']['support']:,} | {per_class_metrics['hybrid']['Fuzzy']['precision']*100:.2f}% | {per_class_metrics['hybrid']['Fuzzy']['recall']*100:.2f}% | {per_class_metrics['hybrid']['Fuzzy']['f1']*100:.2f}% | {class_eval['Fuzzy']['hybrid']['decided_by_rule']:,} | {class_eval['Fuzzy']['hybrid']['decided_by_ml']:,} | {class_eval['Fuzzy']['hybrid']['detected_attacks']:,} (Attack) |
| **Spoof_Gear** | {per_class_metrics['hybrid']['Spoof_Gear']['support']:,} | {per_class_metrics['hybrid']['Spoof_Gear']['precision']*100:.2f}% | {per_class_metrics['hybrid']['Spoof_Gear']['recall']*100:.2f}% | {per_class_metrics['hybrid']['Spoof_Gear']['f1']*100:.2f}% | {class_eval['Spoof_Gear']['hybrid']['decided_by_rule']:,} | {class_eval['Spoof_Gear']['hybrid']['decided_by_ml']:,} | {class_eval['Spoof_Gear']['hybrid']['detected_attacks']:,} (Attack) |
| **Spoof_RPM** | {per_class_metrics['hybrid']['Spoof_RPM']['support']:,} | {per_class_metrics['hybrid']['Spoof_RPM']['precision']*100:.2f}% | {per_class_metrics['hybrid']['Spoof_RPM']['recall']*100:.2f}% | {per_class_metrics['hybrid']['Spoof_RPM']['f1']*100:.2f}% | {class_eval['Spoof_RPM']['hybrid']['decided_by_rule']:,} | {class_eval['Spoof_RPM']['hybrid']['decided_by_ml']:,} | {class_eval['Spoof_RPM']['hybrid']['detected_attacks']:,} (Attack) |

#### Confusion Matrix (Hybrid)
{format_confusion_matrix_md(cm_hybrid)}

---

## 3. Hybrid Decision Stage Distribution

| Class | Support | Decided by Rules (%) | Decided by ML (%) | Detection Status |
|---|---|---|---|---|
| **Normal** | {class_eval['Normal']['total_frames']:,} | {class_eval['Normal']['hybrid']['decided_by_rule'] / class_eval['Normal']['total_frames'] * 100:.2f}% | {class_eval['Normal']['hybrid']['decided_by_ml'] / class_eval['Normal']['total_frames'] * 100:.2f}% | {class_eval['Normal']['hybrid']['detected_normal'] / class_eval['Normal']['total_frames'] * 100:.2f}% Clean Normal |
| **DoS** | {class_eval['DoS']['total_frames']:,} | {class_eval['DoS']['hybrid']['decided_by_rule'] / class_eval['DoS']['total_frames'] * 100:.2f}% | {class_eval['DoS']['hybrid']['decided_by_ml'] / class_eval['DoS']['total_frames'] * 100:.2f}% | {class_eval['DoS']['hybrid']['detected_attacks'] / class_eval['DoS']['total_frames'] * 100:.2f}% Caught |
| **Fuzzy** | {class_eval['Fuzzy']['total_frames']:,} | {class_eval['Fuzzy']['hybrid']['decided_by_rule'] / class_eval['Fuzzy']['total_frames'] * 100:.2f}% | {class_eval['Fuzzy']['hybrid']['decided_by_ml'] / class_eval['Fuzzy']['total_frames'] * 100:.2f}% | {class_eval['Fuzzy']['hybrid']['detected_attacks'] / class_eval['Fuzzy']['total_frames'] * 100:.2f}% Caught |
| **Spoof_Gear** | {class_eval['Spoof_Gear']['total_frames']:,} | {class_eval['Spoof_Gear']['hybrid']['decided_by_rule'] / class_eval['Spoof_Gear']['total_frames'] * 100:.2f}% | {class_eval['Spoof_Gear']['hybrid']['decided_by_ml'] / class_eval['Spoof_Gear']['total_frames'] * 100:.2f}% | {class_eval['Spoof_Gear']['hybrid']['detected_attacks'] / class_eval['Spoof_Gear']['total_frames'] * 100:.2f}% Caught |
| **Spoof_RPM** | {class_eval['Spoof_RPM']['total_frames']:,} | {class_eval['Spoof_RPM']['hybrid']['decided_by_rule'] / class_eval['Spoof_RPM']['total_frames'] * 100:.2f}% | {class_eval['Spoof_RPM']['hybrid']['decided_by_ml'] / class_eval['Spoof_RPM']['total_frames'] * 100:.2f}% | {class_eval['Spoof_RPM']['hybrid']['detected_attacks'] / class_eval['Spoof_RPM']['total_frames'] * 100:.2f}% Caught |
"""

    with MD_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown report table saved to: {MD_OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(md_content)

    return output_summary


if __name__ == "__main__":
    evaluate()
