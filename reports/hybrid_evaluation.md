# CAN Bus IDS — Held-Out Evaluation Report

**Evaluation Date:** 2026-10-28 18:01:17 UTC  
**Dataset:** `data/prepared.parquet`  
**Evaluation Protocol:** Chronological 80/20 held-out test block per class (last 20% in timestamp order), contiguous stream processing with isolated state per class.  
**Per-Class Evaluation Cap:** 20,000 frames (Total: 100,000 frames across 5 classes).  

---

## 1. Binary Detection Performance (Normal vs Any Attack)

| Configuration | Precision | Recall | F1-Score | False Positive Rate (FPR) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
|---|---|---|---|---|---|---|---|---|
| **Rules Only** | 100.00% | 99.49% | 99.75% | 0.00% | 79,595 | 0 | 405 | 20,000 |
| **ML Only** | 99.68% | 100.00% | 99.84% | 1.29% | 79,999 | 259 | 1 | 19,741 |
| **Hybrid (Rules + ML)** | **99.68%** | **100.00%** | **99.84%** | **1.29%** | **80,000** | **259** | **0** | **19,741** |

---

## 2. Per-Class Detection Rate (Any Attack)

> **Note:** The rule stage does not assign attack types; all rule-stage hits are flagged as generic "Rule violation". Specific attack types (DoS, Fuzzy, Spoof_Gear, Spoof_RPM) are classified exclusively when unflagged frames reach the ML stage. The tables below evaluate detection rate for any attack versus Normal.

### Configuration 1: Rules Only
*Deterministic heuristics (INVALID_ID, FREQUENCY, TIMING) learned strictly from training Normal traffic.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Detected as Attack |
|---|---|---|---|---|---|
| **Normal** | 20,000 | 98.02% | 100.00% | 99.00% | 0 (FP) |
| **DoS** | 20,000 | 100.00% | 100.00% | 100.00% | 20,000 |
| **Fuzzy** | 20,000 | 100.00% | 98.59% | 99.29% | 19,719 |
| **Spoof_Gear** | 20,000 | 100.00% | 99.60% | 99.80% | 19,920 |
| **Spoof_RPM** | 20,000 | 100.00% | 99.78% | 99.89% | 19,956 |

#### Confusion Matrix (Rules Only)
| True Class | Normal | Rule violation | DoS | Fuzzy | Spoof_Gear | Spoof_RPM | Total |
|:---| ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| **Normal** | 20,000 | 0 | 0 | 0 | 0 | 0 | 20,000 |
| **DoS** | 0 | 20,000 | 0 | 0 | 0 | 0 | 20,000 |
| **Fuzzy** | 281 | 19,719 | 0 | 0 | 0 | 0 | 20,000 |
| **Spoof_Gear** | 80 | 19,920 | 0 | 0 | 0 | 0 | 20,000 |
| **Spoof_RPM** | 44 | 19,956 | 0 | 0 | 0 | 0 | 20,000 |

### Configuration 2: ML Only
*Random Forest classifier evaluated directly on all frames without rule filtering.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Detected as Attack |
|---|---|---|---|---|---|
| **Normal** | 20,000 | 99.99% | 98.70% | 99.35% | 259 (FP) |
| **DoS** | 20,000 | 98.72% | 100.00% | 99.36% | 20,000 |
| **Fuzzy** | 20,000 | 98.72% | 100.00% | 99.35% | 19,999 |
| **Spoof_Gear** | 20,000 | 98.72% | 100.00% | 99.36% | 20,000 |
| **Spoof_RPM** | 20,000 | 98.72% | 100.00% | 99.36% | 20,000 |

#### Confusion Matrix (ML Only)
| True Class | Normal | Rule violation | DoS | Fuzzy | Spoof_Gear | Spoof_RPM | Total |
|:---| ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| **Normal** | 19,741 | 0 | 0 | 259 | 0 | 0 | 20,000 |
| **DoS** | 0 | 0 | 20,000 | 0 | 0 | 0 | 20,000 |
| **Fuzzy** | 1 | 0 | 0 | 19,999 | 0 | 0 | 20,000 |
| **Spoof_Gear** | 0 | 0 | 0 | 0 | 20,000 | 0 | 20,000 |
| **Spoof_RPM** | 0 | 0 | 0 | 0 | 0 | 20,000 | 20,000 |

### Configuration 3: Hybrid (Rules First, Then ML)
*Deterministic rules catch rapid floods and unknown IDs instantly; unflagged frames are classified by Random Forest.*

#### Per-class detection rate (any attack)
| Class | Support | Precision | Recall | F1-Score | Decided by Rule | Decided by ML | Total Detected |
|---|---|---|---|---|---|---|---|
| **Normal** | 20,000 | 100.00% | 98.70% | 99.35% | 0 | 20,000 | 19,741 (Normal) |
| **DoS** | 20,000 | 98.72% | 100.00% | 99.36% | 20,000 | 0 | 20,000 (Attack) |
| **Fuzzy** | 20,000 | 98.72% | 100.00% | 99.36% | 19,719 | 281 | 20,000 (Attack) |
| **Spoof_Gear** | 20,000 | 98.72% | 100.00% | 99.36% | 19,920 | 80 | 20,000 (Attack) |
| **Spoof_RPM** | 20,000 | 98.72% | 100.00% | 99.36% | 19,956 | 44 | 20,000 (Attack) |

#### Confusion Matrix (Hybrid)
| True Class | Normal | Rule violation | DoS | Fuzzy | Spoof_Gear | Spoof_RPM | Total |
|:---| ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| **Normal** | 19,741 | 0 | 0 | 259 | 0 | 0 | 20,000 |
| **DoS** | 0 | 0 | 20,000 | 0 | 0 | 0 | 20,000 |
| **Fuzzy** | 0 | 1 | 0 | 19,999 | 0 | 0 | 20,000 |
| **Spoof_Gear** | 0 | 0 | 0 | 0 | 20,000 | 0 | 20,000 |
| **Spoof_RPM** | 0 | 0 | 0 | 0 | 0 | 20,000 | 20,000 |

---

## 3. Hybrid Decision Stage Distribution

| Class | Support | Decided by Rules (%) | Decided by ML (%) | Detection Status |
|---|---|---|---|---|
| **Normal** | 20,000 | 0.00% | 100.00% | 98.70% Clean Normal |
| **DoS** | 20,000 | 100.00% | 0.00% | 100.00% Caught |
| **Fuzzy** | 20,000 | 98.59% | 1.41% | 100.00% Caught |
| **Spoof_Gear** | 20,000 | 99.60% | 0.40% | 100.00% Caught |
| **Spoof_RPM** | 20,000 | 99.78% | 0.22% | 100.00% Caught |
