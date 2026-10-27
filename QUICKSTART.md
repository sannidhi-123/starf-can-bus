# 🚀 STARF CAN Bus IDS — Quickstart Guide
*(Smart Traffic Anomaly Recognition Kit)*

This guide describes how to set up, prepare data, train models, fit detection rules, evaluate, and run the STARF Hybrid Intrusion Detection System from a clean checkout.

---

## ⚠️ Important System Notes & Scope

1. **Dataset Scope:** Only the **Car Hacking Dataset** (Normal, DoS, Fuzzy, Gear spoofing, RPM spoofing) is currently implemented and supported.
2. **Future Datasets:** **OTIDS** and **impersonation attacks** are **not included yet**.
3. **Hybrid Architecture vs. Legacy Fallback:** The **Hybrid Pipeline** (Deterministic Rules first, Random Forest multiclass ML second) is the primary detection pipeline powering both REST and WebSocket endpoints. The DistilBERT sequence model is retained strictly as an offline fallback when hybrid artifacts are unavailable.
4. **Simulated Traffic Tag:** The dashboard can stream real-time demo traffic generated from background traces. Any simulated traffic is explicitly marked with the `SIMULATED` badge in the threat log and API responses (`"simulated": true`).
5. **Evaluation Distribution:** The evaluation test split uses a **balanced chronological held-out block** (20,000 contiguous frames per class, total 100,000 frames) to measure per-class recall and precision fairly; this split is balanced by design and does **not** reflect real-world attack frequency distribution (where normal traffic accounts for >99% of total frames).

---

## 📋 Prerequisites

- **Python 3.10+** (tested on Python 3.13)
- **Node.js 18+** & **npm**
- **Git**

---

## 🛠️ Step-by-Step Setup

### Step 1: Clone & Setup Virtual Environment

```bash
git clone <repo-url>
cd can_bus_ids_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

---

### Step 2: Car Hacking Dataset Placement

Extract the raw Car Hacking dataset text files into the `data/car_hacking/` directory:

```
data/
└── car_hacking/
    ├── normal_run_data.txt      (or normal_traffic_augmented.csv)
    ├── DoS_dataset.csv
    ├── Fuzzy_dataset.csv
    ├── gear_dataset.csv
    └── RPM_dataset.csv
```

> **Note:** Do not rename or delete anything in `data/car_hacking/` or `data/raw_downloads/`. Outputs will be written directly into `data/`.

---

### Step 3: Dataset Preparation

Parse, align timestamps, extract arbitration IDs and payload bytes, and build the canonical dataset:

```bash
python src/dataset_prepare.py
```

This produces `data/prepared.parquet` with the unified schema:
`timestamp`, `can_id` (int), `dlc`, `d0..d7`, `label` (`Normal`, `DoS`, `Fuzzy`, `Spoof_Gear`, `Spoof_RPM`), and `source_file`.

---

### Step 4: Fit Deterministic Rule Thresholds

Learn the normal CAN ID whitelist, frequency limits, and minimum inter-arrival times strictly from the training portion of Normal traffic:

```bash
python src/rules.py
```

This saves the fitted configuration to `models/rules_config.json`.

---

### Step 5: Train Multiclass ML Classifier

Train the Random Forest classifier with SMOTE applied strictly to the training split:

```bash
python -m src.train_multiclass
```

Outputs saved:
- `models/multiclass_model.pkl` (trained Random Forest)
- `models/multiclass_features.json` (canonical feature order)
- `models/multiclass_label_encoder.pkl` (label encoder)
- `models/multiclass_meta.json` (training metadata & leakage audit)

---

### Step 6: Run Test Suite

Run the full automated test suite (rules, preprocessing, stream state, parity, SHAP, and accuracy):

```bash
pytest tests/ -v
```

All tests should pass.

---

### Step 7: Run Held-Out Evaluation

Run the held-out evaluation across all 3 configurations (Rules Only, ML Only, Hybrid) on the chronological 80/20 test split:

```bash
python src/evaluate.py
```

Outputs generated:
- `reports/hybrid_evaluation.json`
- `reports/hybrid_evaluation.md`

---

### Step 8: Start Backend Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Verify backend health at: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or explore interactive Swagger docs at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### Step 9: Start Frontend Dashboard

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Open your browser to: **http://localhost:5173** (or http://localhost:3000).

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | `POST` | REST prediction endpoint. Accepts raw CAN frame or legacy sequence. |
| `/predict/frame` | `POST` | Dedicated single-frame prediction endpoint. |
| `/ws/live` | `WebSocket` | Live streaming WebSocket emitting real-time telemetry and threat logs. |
| `/health` | `GET` | Health status and asset loading check. |
| `/stats` | `GET` | Aggregated detection statistics (total analyzed, attacks, breakdown). |
| `/reports/manifest` | `GET` | Manifest of available evaluation charts, PDFs, and metrics. |
