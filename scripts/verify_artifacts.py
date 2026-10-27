import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.preprocess import build_frame_features

# 1. Load artifacts from models/
model_path = Path("models/multiclass_model.pkl")
features_path = Path("models/multiclass_features.json")
encoder_path = Path("models/multiclass_label_encoder.pkl")
meta_path = Path("models/multiclass_meta.json")

print("Loading saved artifacts from models/...")
model = joblib.load(model_path)
with open(features_path, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)
label_encoder = joblib.load(encoder_path)
with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"  [OK] Model loaded: {type(model).__name__} (n_estimators={model.n_estimators})")
print(f"  [OK] Features loaded ({len(feature_cols)}): {feature_cols}")
print(f"  [OK] Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
print(f"  [OK] Metadata date: {metadata['date']}")

# 2. Extract 5 sample rows from the held-out test split (one from each of the 5 classes)
df = pd.read_parquet("data/prepared.parquet")
classes = ["Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]
sample_dfs = []

for cls in classes:
    cls_df = df[df["label"] == cls].sort_values("timestamp").reset_index(drop=True)
    test_start = int(len(cls_df) * 0.8)
    # Pick a sample from the held-out test split
    sample_dfs.append(cls_df.iloc[[test_start + 1000]])

sample_df = pd.concat(sample_dfs, ignore_index=True)

# 3. Extract features using build_frame_features
sample_feats = build_frame_features(sample_df)
X_sample = sample_feats[feature_cols].to_numpy(dtype=np.float32)
true_labels = sample_df["label"].values

# 4. Predict
preds_encoded = model.predict(X_sample)
pred_labels = label_encoder.inverse_transform(preds_encoded)
probs = model.predict_proba(X_sample)

print("\n" + "=" * 72)
print("  PREDICTION PROOF ON 5 HELD-OUT TEST SPLIT SAMPLES")
print("=" * 72)
header = f"  {'CAN ID':<10} {'True Label':<14} {'Predicted':<14} {'Confidence':>12} {'Result'}"
print(header)
print("  " + "-" * 66)
for i in range(len(sample_df)):
    cid_hex = f"0x{int(sample_df.iloc[i]['can_id']):04x}"
    true_l = true_labels[i]
    pred_l = pred_labels[i]
    conf = float(np.max(probs[i])) * 100
    match = "MATCH" if true_l == pred_l else "MISMATCH"
    print(f"  {cid_hex:<10} {true_l:<14} {pred_l:<14} {conf:>11.1f}%   [{match}]")
print("=" * 72 + "\n")
