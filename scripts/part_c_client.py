"""Client for Part C:
Sends 10 consecutive frames from the held-out test block of Fuzzy, Spoof_Gear, and Spoof_RPM
to http://127.0.0.1:8000/predict.
For each class, finds and displays ONE full JSON response of an ML-stage attack (with top_features).
"""

import json
import urllib.request
import urllib.error
import pandas as pd

PREDICT_URL = "http://127.0.0.1:8000/predict"

def frame_to_dict(row):
    data_bytes = []
    for i in range(8):
        val = row[f"d{i}"]
        if pd.isna(val) or val == "":
            data_bytes.append(0)
        else:
            try:
                data_bytes.append(int(str(val), 16))
            except ValueError:
                data_bytes.append(0)
    return {
        "timestamp": float(row["timestamp"]),
        "can_id": int(row["can_id"]),
        "dlc": int(row["dlc"]),
        "data_bytes": data_bytes,
        "simulated": False,
    }

def call_predict(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        PREDICT_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 75)
    print("  PART C: ML-STAGE ALERT DEMO (Real Server Calls -> POST /predict)")
    print("=" * 75)

    df = pd.read_parquet("data/prepared.parquet")
    target_classes = ["Fuzzy", "Spoof_Gear", "Spoof_RPM"]
    ml_alert_results = {}

    for cls in target_classes:
        print(f"\n--- Testing Class: {cls} ---")
        cls_df = df[df["label"] == cls].sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(cls_df) * 0.8)
        test_block = cls_df.iloc[split_idx:split_idx+200].reset_index(drop=True)

        print(f"[{cls}] Sending 10 consecutive frames from held-out test block...")
        first_10_responses = []
        ml_hit = None
        ml_hit_idx = None

        for i in range(10):
            payload = frame_to_dict(test_block.iloc[i])
            res = call_predict(payload)
            first_10_responses.append(res)
            if ml_hit is None and res.get("stage") == "ml" and res.get("is_attack"):
                ml_hit = res
                ml_hit_idx = i

        stages = [r.get("stage") for r in first_10_responses]
        print(f"[{cls}] Stages for the 10 frames: {stages}")

        if ml_hit is not None:
            print(f"[{cls}] ML-stage alert found within the first 10 frames at index {ml_hit_idx}!")
            ml_alert_results[cls] = (ml_hit_idx, ml_hit)
        else:
            print(f"[{cls}] None of the first 10 frames reached the ML stage (all 10 triggered rule-stage detection).")
            print(f"[{cls}] Searching within consecutive frames up to 200 for the first stage-'ml' attack alert...")
            for i in range(10, min(200, len(test_block))):
                payload = frame_to_dict(test_block.iloc[i])
                res = call_predict(payload)
                if res.get("stage") == "ml" and res.get("is_attack"):
                    ml_hit = res
                    ml_hit_idx = i
                    print(f"[{cls}] First ML-stage attack alert found at frame index {ml_hit_idx} (timestamp: {payload['timestamp']}, can_id: 0x{payload['can_id']:03X})!")
                    break

            if ml_hit is not None:
                ml_alert_results[cls] = (ml_hit_idx, ml_hit)
            else:
                print(f"[{cls}] No ML-stage alert found in the first 200 frames.")

    print("\n" + "#" * 75)
    print("  ONE FULL JSON RESPONSE OF AN ML-STAGE ATTACK PER CLASS")
    print("#" * 75)

    for cls in target_classes:
        if cls in ml_alert_results:
            idx, res_json = ml_alert_results[cls]
            print(f"\n===========================================================================")
            print(f"  CLASS: {cls} | STAGE: ml | ATTACK TYPE: {res_json.get('attack_type')} | FRAME INDEX: {idx}")
            print(f"===========================================================================")
            print(json.dumps(res_json, indent=2))

if __name__ == "__main__":
    main()
