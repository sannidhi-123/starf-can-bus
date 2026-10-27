"""Part C verification script:
Starts the FastAPI backend, sends 10 consecutive frames from Fuzzy, Spoof_Gear, and Spoof_RPM
from the held-out test block of data/prepared.parquet to POST /predict.
For each class, shows ONE full JSON response of an ML-stage attack with top_features.
If none of the first 10 reaches ML stage, continues up to 200 consecutive frames.
Then cleanly stops the server.
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
import pandas as pd

SERVER_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{SERVER_URL}/predict"

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

def wait_for_server(timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

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
    print("[Part C] Starting backend server with .\\venv\\Scripts\\python.exe...")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        ready = wait_for_server(30.0)
        if not ready:
            print("[ERROR] Server failed to start within 30 seconds.")
            return

        print("[Part C] Server is online at", SERVER_URL)

        df = pd.read_parquet("data/prepared.parquet")
        target_classes = ["Fuzzy", "Spoof_Gear", "Spoof_RPM"]
        ml_alert_responses = {}

        for cls in target_classes:
            print("\n" + "=" * 70)
            print(f"  PROCESSING CLASS: {cls}")
            print("=" * 70)
            
            cls_df = df[df["label"] == cls].sort_values("timestamp").reset_index(drop=True)
            split_idx = int(len(cls_df) * 0.8)
            test_block = cls_df.iloc[split_idx:split_idx+200].reset_index(drop=True)
            
            print(f"[{cls}] Sending first 10 consecutive frames from held-out test block...")
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

            stages_in_10 = [r.get("stage") for r in first_10_responses]
            print(f"[{cls}] Stages for first 10 frames: {stages_in_10}")

            if ml_hit is not None:
                print(f"[{cls}] Found ML-stage attack within first 10 frames at index {ml_hit_idx}!")
                ml_alert_responses[cls] = (ml_hit_idx, ml_hit)
            else:
                print(f"[{cls}] Notice: None of the first 10 frames reached the ML stage (all caught by rules).")
                print(f"[{cls}] Searching up to 200 consecutive frames for the first stage-'ml' attack...")
                
                for i in range(10, len(test_block)):
                    payload = frame_to_dict(test_block.iloc[i])
                    res = call_predict(payload)
                    if res.get("stage") == "ml" and res.get("is_attack"):
                        ml_hit = res
                        ml_hit_idx = i
                        print(f"[{cls}] First ML-stage attack found at frame index {ml_hit_idx} (timestamp: {payload['timestamp']}, can_id: 0x{payload['can_id']:03X})!")
                        break
                
                if ml_hit is not None:
                    ml_alert_responses[cls] = (ml_hit_idx, ml_hit)
                else:
                    print(f"[{cls}] Warning: No ML-stage attack found in the first 200 frames.")

        print("\n" + "#" * 70)
        print("  PART C RESULTS — FULL JSON RESPONSES OF ML-STAGE ATTACKS")
        print("#" * 70)

        for cls in target_classes:
            if cls in ml_alert_responses:
                idx, alert_json = ml_alert_responses[cls]
                print(f"\n======================================================================")
                print(f"  {cls.upper()} — ML-STAGE ATTACK (Frame Index: {idx})")
                print(f"======================================================================")
                print(json.dumps(alert_json, indent=2))

    finally:
        print("\n[Part C] Stopping backend server...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[Part C] Backend server stopped.")

if __name__ == "__main__":
    main()
