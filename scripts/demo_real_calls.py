"""Part E verification script:
Starts the FastAPI backend with uvicorn in a subprocess,
calls POST /predict with 10 consecutive DoS frames and 10 consecutive Normal frames,
prints the exact JSON responses, and cleanly shuts down the server.
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
    print("[Part E] Starting backend server with .\\venv\\Scripts\\python.exe...")
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

        print("[Part E] Server is online at", SERVER_URL)

        # Load real dataset
        print("\n[Part E] Loading 10 consecutive frames from DoS and 10 from Normal...")
        df = pd.read_parquet("data/prepared.parquet")
        dos_df = df[df["label"] == "DoS"].head(10).reset_index(drop=True)
        normal_df = df[df["label"] == "Normal"].head(10).reset_index(drop=True)

        print("\n" + "=" * 70)
        print("  10 CONSECUTIVE DoS FRAMES -> POST /predict")
        print("=" * 70)
        for i in range(len(dos_df)):
            payload = frame_to_dict(dos_df.iloc[i])
            res = call_predict(payload)
            print(f"\n--- [DoS Frame {i+1}] (CAN ID: 0x{payload['can_id']:03X}, ts: {payload['timestamp']}) ---")
            print(json.dumps(res, indent=2))

        print("\n" + "=" * 70)
        print("  10 CONSECUTIVE NORMAL FRAMES -> POST /predict")
        print("=" * 70)
        for i in range(len(normal_df)):
            payload = frame_to_dict(normal_df.iloc[i])
            res = call_predict(payload)
            print(f"\n--- [Normal Frame {i+1}] (CAN ID: 0x{payload['can_id']:03X}, ts: {payload['timestamp']}) ---")
            print(json.dumps(res, indent=2))

    finally:
        print("\n[Part E] Stopping backend server...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[Part E] Backend server stopped.")

if __name__ == "__main__":
    main()
