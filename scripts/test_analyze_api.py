import urllib.request
import json

def post(payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:8000/predict', data=data, headers={'Content-Type': 'application/json'})
    try:
        resp = urllib.request.urlopen(req)
        return resp.getcode(), json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

print("--- Test 1: Normal traffic ['0x18F'] ---")
status, res = post({"sequence": ["0x18F"]})
print("Status:", status)
print("Decision:", res.get("decision"), "| Label:", res.get("label"), "| Stage:", res.get("stage"), "| Attack Type:", res.get("attack_type"), "| Confidence:", res.get("confidence"), "| Severity:", res.get("severity"))

print("\n--- Test 2: Repeated 0x000 ['0x000'] * 10 ---")
status, res = post({"sequence": ["0x000"] * 10})
print("Status:", status)
print("Decision:", res.get("decision"), "| Label:", res.get("label"), "| Stage:", res.get("stage"), "| Rule ID:", res.get("rule_id"), "| Reason:", res.get("reason"), "| Severity:", res.get("severity"))

print("\n--- Test 3: Unknown CAN ID ['0x7FFF'] ---")
status, res = post({"sequence": ["0x7FFF"]})
print("Status:", status)
print("Decision:", res.get("decision"), "| Label:", res.get("label"), "| Stage:", res.get("stage"), "| Rule ID:", res.get("rule_id"), "| Reason:", res.get("reason"), "| Severity:", res.get("severity"))

print("\n--- Test 4: Genuinely invalid input ['xyz'] ---")
status, res = post({"sequence": ["xyz"]})
print("Status:", status)
print("Validation error detail:", res.get("detail"))

print("\n--- Test 5: Empty sequence [] ---")
status, res = post({"sequence": []})
print("Status:", status)
print("Validation error detail:", res.get("detail"))
