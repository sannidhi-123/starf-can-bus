"""Tests for src/pipeline.py — HybridPipeline.

Run with:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_pipeline.py -v -s

Test coverage
-------------
1. test_unknown_id_triggers_rule          : Unknown CAN ID fires INVALID_ID rule.
2. test_normal_frame_goes_to_ml           : Whitelisted first-seen frame → ml path.
3. test_dos_flood_triggers_frequency_rule : Flood triggers FREQUENCY rule.
4. test_rest_ws_parity_50_frames          : 50 consecutive frames through pipeline;
                                             decision/stage/attack_type/rule_id/confidence
                                             are identical for the same input.
5. test_real_dataset_per_class_accuracy   : 200 consecutive frames per class from
                                             prepared.parquet; each class ≥ 80% acc.
6. test_shap_top_features_ml_attack       : ML-attack result has exactly 5 top_features
                                             from the saved feature list.
7. test_stream_state_isolation            : Two StreamStates don't share windows.
8. test_backward_time_reset               : Backward timestamp resets state cleanly.
9. test_unified_schema_completeness       : Every response has all required keys.
10. test_simulated_flag                   : simulated=True propagates to response.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — artifact paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
MODELS_DIR       = PROJECT_ROOT / "models"
DATA_DIR         = PROJECT_ROOT / "data"
PREPARED_PARQUET = DATA_DIR / "prepared.parquet"

REQUIRED_ARTIFACTS = [
    MODELS_DIR / "rules_config.json",
    MODELS_DIR / "multiclass_model.pkl",
    MODELS_DIR / "multiclass_features.json",
    MODELS_DIR / "multiclass_label_encoder.pkl",
]

_ARTIFACTS_PRESENT = all(p.exists() for p in REQUIRED_ARTIFACTS)
_PARQUET_PRESENT   = PREPARED_PARQUET.exists()

skip_no_artifacts = pytest.mark.skipif(
    not _ARTIFACTS_PRESENT,
    reason="Model artifacts missing — run train_multiclass.py and rules.py first.",
)
skip_no_parquet = pytest.mark.skipif(
    not _PARQUET_PRESENT,
    reason="data/prepared.parquet missing — run dataset_prepare.py first.",
)

REQUIRED_KEYS = {
    "anomaly_score", "threshold", "label", "attack_type", "confidence",
    "is_attack", "processing_time_ms", "rules_triggered", "pipeline_path",
    "decision", "stage", "rule_id", "reason", "severity", "simulated", "top_features",
}


@pytest.fixture(scope="session")
def pipeline():
    if not _ARTIFACTS_PRESENT:
        pytest.skip("Model artifacts missing.")
    from src.pipeline import HybridPipeline
    return HybridPipeline()


@pytest.fixture(scope="session")
def feature_names():
    path = MODELS_DIR / "multiclass_features.json"
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Unknown ID → rule path
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_unknown_id_triggers_rule(pipeline):
    from src.pipeline import StreamState
    state  = pipeline.new_stream_state()
    engine = pipeline.new_rule_state()

    UNKNOWN_ID = 0x7FFF
    if UNKNOWN_ID in engine.whitelist:
        pytest.skip(f"0x{UNKNOWN_ID:04x} is in the whitelist.")

    result = pipeline.predict_frame(
        timestamp=1000.0, can_id=UNKNOWN_ID, dlc=8, data_bytes=[0]*8,
        state=state, rule_engine=engine,
    )
    assert result["stage"] == "rule"
    assert result["decision"] == "attack"
    assert result["is_attack"] is True
    assert result["attack_type"] == "Rule violation"
    assert result["rule_id"] == "INVALID_ID"
    assert len(result["rules_triggered"]) > 0
    triggered_ids = {r["rule_id"] for r in result["rules_triggered"]}
    assert "INVALID_ID" in triggered_ids


# ---------------------------------------------------------------------------
# 2. Normal frame → ml path
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_normal_frame_goes_to_ml(pipeline):
    from src.pipeline import StreamState
    state  = pipeline.new_stream_state()
    engine = pipeline.new_rule_state()

    assert len(engine.whitelist) > 0
    wl_id = next(iter(sorted(engine.whitelist)))

    result = pipeline.predict_frame(
        timestamp=2000.0, can_id=wl_id, dlc=8, data_bytes=[0]*8,
        state=state, rule_engine=engine,
    )
    assert result["stage"] == "ml", (
        f"Whitelisted ID on first hit should be ml; rules: {result['rules_triggered']}"
    )
    assert result["label"] in {"ATTACK", "NORMAL"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["processing_time_ms"] > 0


# ---------------------------------------------------------------------------
# 3. DoS flood → FREQUENCY rule
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_dos_flood_triggers_frequency_rule(pipeline):
    from src.pipeline import StreamState
    state  = pipeline.new_stream_state()
    engine = pipeline.new_rule_state()

    wl_id = next(iter(sorted(engine.whitelist)))
    limit = engine.freq_thresholds.get(wl_id, engine.default_freq_threshold)
    flood_count = limit + 50

    triggered_freq = False
    for i in range(flood_count):
        ts = 1.0 + i * 0.0001
        result = pipeline.predict_frame(
            timestamp=ts, can_id=wl_id, dlc=8, data_bytes=[0]*8,
            state=state, rule_engine=engine,
        )
        if result["stage"] == "rule":
            rule_ids = {r["rule_id"] for r in result["rules_triggered"]}
            if "FREQUENCY" in rule_ids:
                triggered_freq = True
                break

    assert triggered_freq, f"FREQUENCY should trigger after {flood_count} frames (limit={limit})"


# ---------------------------------------------------------------------------
# 4. REST / WebSocket parity — 50 consecutive frames
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_rest_ws_parity_50_frames(pipeline):
    """Send identical 50 frames through two independent pipeline calls.

    Both calls must use the SAME CAN IDs and timestamps so we can assert
    that decision, stage, attack_type, rule_id, confidence are identical.
    We run them sequentially with two independently reset StreamStates and
    RuleEngines to simulate a REST call vs a WS connection that receives the
    same stream.
    """
    from src.pipeline import StreamState

    assert len(pipeline.whitelist) > 0
    wl_ids = sorted(pipeline.whitelist)[:5]

    # Build deterministic frame stream: 50 frames cycling through whitelist IDs
    frames = []
    base_ts = 5000.0
    for i in range(50):
        cid = wl_ids[i % len(wl_ids)]
        ts  = base_ts + i * 0.01
        frames.append((ts, cid))

    # REST path: fresh isolated state
    rest_state  = pipeline.new_stream_state()
    rest_engine = pipeline.new_rule_state()

    # WS path: another fresh isolated state (same config, clean windows)
    ws_state    = pipeline.new_stream_state()
    ws_engine   = pipeline.new_rule_state()

    PARITY_KEYS = ["decision", "stage", "attack_type", "rule_id"]

    for i, (ts, cid) in enumerate(frames):
        r_rest = pipeline.predict_frame(
            timestamp=ts, can_id=cid, dlc=8, data_bytes=[0]*8,
            state=rest_state, rule_engine=rest_engine,
        )
        r_ws = pipeline.predict_frame(
            timestamp=ts, can_id=cid, dlc=8, data_bytes=[0]*8,
            state=ws_state, rule_engine=ws_engine,
        )

        for key in PARITY_KEYS:
            assert r_rest[key] == r_ws[key], (
                f"Frame {i}: REST vs WS mismatch on '{key}': "
                f"REST={r_rest[key]!r}  WS={r_ws[key]!r}"
            )

        # Confidence should also be identical (same deterministic model)
        assert abs(r_rest["confidence"] - r_ws["confidence"]) < 1e-6, (
            f"Frame {i}: confidence mismatch REST={r_rest['confidence']} WS={r_ws['confidence']}"
        )

    print("\n[test] REST/WS parity: all 50 frames matched on decision/stage/attack_type/rule_id/confidence")


# ---------------------------------------------------------------------------
# 5. Per-class accuracy on 200 consecutive frames each
# ---------------------------------------------------------------------------
@skip_no_artifacts
@skip_no_parquet
def test_real_dataset_per_class_accuracy(pipeline):
    """Run 200 consecutive frames per class and assert each class >= 80% accuracy."""
    import pandas as pd

    df = pd.read_parquet(PREPARED_PARQUET)
    CLASSES = ["Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]
    N = 200

    print("\n" + "=" * 55)
    print("  Per-class accuracy on 200 consecutive frames each")
    print("=" * 55)

    for cls in CLASSES:
        class_df = df[df["label"] == cls].iloc[:N].reset_index(drop=True)
        assert len(class_df) == N, f"Not enough rows for {cls}"

        results = pipeline.predict_dataframe(class_df)
        assert len(results) == N

        correct = 0
        for res, (_, row) in zip(results, class_df.iterrows()):
            true_label = str(row["label"])
            pred_attack_type = res.get("attack_type")
            pred_label = res.get("label")

            if true_label == "Normal":
                if pred_label == "NORMAL":
                    correct += 1
            else:
                # Attack frame: attack_type or stage=rule
                if pred_label == "ATTACK":
                    correct += 1

        accuracy = correct / N
        print(f"  {cls:<12}: {correct}/{N} = {accuracy*100:.1f}%")
        assert accuracy >= 0.80, (
            f"Class '{cls}' accuracy {accuracy*100:.1f}% is below 80% threshold"
        )

    print("=" * 55)


# ---------------------------------------------------------------------------
# 6. SHAP top_features for ML-attack frame
# ---------------------------------------------------------------------------
@skip_no_artifacts
@skip_no_parquet
def test_shap_top_features_ml_attack(pipeline, feature_names):
    """An ML-path attack must have exactly 5 top_features from the feature list."""
    import pandas as pd

    if pipeline._shap_explainer is None:
        pytest.skip("SHAP explainer not initialised (shap not installed or parquet missing).")

    df = pd.read_parquet(PREPARED_PARQUET)
    # Use Spoof_Gear / Spoof_RPM frames — spoofed packets use valid CAN IDs and
    # pass rule checks to reach the ML stage
    attack_df = df[df["label"].isin(["Spoof_Gear", "Spoof_RPM", "Fuzzy"])].head(500).reset_index(drop=True)

    found_ml_attack = None
    for chunk_start in range(0, min(500, len(attack_df)), 20):
        chunk = attack_df.iloc[chunk_start:chunk_start+20]
        results = pipeline.predict_dataframe(chunk)
        for res in results:
            if res["stage"] == "ml" and res["is_attack"]:
                found_ml_attack = res
                break
        if found_ml_attack:
            break

    if found_ml_attack is None:
        pytest.skip("No ML-path attack found in the sample frames.")

    top_features = found_ml_attack["top_features"]
    assert len(top_features) == 5, (
        f"Expected 5 top_features, got {len(top_features)}: {top_features}"
    )

    for entry in top_features:
        assert "feature" in entry and "value" in entry and "shap_value" in entry
        assert entry["feature"] in feature_names, (
            f"Feature '{entry['feature']}' not in saved feature list {feature_names}"
        )

    print(f"\n[test] SHAP top features for ML attack: {[e['feature'] for e in top_features]}")


# ---------------------------------------------------------------------------
# 7. Stream state isolation
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_stream_state_isolation(pipeline):
    from src.pipeline import StreamState
    stateA = pipeline.new_stream_state()
    stateB = pipeline.new_stream_state()

    wl_id = next(iter(sorted(pipeline.whitelist))) if pipeline.whitelist else 0x0001

    for i in range(10):
        stateA.update(wl_id, 1000.0 + i * 0.01)

    rep_A, _ = stateA.update(wl_id, 1000.11)
    rep_B, _ = stateB.update(wl_id, 1000.11)

    assert rep_A > rep_B, (
        f"StateA rep={rep_A} should exceed fresh StateB rep={rep_B}"
    )


# ---------------------------------------------------------------------------
# 8. Backward time reset
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_backward_time_reset(pipeline):
    from src.pipeline import StreamState
    state = pipeline.new_stream_state()

    wl_id = next(iter(sorted(pipeline.whitelist))) if pipeline.whitelist else 0x0001

    for i in range(20):
        state.update(wl_id, 500.0 + i * 0.01)

    rep_after, iat_after = state.update(wl_id, 100.0)

    assert iat_after == 0.0, f"After backward jump, IAT should be 0.0, got {iat_after}"
    assert rep_after == 1, f"After backward jump, rep should be 1, got {rep_after}"


# ---------------------------------------------------------------------------
# 9. Unified schema completeness
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_unified_schema_completeness(pipeline):
    from src.pipeline import StreamState
    state  = pipeline.new_stream_state()
    engine = pipeline.new_rule_state()

    wl_id = next(iter(sorted(pipeline.whitelist))) if pipeline.whitelist else 0x0001

    # Normal frame
    result_normal = pipeline.predict_frame(
        timestamp=9000.0, can_id=wl_id, dlc=8, data_bytes=[0]*8,
        state=state, rule_engine=engine,
    )
    missing = REQUIRED_KEYS - set(result_normal.keys())
    assert not missing, f"Normal result missing keys: {missing}"

    # Unknown ID (rule path)
    UNKNOWN_ID = 0x7FFF
    if UNKNOWN_ID not in engine.whitelist:
        state2  = pipeline.new_stream_state()
        engine2 = pipeline.new_rule_state()
        result_rule = pipeline.predict_frame(
            timestamp=9001.0, can_id=UNKNOWN_ID, dlc=8, data_bytes=[0]*8,
            state=state2, rule_engine=engine2,
        )
        missing_r = REQUIRED_KEYS - set(result_rule.keys())
        assert not missing_r, f"Rule result missing keys: {missing_r}"
        assert result_rule["attack_type"] == "Rule violation"
        assert result_rule["rule_id"] is not None
        assert result_rule["reason"] is not None


# ---------------------------------------------------------------------------
# 10. Simulated flag propagation
# ---------------------------------------------------------------------------
@skip_no_artifacts
def test_simulated_flag(pipeline):
    from src.pipeline import StreamState
    state  = pipeline.new_stream_state()
    engine = pipeline.new_rule_state()

    wl_id = next(iter(sorted(pipeline.whitelist))) if pipeline.whitelist else 0x0001

    for flag in [True, False]:
        result = pipeline.predict_frame(
            timestamp=10000.0, can_id=wl_id, dlc=8, data_bytes=[0]*8,
            state=pipeline.new_stream_state(),
            rule_engine=pipeline.new_rule_state(),
            simulated=flag,
        )
        assert result["simulated"] is flag, (
            f"simulated flag not propagated: expected {flag}, got {result['simulated']}"
        )
