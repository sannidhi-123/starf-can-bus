"""Unit tests for RuleEngine in src/rules.py.

Covers:
  - Clean normal sample (no triggers)
  - Crafted unknown CAN ID
  - Flood / frequency anomaly
  - Timing / inter-arrival time anomaly
  - Fit, serialization, and deserialization
  - Rule enabling/disabling via configuration dict
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from src.rules import RuleEngine


@pytest.fixture
def sample_config():
    """Sample rule engine configuration with predefined thresholds."""
    return {
        "enabled_rules": ["INVALID_ID", "FREQUENCY", "TIMING"],
        "whitelist": [0x0130, 0x0131, 0x0316, 0x043F],
        "frequency": {
            "window_sec": 0.5,
            "default_threshold": 20,
            "thresholds": {
                str(0x0130): 15,
                str(0x0316): 15,
            },
        },
        "timing": {
            "default_min_iat": 0.005,
            "thresholds": {
                str(0x0130): 0.005,  # 5 ms min IAT
                str(0x0316): 0.005,
            },
        },
    }


def test_clean_normal_sample_no_triggers(sample_config):
    """A stream of valid normal frames at regular intervals produces no rule triggers."""
    engine = RuleEngine(sample_config)

    # Stream CAN ID 0x0130 every 40 ms (12 msgs in 0.5s, well below threshold 15, well above 5ms min IAT)
    t = 1000.0
    for _ in range(25):
        triggered = engine.process_frame(timestamp=t, can_id=0x0130)
        assert triggered == [], f"Expected no triggers for normal frame at t={t}, got: {triggered}"
        t += 0.040  # 40 ms spacing



def test_crafted_unknown_id(sample_config):
    """A crafted CAN ID not in the normal whitelist triggers the INVALID_ID rule."""
    engine = RuleEngine(sample_config)

    # 0x0999 is not in whitelist [0x0130, 0x0131, 0x0316, 0x043F]
    triggered = engine.process_frame(timestamp=1000.0, can_id=0x0999)
    assert len(triggered) >= 1

    invalid_id_rule = next((r for r in triggered if r["rule_id"] == "INVALID_ID"), None)
    assert invalid_id_rule is not None
    assert "0x0999" in invalid_id_rule["reason"] or "2457" in invalid_id_rule["reason"]
    assert invalid_id_rule["evidence"]["can_id"] == 0x0999
    assert invalid_id_rule["evidence"]["whitelist_size"] == 4


def test_flood_frequency_anomaly(sample_config):
    """Injecting a flood of messages in a short window triggers the FREQUENCY rule."""
    engine = RuleEngine(sample_config)

    # Threshold for 0x0316 is 15 msgs in 0.5s window
    t = 1000.0
    flood_triggers = []

    for i in range(25):
        t += 0.006  # 6 ms spacing (passes timing rule of 5 ms)
        res = engine.process_frame(timestamp=t, can_id=0x0316)
        freq_trigger = next((r for r in res if r["rule_id"] == "FREQUENCY"), None)
        if freq_trigger:
            flood_triggers.append(freq_trigger)

    # The first 15 frames should not trigger frequency; frame 16+ should trigger
    assert len(flood_triggers) == 10, f"Expected 10 frequency triggers, got {len(flood_triggers)}"
    first_trigger = flood_triggers[0]
    assert first_trigger["rule_id"] == "FREQUENCY"
    assert first_trigger["evidence"]["threshold"] == 15
    assert first_trigger["evidence"]["count"] == 16
    assert "exceeded threshold" in first_trigger["reason"]


def test_timing_anomaly(sample_config):
    """Two consecutive frames arriving below the minimum IAT trigger the TIMING rule."""
    engine = RuleEngine(sample_config)

    # First frame at t=1000.0
    engine.process_frame(timestamp=1000.0, can_id=0x0130)

    # Second frame at t=1000.0005 (0.5 ms later, well below 5 ms minimum IAT)
    triggered = engine.process_frame(timestamp=1000.0005, can_id=0x0130)
    assert len(triggered) >= 1

    timing_rule = next((r for r in triggered if r["rule_id"] == "TIMING"), None)
    assert timing_rule is not None
    assert timing_rule["rule_id"] == "TIMING"
    assert pytest.approx(timing_rule["evidence"]["iat"], rel=1e-3) == 0.0005
    assert timing_rule["evidence"]["min_iat_threshold"] == 0.005
    assert "below" in timing_rule["reason"]


def test_fit_and_save_load_config(tmp_path):
    """fit() learns whitelist and thresholds from normal DataFrame and saves/loads JSON."""
    # Synthetic normal traffic: 2 IDs, periodic at 10 ms and 20 ms
    rows = []
    t = 100.0
    for _ in range(500):
        rows.append({"timestamp": t, "can_id": 0x0100})
        rows.append({"timestamp": t + 0.005, "can_id": 0x0200})
        t += 0.020

    df_normal = pd.DataFrame(rows)
    config_file = tmp_path / "test_rules_config.json"

    engine = RuleEngine()
    engine.fit(df_normal, save_path=config_file, window_sec=0.5)

    assert config_file.exists()
    assert 0x0100 in engine.whitelist
    assert 0x0200 in engine.whitelist
    assert 0x0999 not in engine.whitelist

    # Load into a fresh engine instance
    engine2 = RuleEngine()
    engine2.load_config(config_file)
    assert engine2.whitelist == engine.whitelist
    assert engine2.window_sec == engine.window_sec
    assert engine2.freq_thresholds == engine.freq_thresholds
    assert engine2.timing_thresholds == engine.timing_thresholds


def test_rule_disabling_via_config():
    """Rules can be selectively enabled or disabled via configuration dict."""
    config = {
        "enabled_rules": ["INVALID_ID"],  # Only check Invalid ID, disable Frequency and Timing
        "whitelist": [0x0130],
        "frequency": {"default_threshold": 5, "window_sec": 0.5},
        "timing": {"default_min_iat": 0.010},
    }
    engine = RuleEngine(config)

    # 1. Flood with known ID: frequency is disabled, so no trigger
    for _ in range(20):
        res = engine.process_frame(timestamp=100.0, can_id=0x0130)
        assert res == []

    # 2. Unknown ID: INVALID_ID is enabled, so triggers
    res = engine.process_frame(timestamp=100.0, can_id=0x0999)
    assert len(res) == 1
    assert res[0]["rule_id"] == "INVALID_ID"
