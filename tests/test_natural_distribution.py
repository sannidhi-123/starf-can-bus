"""Tests for realistic Natural Class Distribution evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPORT_PATH = _PROJECT_ROOT / "reports" / "natural_distribution_metrics.json"

REQUIRED_CLASSES = ("Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM")
REQUIRED_MODEL_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "false_positive_rate_normal",
    "false_alarms_per_100k_normal",
    "per_class",
    "confusion_matrix",
)


@pytest.mark.skipif(
    not _REPORT_PATH.exists(),
    reason="reports/natural_distribution_metrics.json does not exist",
)
def test_natural_distribution_report_exists_and_valid():
    """Verify natural_distribution_metrics.json has all required top-level keys and structure."""
    with open(_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for top_key in ("distribution", "xgboost", "random_forest", "date"):
        assert top_key in data, f"Missing top-level key: {top_key}"

    # Distribution checks
    dist = data["distribution"]
    assert "counts" in dist
    assert "percentages" in dist
    assert "total_frames" in dist
    assert dist["total_frames"] == sum(dist["counts"].values())

    # Check 90% Normal / 2.5% per attack class proportions
    assert dist["percentages"]["Normal"] == pytest.approx(90.0, abs=0.5)
    for cls in ("DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"):
        assert dist["percentages"][cls] == pytest.approx(2.5, abs=0.5)


@pytest.mark.skipif(
    not _REPORT_PATH.exists(),
    reason="reports/natural_distribution_metrics.json does not exist",
)
@pytest.mark.parametrize("model_key", ["xgboost", "random_forest"])
def test_natural_distribution_model_metrics(model_key: str):
    """Verify both models have per-class metrics, FPR, and false alarm rates."""
    with open(_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_metrics = data[model_key]
    for key in REQUIRED_MODEL_KEYS:
        assert key in model_metrics, f"{model_key} missing key: {key}"

    # Validate metric ranges
    assert 0.0 <= model_metrics["accuracy"] <= 1.0
    assert 0.0 <= model_metrics["macro_f1"] <= 1.0
    assert 0.0 <= model_metrics["false_positive_rate_normal"] <= 1.0
    assert model_metrics["false_alarms_per_100k_normal"] >= 0.0

    # Validate per_class metrics for all 5 classes
    per_class = model_metrics["per_class"]
    for cls in REQUIRED_CLASSES:
        assert cls in per_class, f"{model_key} missing per_class for {cls}"
        m = per_class[cls]
        for metric in ("precision", "recall", "f1-score", "support"):
            assert metric in m, f"{model_key} class {cls} missing {metric}"
            if metric != "support":
                assert 0.0 <= m[metric] <= 1.0
            else:
                assert m["support"] > 0
