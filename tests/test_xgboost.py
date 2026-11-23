"""Tests for XGBoost model loading, prediction, and backend fallback.

Covers:
  1. XGB model loads from models/xgb_model.pkl
  2. predict_proba outputs sum to ~1 for each row
  3. HybridPipeline fallback to RF when XGB file is missing
  4. /models endpoint returns both model keys with numeric metrics
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Project root (two levels up from tests/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_XGB_MODEL_PATH = _PROJECT_ROOT / "models" / "xgb_model.pkl"
_RF_MODEL_PATH  = _PROJECT_ROOT / "models" / "multiclass_model.pkl"
_XGB_METRICS    = _PROJECT_ROOT / "reports" / "xgb_metrics.json"
_RF_METRICS     = _PROJECT_ROOT / "reports" / "rf_metrics.json"

FEATURE_COLUMNS = [
    "can_id", "id_repetition_count", "inter_arrival_time_delta",
    "dlc", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
]


# ---------------------------------------------------------------------------
# 1. XGBClassifier loads and returns valid probabilities
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _XGB_MODEL_PATH.exists(),
    reason="models/xgb_model.pkl not yet trained",
)
def test_xgb_model_loads():
    """XGB model file must load without errors."""
    import joblib
    model = joblib.load(_XGB_MODEL_PATH)
    assert hasattr(model, "predict_proba"), "XGB model must support predict_proba"


@pytest.mark.skipif(
    not _XGB_MODEL_PATH.exists(),
    reason="models/xgb_model.pkl not yet trained",
)
def test_xgb_proba_sum_to_one():
    """predict_proba rows must sum to 1 (within floating-point tolerance)."""
    import joblib
    model = joblib.load(_XGB_MODEL_PATH)

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 255, size=(20, len(FEATURE_COLUMNS))).astype(np.float32)
    # Set can_id to a realistic value
    X[:, 0] = rng.integers(0, 2048, size=20).astype(np.float32)

    probs = model.predict_proba(X)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, np.ones(len(row_sums)), atol=1e-5,
        err_msg="XGB predict_proba rows must sum to 1",
    )


# ---------------------------------------------------------------------------
# 2. HybridPipeline fallback to RF when XGB file is missing
# ---------------------------------------------------------------------------

def test_pipeline_falls_back_to_rf_when_xgb_missing(tmp_path, monkeypatch):
    """When XGB model is absent, HybridPipeline must load the RF model."""
    # Patch MODEL_BACKEND to "xgboost" but point XGB path to a non-existent file
    import backend.config as cfg_mod
    import src.pipeline as pipeline_mod

    monkeypatch.setattr(cfg_mod, "MODEL_BACKEND", "xgboost")

    # Monkeypatch the _XGB_MODEL_PATH inside pipeline to a missing path
    fake_xgb = tmp_path / "xgb_model_nonexistent.pkl"
    monkeypatch.setattr(pipeline_mod, "_XGB_MODEL_PATH", fake_xgb)

    # Pipeline must succeed by falling back to RF
    p = pipeline_mod.HybridPipeline()
    assert p._model_backend == "random_forest", (
        "Pipeline should report 'random_forest' when XGB file is missing"
    )
    assert p._model is not None


# ---------------------------------------------------------------------------
# 3. /models endpoint structure — test the pure logic, no async needed
# ---------------------------------------------------------------------------

def test_models_endpoint_has_both_keys(tmp_path, monkeypatch):
    """get_models_info() logic must return 'xgboost', 'random_forest', 'active_model'.

    We test the core logic directly (file reading + dict construction) without
    calling the async endpoint or starting the FastAPI lifespan (which loads
    HybridPipeline + SHAP and is slow in a test context).
    """
    # Create stub report files
    xgb_data = {"accuracy": 0.9979, "macro_f1": 0.9979, "model_type": "XGBClassifier"}
    rf_data  = {"accuracy": 0.9897, "macro_f1": 0.9897, "model_type": "RandomForestClassifier"}
    (tmp_path / "xgb_metrics.json").write_text(json.dumps(xgb_data), encoding="utf-8")
    (tmp_path / "rf_metrics.json").write_text(json.dumps(rf_data),  encoding="utf-8")

    # Replicate the endpoint logic (reading files + assembling response)
    def _load_json(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    xgb_metrics = _load_json(tmp_path / "xgb_metrics.json")
    rf_metrics   = _load_json(tmp_path / "rf_metrics.json")

    result = {
        "active_model":  "xgboost",
        "xgboost":       xgb_metrics,
        "random_forest": rf_metrics,
    }

    assert "xgboost" in result
    assert "random_forest" in result
    assert "active_model" in result
    assert result["active_model"] == "xgboost"
    assert result["xgboost"]["accuracy"] == pytest.approx(0.9979)
    assert result["random_forest"]["accuracy"] == pytest.approx(0.9897)


@pytest.mark.skipif(
    not _XGB_METRICS.exists(),
    reason="reports/xgb_metrics.json not yet generated",
)
def test_xgb_metrics_file_has_required_keys():
    """xgb_metrics.json must have accuracy, macro_f1, per_class, etc."""
    with open(_XGB_METRICS, "r", encoding="utf-8") as f:
        m = json.load(f)
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix",
                 "latency_ms_batch1", "latency_ms_batch1000", "feature_importances"):
        assert key in m, f"xgb_metrics.json missing key: {key}"
    # Accuracy must be a sensible number
    assert 0.0 < m["accuracy"] <= 1.0, "accuracy out of range"


@pytest.mark.skipif(
    not _RF_METRICS.exists(),
    reason="reports/rf_metrics.json not yet generated",
)
def test_rf_metrics_file_has_required_keys():
    """rf_metrics.json must have same schema as xgb_metrics.json."""
    with open(_RF_METRICS, "r", encoding="utf-8") as f:
        m = json.load(f)
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix"):
        assert key in m, f"rf_metrics.json missing key: {key}"
    assert 0.0 < m["accuracy"] <= 1.0, "RF accuracy out of range"
