"""Hybrid detection pipeline: deterministic rules first, then ML classification.

Architecture
------------
1. RuleEngine (src/rules.py) evaluates three deterministic heuristics:
   - INVALID_ID  : CAN ID not in the whitelist built from normal traffic.
   - FREQUENCY   : Messages per ID in a sliding window above learned threshold.
   - TIMING      : Inter-arrival time per ID below learned minimum.
   If any rule fires, the frame is immediately labelled as an attack using the
   rule evidence — NO specific attack type is inferred from rule type.

2. RandomForestClassifier (models/multiclass_model.pkl) is called ONLY when
   no rule fires. Features: can_id, id_repetition_count,
   inter_arrival_time_delta, dlc, d0..d7.

3. SHAP (optional): A TreeExplainer is built at startup from a small background
   sample. For ML-path attack frames, top_features contains the 5 features with
   largest |SHAP value| for the predicted class.

Per-session / per-connection state
------------------------------------
All mutable state lives in StreamState (feature windows) and a companion
RuleEngine (rule sliding windows). These are created per connection/session.

Unified response schema
-----------------------
Every predict_frame() call returns:
{
    # Legacy fields (keep existing frontend working)
    "anomaly_score":      float,     # 1.0 rules; 1-P(Normal) for ML
    "threshold":          float,     # 1.0 sentinel for rules; 0.5 for ML
    "label":              str,       # "ATTACK" or "NORMAL"
    "is_attack":          bool,
    "processing_time_ms": float,
    "rules_triggered":    list,      # rule dicts: rule_id, reason, evidence
    "pipeline_path":      str,       # "rules" or "ml"

    # New fields
    "decision":           str,       # "attack" or "normal"
    "stage":              str,       # "rule" or "ml"
    "attack_type":        str|None,  # ML label or "Rule violation" or None
    "confidence":         float,
    "rule_id":            str|None,  # first triggered rule_id, or None
    "reason":             str|None,  # first triggered reason, or None
    "severity":           str|None,  # "High", "Medium", or None
    "simulated":          bool,      # passed through from caller
    "top_features":       list,      # [{feature, value, shap_value}] x 5 for ML attacks
}
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.rules import RuleEngine

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "models"

_RULES_CONFIG_PATH = _MODELS_DIR / "rules_config.json"
_MODEL_PATH        = _MODELS_DIR / "multiclass_model.pkl"
_FEATURES_PATH     = _MODELS_DIR / "multiclass_features.json"
_LABEL_ENC_PATH    = _MODELS_DIR / "multiclass_label_encoder.pkl"
_META_PATH         = _MODELS_DIR / "multiclass_meta.json"
_PREPARED_PATH     = _PROJECT_ROOT / "data" / "prepared.parquet"

_NORMAL_LABEL      = "Normal"
_DEFAULT_WINDOW_N  = 50
_SHAP_BACKGROUND_N = 200    # rows sampled for SHAP explainer background
_SHAP_TOP_K        = 5      # number of top features to return


# ---------------------------------------------------------------------------
# Severity map (Part B) — configured here; also exposed to backend/config.py
# ---------------------------------------------------------------------------
SEVERITY_MAP: Dict[str, str] = {
    "DoS":         "High",
    "Fuzzy":       "Medium",
    "Spoof_Gear":  "High",
    "Spoof_RPM":   "High",
    "Normal":      "None",
}

RULE_SEVERITY_MAP: Dict[str, str] = {
    "INVALID_ID": "Medium",
    "FREQUENCY":  "High",
    "TIMING":     "Medium",
}

SEVERITY_ORDER: Dict[Optional[str], int] = {
    "High": 2,
    "Medium": 1,
    "None": 0,
    None: 0,
}


def _highest_rule_severity(triggered: List[Dict[str, Any]]) -> Optional[str]:
    """Return the highest severity among all triggered rules (High > Medium)."""
    highest: Optional[str] = None
    for r in triggered:
        r_sev = RULE_SEVERITY_MAP.get(r.get("rule_id", ""))
        if r_sev and SEVERITY_ORDER.get(r_sev, 0) > SEVERITY_ORDER.get(highest, 0):
            highest = r_sev
    return highest


def _get_severity(attack_type: Optional[str], rule_id: Optional[str]) -> Optional[str]:
    """Return severity string for an attack result."""
    if rule_id:
        return RULE_SEVERITY_MAP.get(rule_id)
    if attack_type:
        s = SEVERITY_MAP.get(attack_type)
        return None if s == "None" else s
    return None


# ---------------------------------------------------------------------------
# StreamState
# ---------------------------------------------------------------------------
@dataclass
class StreamState:
    """Mutable state for one streaming context (REST session or WS connection).

    Holds:
      - id_window    : deque of last N can_id integers (id_repetition_count)
      - id_counts    : Counter mirroring the deque for O(1) lookups
      - last_ts      : dict mapping can_id -> last seen timestamp (IAT delta)
      - last_global_t: float, last overall timestamp seen (backward-jump detection)
    """

    window_n: int = _DEFAULT_WINDOW_N
    id_window: deque = field(default_factory=deque)
    id_counts: Counter = field(default_factory=Counter)
    last_ts: Dict[int, float] = field(default_factory=dict)
    last_global_t: Optional[float] = None

    def reset(self) -> None:
        self.id_window.clear()
        self.id_counts.clear()
        self.last_ts.clear()
        self.last_global_t = None

    def update(self, can_id: int, timestamp: float):
        """Ingest one frame; return (id_repetition_count, inter_arrival_time_delta)."""
        if self.last_global_t is not None and timestamp < self.last_global_t:
            self.id_window.clear()
            self.id_counts.clear()
            self.last_ts.clear()
        self.last_global_t = timestamp

        if len(self.id_window) == self.window_n:
            evicted = self.id_window.popleft()
            self.id_counts[evicted] -= 1
        self.id_window.append(can_id)
        self.id_counts[can_id] += 1
        rep_count = int(self.id_counts[can_id])

        iat = float(timestamp - self.last_ts[can_id]) if can_id in self.last_ts else 0.0
        self.last_ts[can_id] = timestamp

        return rep_count, iat


# ---------------------------------------------------------------------------
# HybridPipeline
# ---------------------------------------------------------------------------
class HybridPipeline:
    """Rules-first, ML-second hybrid IDS pipeline for CAN bus frames.

    The instance itself is stateless after __init__ (no mutable fields).
    All streaming state lives in caller-managed StreamState + RuleEngine objects.

    Usage
    -----
    pipeline = HybridPipeline()            # loads all assets once at startup
    state    = pipeline.new_stream_state() # one per connection / REST session
    engine   = pipeline.new_rule_state()   # one per connection / REST session
    result   = pipeline.predict_frame(
        timestamp=..., can_id=..., dlc=..., data_bytes=[...],
        state=state, rule_engine=engine,
        simulated=False,
    )
    """

    def __init__(
        self,
        rules_config_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        features_path: Optional[Path] = None,
        label_enc_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        window_n: int = _DEFAULT_WINDOW_N,
    ) -> None:
        rcp = Path(rules_config_path or _RULES_CONFIG_PATH)
        mp  = Path(model_path        or _MODEL_PATH)
        fp  = Path(features_path     or _FEATURES_PATH)
        lp  = Path(label_enc_path    or _LABEL_ENC_PATH)
        mtp = Path(meta_path         or _META_PATH)

        missing = [str(p) for p in (rcp, mp, fp, lp) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "HybridPipeline: required model artifacts missing:\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\nRun 'python -m src.train_multiclass' and "
                  "'python -m src.rules' first."
            )

        # Rule Engine
        self._rule_engine = RuleEngine()
        self._rule_engine.load_config(rcp)

        # ML assets
        self._model = joblib.load(mp)
        with fp.open("r", encoding="utf-8") as f:
            self._feature_names: List[str] = json.load(f)
        self._label_enc = joblib.load(lp)
        self._meta: Dict[str, Any] = (
            json.loads(mtp.read_text(encoding="utf-8")) if mtp.exists() else {}
        )
        self._classes: List[str] = list(self._label_enc.classes_)
        self._window_n = window_n

        # SHAP explainer (optional)
        self._shap_explainer: Optional[Any] = None
        self._shap_feature_names: List[str] = list(self._feature_names)
        self._try_init_shap()

    # ------------------------------------------------------------------
    # SHAP initialisation
    # ------------------------------------------------------------------
    def _try_init_shap(self) -> None:
        """Build a SHAP TreeExplainer using a small background sample."""
        try:
            import shap  # noqa: F401
        except ImportError:
            return  # SHAP not installed — skip silently

        try:
            if not _PREPARED_PATH.exists():
                return

            df = pd.read_parquet(_PREPARED_PATH)
            normal_df = df[df["label"] == _NORMAL_LABEL]
            bg = normal_df.sample(
                n=min(_SHAP_BACKGROUND_N, len(normal_df)), random_state=42
            )
            # Build feature matrix for background using the feature builder
            from src.preprocess import build_frame_features  # noqa: PLC0415
            feat_bg = build_frame_features(bg.reset_index(drop=True))

            # Align to model feature names
            X_bg = np.zeros((len(feat_bg), len(self._feature_names)), dtype=np.float32)
            for col_idx, name in enumerate(self._feature_names):
                if name in feat_bg.columns:
                    col_vals = feat_bg[name].values
                    X_bg[:, col_idx] = self._parse_feature_col(col_vals, name)

            import shap  # noqa: F811
            self._shap_explainer = shap.TreeExplainer(self._model, X_bg)
        except Exception as exc:  # pragma: no cover
            # Non-fatal — SHAP is best-effort
            print(f"[Pipeline] SHAP init failed (non-fatal): {exc}")
            self._shap_explainer = None

    @staticmethod
    def _parse_feature_col(values: np.ndarray, name: str) -> np.ndarray:
        """Parse a column that may contain hex strings or numerics."""
        result = np.zeros(len(values), dtype=np.float32)
        for i, v in enumerate(values):
            result[i] = HybridPipeline._parse_byte_val(v)
        return result

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def new_stream_state(self) -> StreamState:
        """Return a fresh per-connection StreamState."""
        return StreamState(window_n=self._window_n)

    def new_rule_state(self) -> RuleEngine:
        """Return a fresh RuleEngine with the same config but clean runtime state."""
        return RuleEngine(config=self._rule_engine.config)

    @property
    def whitelist(self):
        return self._rule_engine.whitelist

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def shap_explainer(self) -> Optional[Any]:
        """Return the initialized SHAP TreeExplainer, or None if unavailable."""
        return self._shap_explainer

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_frame(
        self,
        timestamp: float,
        can_id: int,
        dlc: int,
        data_bytes: List[int],
        state: StreamState,
        rule_engine: Optional[RuleEngine] = None,
        simulated: bool = False,
        compute_shap: bool = True,
    ) -> Dict[str, Any]:
        """Process one CAN bus frame through the hybrid pipeline.

        Parameters
        ----------
        timestamp   : Frame arrival timestamp (seconds, float).
        can_id      : CAN arbitration ID (integer).
        dlc         : Data Length Code (0-8).
        data_bytes  : Up to 8 payload bytes as ints [0-255]. Zero-padded to 8.
        state       : Per-connection StreamState (mutable, updated in place).
        rule_engine : Isolated RuleEngine for this connection (optional).
        simulated   : If True, tag the response with simulated=True.

        Returns
        -------
        dict  — unified response schema documented in module docstring.
        """
        t_start = time.perf_counter()

        cid     = int(can_id)
        ts      = float(timestamp)
        dlc_val = int(dlc)
        padded  = (list(data_bytes) + [0] * 8)[:8]
        padded  = [int(b) & 0xFF for b in padded]

        # ── 1. Rule Engine ────────────────────────────────────────────────
        engine   = rule_engine if rule_engine is not None else self._rule_engine
        triggered = engine.process_frame(timestamp=ts, can_id=cid)

        if triggered:
            rep_count, iat_delta = state.update(cid, ts)  # keep feature windows coherent
            first  = triggered[0]
            rid    = first["rule_id"]
            reason = first["reason"]
            # When several rules fire for one frame, set severity to the highest among them (High > Medium)
            sev    = _highest_rule_severity(triggered)

            # Evaluate Stage 2 ML classifier to identify specific attack type
            feature_row = self._build_feature_row(cid, rep_count, iat_delta, dlc_val, padded)
            X = np.array([feature_row], dtype=np.float32)
            probs: np.ndarray = self._model.predict_proba(X)[0]
            top_idx = int(np.argmax(probs))
            ml_label = str(self._classes[top_idx])
            ml_conf = float(probs[top_idx])

            # If ML model recognizes a known attack type (DoS, Fuzzy, Spoof_Gear, Spoof_RPM)
            top_features: List[Dict[str, Any]] = []
            if ml_label != _NORMAL_LABEL:
                attack_type = ml_label
                if compute_shap and self._shap_explainer is not None:
                    top_features = self._compute_shap_top_k(X, top_idx, feature_row)
                specific_sev = _get_severity(attack_type, None)
                if specific_sev and SEVERITY_ORDER.get(specific_sev, 0) > SEVERITY_ORDER.get(sev, 0):
                    sev = specific_sev
            else:
                attack_type = "Rule violation"

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                # Legacy fields (preserved for backwards compatibility with earlier dashboard components)
                "anomaly_score":      1.0,
                "threshold":          1.0,
                "label":              "ATTACK",
                "attack_type":        attack_type,
                "confidence":         round(ml_conf, 4) if attack_type != "Rule violation" else 1.0,
                "is_attack":          True,
                "processing_time_ms": round(elapsed_ms, 3),
                "rules_triggered":    triggered,
                "pipeline_path":      "rules",
                # Unified hybrid pipeline fields
                "decision":   "attack",
                "stage":      "rule",
                "rule_id":    rid,
                "reason":     reason,
                "severity":   sev,
                "simulated":  simulated,
                "top_features": top_features,
            }

        # ── 2. ML path ───────────────────────────────────────────────────
        rep_count, iat_delta = state.update(cid, ts)
        feature_row = self._build_feature_row(cid, rep_count, iat_delta, dlc_val, padded)
        X = np.array([feature_row], dtype=np.float32)

        probs: np.ndarray  = self._model.predict_proba(X)[0]
        top_idx            = int(np.argmax(probs))
        ml_label: str      = str(self._classes[top_idx])
        confidence         = float(probs[top_idx])

        is_attack   = ml_label != _NORMAL_LABEL
        label       = "ATTACK" if is_attack else "NORMAL"
        attack_type = ml_label if is_attack else None
        decision    = "attack" if is_attack else "normal"
        sev         = _get_severity(attack_type, None)

        normal_idx    = self._normal_class_idx()
        anomaly_score = round(float(1.0 - probs[normal_idx]), 6)

        # SHAP for ML-path attacks only
        top_features: List[Dict[str, Any]] = []
        if compute_shap and is_attack and self._shap_explainer is not None:
            top_features = self._compute_shap_top_k(X, top_idx, feature_row)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            # Legacy fields (preserved for backwards compatibility with earlier dashboard components)
            "anomaly_score":      anomaly_score,
            "threshold":          0.5,
            "label":              label,
            "attack_type":        attack_type,
            "confidence":         round(confidence, 4),
            "is_attack":          is_attack,
            "processing_time_ms": round(elapsed_ms, 3),
            "rules_triggered":    [],
            "pipeline_path":      "ml",
            # Unified hybrid pipeline fields
            "decision":    decision,
            "stage":       "ml",
            "rule_id":     None,
            "reason":      None,
            "severity":    sev,
            "simulated":   simulated,
            "top_features": top_features,
        }

    # ------------------------------------------------------------------
    # SHAP computation
    # ------------------------------------------------------------------

    def _compute_shap_top_k(
        self,
        X: np.ndarray,
        class_idx: int,
        feature_row: List[float],
    ) -> List[Dict[str, Any]]:
        """Return top-K features by |SHAP value| for the predicted class."""
        try:
            shap_vals = self._shap_explainer.shap_values(X)
            # shap_vals shape: (n_classes, n_samples, n_features)
            # for multi-output RF
            if isinstance(shap_vals, list):
                class_shap = np.array(shap_vals[class_idx][0])
            else:
                # (n_samples, n_features, n_classes) shape
                class_shap = shap_vals[0, :, class_idx]

            top_k_idx = np.argsort(np.abs(class_shap))[::-1][:_SHAP_TOP_K]
            return [
                {
                    "feature":    self._feature_names[i],
                    "value":      float(feature_row[i]),
                    "shap_value": round(float(class_shap[i]), 6),
                }
                for i in top_k_idx
            ]
        except Exception:  # pragma: no cover
            return []

    # ------------------------------------------------------------------
    # Batch helper (evaluation / testing)
    # ------------------------------------------------------------------

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        simulated: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run predict_frame over every row of a time-ordered DataFrame.

        Required columns: timestamp, can_id, dlc, d0..d7.
        Byte columns may contain integers or hex strings (e.g. '0c', 'ff').
        Returns list of result dicts aligned with df rows.
        """
        state  = self.new_stream_state()
        engine = self.new_rule_state()
        byte_cols = [f"d{i}" for i in range(8)]
        results = []
        for row in df.itertuples():
            data_bytes = [self._parse_byte_val(getattr(row, c, 0)) for c in byte_cols]
            res = self.predict_frame(
                timestamp=float(row.timestamp),
                can_id=int(row.can_id),
                dlc=int(getattr(row, "dlc", 8)),
                data_bytes=data_bytes,
                state=state,
                rule_engine=engine,
                simulated=simulated,
            )
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normal_class_idx(self) -> int:
        try:
            return self._classes.index(_NORMAL_LABEL)
        except ValueError:
            return 0

    def _build_feature_row(
        self,
        can_id: int,
        rep_count: int,
        iat_delta: float,
        dlc: int,
        padded_bytes: List[int],
    ) -> List[float]:
        lookup: Dict[str, float] = {
            "can_id":                   float(can_id),
            "id_repetition_count":      float(rep_count),
            "inter_arrival_time_delta": float(iat_delta),
            "dlc":                      float(dlc),
        }
        for i, b in enumerate(padded_bytes):
            lookup[f"d{i}"] = float(b)
        return [lookup.get(name, 0.0) for name in self._feature_names]

    @staticmethod
    def _parse_byte_val(val: Any) -> int:
        """Parse a byte value that may be an int, float, or hex string."""
        if val is None:
            return 0
        try:
            if isinstance(val, (int, float, np.integer, np.floating)):
                return int(val) & 0xFF
            s = str(val).strip().lower()
            if not s or s in ("nan", "none", ""):
                return 0
            if s.startswith("0x"):
                return int(s, 16) & 0xFF
            if len(s) <= 2:
                try:
                    return int(s, 16) & 0xFF
                except ValueError:
                    pass
            return int(float(s)) & 0xFF
        except (ValueError, TypeError):
            return 0
