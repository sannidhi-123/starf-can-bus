"""Production-ready FastAPI backend for CAN Bus IDS."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from transformers import DistilBertConfig, DistilBertForMaskedLM

from backend.config import settings
from src.multiclass_inference import predict_attack_type

try:
    from src.pipeline import HybridPipeline, StreamState
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False


BASE_DIR = settings.PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

# StaticFiles validates the directory when mounted, so create it at import time.
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = settings.VOCAB_PATH
MODEL_PATH = settings.MODEL_PATH
THRESHOLD_PATH = settings.THRESHOLD_PATH
SEQUENCES_PATH = DATA_DIR / "sequences.pt"
RF_CLASSIFIER_PATH = MODELS_DIR / "attack_classifier.pkl"
RF_META_PATH = MODELS_DIR / "attack_classifier_meta.json"

WINDOW_SIZE = 64
MASK_RATIO = 0.15

PAD_ID = 0
MASK_ID = 1
UNK_ID = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UVICORN_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "log_level": "info",
}


class PredictRequest(BaseModel):
    # Optional sequence (legacy DistilBERT)
    sequence: Optional[List[str]] = Field(
        default=None,
        description="List of 64 hex CAN IDs (legacy)",
    )
    return_details: bool = Field(default=True)

    # Optional single frame fields (hybrid pipeline)
    timestamp: Optional[float] = Field(default=None, description="Frame arrival timestamp in seconds")
    can_id: Optional[Union[int, str]] = Field(default=None, description="CAN arbitration ID (int or hex)")
    dlc: Optional[int] = Field(default=8, ge=0, le=8, description="Data Length Code")
    data_bytes: Optional[List[int]] = Field(
        default=None,
        description="Up to 8 payload bytes [0-255]",
    )
    simulated: Optional[bool] = Field(default=False, description="Tag response as simulated data")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        if len(value) != WINDOW_SIZE:
            raise ValueError(f"sequence must contain exactly {WINDOW_SIZE} CAN IDs")
        for item in value:
            if not str(item).strip():
                raise ValueError("sequence entries cannot be empty")
        return value


class PredictResponse(BaseModel):
    # --- Legacy fields (preserved for backwards compatibility with earlier dashboard components) ---
    anomaly_score: float = Field(
        ...,
        description="[Legacy field] Anomaly score between 0.0 and 1.0 (1.0 for rule hits, 1 - P(Normal) for ML). Maintained for backwards compatibility.",
    )
    threshold: float = Field(
        ...,
        description="[Legacy field] Detection threshold cutoff (1.0 for rule stage, 0.5 for ML classification). Maintained for backwards compatibility.",
    )
    label: str = Field(..., description="[Legacy field] Binary detection label ('ATTACK' or 'NORMAL').")
    attack_type: Optional[str] = None
    confidence: float
    is_attack: bool
    processing_time_ms: float
    details: Optional[Dict[str, Any]] = None
    # --- Unified hybrid pipeline fields ---
    decision: Optional[str] = None
    stage: Optional[str] = None
    rule_id: Optional[str] = None
    reason: Optional[str] = None
    severity: Optional[str] = None
    simulated: Optional[bool] = False
    top_features: Optional[List[Dict[str, Any]]] = None
    rules_triggered: Optional[List[Any]] = None
    pipeline_path: Optional[str] = None


class BatchPredictRequest(BaseModel):
    sequences: List[List[str]] = Field(..., min_length=1, max_length=1000)
    return_details: bool = Field(default=False)

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, sequences: List[List[str]]) -> List[List[str]]:
        for idx, seq in enumerate(sequences):
            if len(seq) != WINDOW_SIZE:
                raise ValueError(f"sequence at index {idx} must contain exactly {WINDOW_SIZE} CAN IDs")
            if any(not str(item).strip() for item in seq):
                raise ValueError(f"sequence at index {idx} contains empty token")
        return sequences


class BatchPredictResponse(BaseModel):
    total: int
    attacks: int
    normals: int
    average_score: float
    predictions: List[PredictResponse]


class ThresholdUpdateRequest(BaseModel):
    threshold: float = Field(..., gt=0.0, le=20.0)
    persist: bool = Field(default=True)


class StreamConfigUpdateRequest(BaseModel):
    interval_ms: int = Field(..., ge=100, le=5000)


@dataclass
class RuntimeState:
    model: Optional[DistilBertForMaskedLM] = None
    vocab: Dict[str, int] = field(default_factory=dict)
    threshold_data: Dict[str, Any] = field(default_factory=dict)
    rf_classifier: Optional[Any] = None
    rf_meta: Dict[str, Any] = field(default_factory=dict)
    demo_sequences: Optional[torch.Tensor] = None
    demo_index: int = 0
    stream_interval_ms: int = 650
    loaded: bool = False
    # Hybrid pipeline (rules + ML)
    pipeline: Optional[Any] = None  # HybridPipeline | None
    pipeline_loaded: bool = False


@dataclass
class DetectionStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_analyzed: int = 0
    attacks_detected: int = 0
    normals_detected: int = 0
    score_sum: float = 0.0
    attack_type_counts: Counter = field(default_factory=Counter)
    last_prediction: Dict[str, Any] = field(default_factory=dict)

    def update(self, result: Dict[str, Any]) -> None:
        self.total_analyzed += 1
        self.score_sum += float(result["anomaly_score"])
        if result["is_attack"]:
            self.attacks_detected += 1
            if result.get("attack_type"):
                self.attack_type_counts[result["attack_type"]] += 1
        else:
            self.normals_detected += 1

        self.last_prediction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": result["label"],
            "attack_type": result.get("attack_type"),
            "score": result["anomaly_score"],
            "confidence": result["confidence"],
        }

    def snapshot(self) -> Dict[str, Any]:
        avg_score = self.score_sum / self.total_analyzed if self.total_analyzed else 0.0
        detection_rate = (self.attacks_detected / self.total_analyzed * 100.0) if self.total_analyzed else 0.0
        uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()

        return {
            "total_analyzed": self.total_analyzed,
            "attacks_detected": self.attacks_detected,
            "normals_detected": self.normals_detected,
            "detection_rate": round(detection_rate, 4),
            "average_anomaly_score": round(avg_score, 6),
            "attack_type_breakdown": dict(self.attack_type_counts),
            "last_prediction": self.last_prediction,
            "uptime_seconds": round(uptime, 2),
        }


runtime_state = RuntimeState()
stats_state = DetectionStats()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_threshold_data(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sync_report_artifacts() -> None:
    """Keep backend-served report artifacts in sync with generated evaluation outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    chart_sources: Dict[str, Sequence[Path]] = {
        "confusion_matrix.png": [RESULTS_FIGURES_DIR / "confusion_matrix.png"],
        "roc_curve.png": [RESULTS_FIGURES_DIR / "roc_curve.png"],
        "pr_curve.png": [
            RESULTS_FIGURES_DIR / "pr_curve.png",
            RESULTS_FIGURES_DIR / "precision_recall_curve.png",
        ],
        "score_distribution.png": [RESULTS_FIGURES_DIR / "score_distribution.png"],
        "metrics_summary.png": [RESULTS_FIGURES_DIR / "metrics_summary.png"],
    }

    for target_name, candidates in chart_sources.items():
        for source_path in candidates:
            if source_path.exists():
                shutil.copy2(source_path, REPORTS_DIR / target_name)
                break

    metrics_source = RESULTS_DIR / "evaluation_metrics.json"
    if metrics_source.exists():
        shutil.copy2(metrics_source, REPORTS_DIR / "metrics.json")


def normalize_can_id(raw: str) -> str:
    token = str(raw).strip().lower()
    if token.startswith("0x"):
        token = token[2:]
    return token.zfill(4)


def build_model(vocab_size: int) -> DistilBertForMaskedLM:
    config = DistilBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=WINDOW_SIZE,
        dim=256,
        n_layers=4,
        n_heads=4,
        hidden_dim=1024,
        dropout=0.1,
        attention_dropout=0.1,
        pad_token_id=PAD_ID,
    )
    return DistilBertForMaskedLM(config)


def _load_demo_sequences() -> Optional[torch.Tensor]:
    if not SEQUENCES_PATH.exists():
        return None

    payload = torch.load(SEQUENCES_PATH, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        seq = payload
    elif isinstance(payload, dict) and "sequences" in payload:
        seq = payload["sequences"]
    else:
        return None

    if seq.ndim != 2 or seq.shape[1] != WINDOW_SIZE:
        return None

    return seq.long()


def load_runtime_assets() -> None:
    ensure_exists(VOCAB_PATH, "vocab.json")
    ensure_exists(MODEL_PATH, "best_model.pt")
    ensure_exists(THRESHOLD_PATH, "threshold.json")

    vocab = load_json(VOCAB_PATH)
    runtime_state.vocab = {str(k).lower(): int(v) for k, v in vocab.items()}

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(len(runtime_state.vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    runtime_state.model = model

    runtime_state.threshold_data = load_json(THRESHOLD_PATH)

    if RF_CLASSIFIER_PATH.exists():
        runtime_state.rf_classifier = joblib.load(RF_CLASSIFIER_PATH)
    if RF_META_PATH.exists():
        runtime_state.rf_meta = load_json(RF_META_PATH)

    runtime_state.demo_sequences = _load_demo_sequences()
    runtime_state.demo_index = 0
    runtime_state.loaded = True

    # Hybrid pipeline (optional — doesn't block startup if missing)
    if _PIPELINE_AVAILABLE:
        try:
            runtime_state.pipeline = HybridPipeline()
            runtime_state.pipeline_loaded = True
            print("[STARF] HybridPipeline loaded (rules + multiclass ML).")
        except FileNotFoundError as exc:
            runtime_state.pipeline = None
            runtime_state.pipeline_loaded = False
            print(f"[STARF] HybridPipeline not loaded (missing artifacts): {exc}")


def _ensure_at_least_one_mask(masked_indices: torch.Tensor, special: torch.Tensor) -> torch.Tensor:
    rows = torch.where(masked_indices.sum(dim=1) == 0)[0]
    if rows.numel() == 0:
        return masked_indices

    for row in rows.tolist():
        candidates = torch.where(~special[row])[0]
        if candidates.numel() == 0:
            candidates = torch.arange(masked_indices.shape[1], device=masked_indices.device)
        pick = candidates[torch.randint(0, candidates.numel(), (1,), device=masked_indices.device)]
        masked_indices[row, pick] = True

    return masked_indices


def score_token_tensor(token_tensor: torch.Tensor) -> np.ndarray:
    if runtime_state.model is None:
        raise RuntimeError("Model is not loaded")

    batch = token_tensor.to(DEVICE)
    bsz, seq_len = batch.shape

    input_ids = batch.clone()
    labels = torch.full((bsz, seq_len), -100, dtype=torch.long, device=DEVICE)

    mask_prob = torch.rand((bsz, seq_len), device=DEVICE)
    masked_indices = mask_prob < MASK_RATIO

    special = (batch == PAD_ID) | (batch == MASK_ID) | (batch == UNK_ID)
    masked_indices = masked_indices & ~special
    masked_indices = _ensure_at_least_one_mask(masked_indices, special)

    labels[masked_indices] = batch[masked_indices]
    input_ids[masked_indices] = MASK_ID

    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        outputs = runtime_state.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss_per_pos = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bsz, seq_len)

        mask_counts = (labels != -100).sum(dim=1).clamp(min=1)
        scores = loss_per_pos.sum(dim=1) / mask_counts

    return scores.detach().cpu().numpy()


def tokenize_sequence(sequence: Sequence[str]) -> Tuple[List[int], int]:
    if not runtime_state.vocab:
        raise RuntimeError("Vocabulary not loaded")

    tokens: List[int] = []
    unknown = 0

    for item in sequence:
        can_id = normalize_can_id(str(item))
        token = runtime_state.vocab.get(can_id, UNK_ID)
        if token == UNK_ID:
            unknown += 1
        tokens.append(token)

    return tokens, unknown


def _feature_vector(tokens: Sequence[int], anomaly_score: float) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.float32)
    unique_ratio = float(len(np.unique(arr)) / max(len(arr), 1))
    return np.array(
        [
            float(anomaly_score),
            float(np.mean(arr)),
            float(np.std(arr)),
            unique_ratio,
            float(np.max(arr)),
            float(np.min(arr)),
        ],
        dtype=np.float32,
    )


def _range_based_label(score: float, score_ranges: Dict[str, Dict[str, float]]) -> str:
    candidates = [
        label
        for label, params in score_ranges.items()
        if float(params["low"]) <= score <= float(params["high"])
    ]

    if candidates:
        return min(candidates, key=lambda lbl: abs(score - float(score_ranges[lbl]["mean"])))

    return min(score_ranges.keys(), key=lambda lbl: abs(score - float(score_ranges[lbl]["mean"])))


def classify_attack(score: float, tokens: Sequence[int], threshold: float) -> Tuple[Optional[str], float]:
    if score <= threshold:
        confidence = max(0.5, min(0.99, (threshold - score) / max(threshold, 1e-8)))
        return None, float(confidence)

    rf_classifier = runtime_state.rf_classifier
    score_ranges = getattr(rf_classifier, "score_ranges_", {}) if rf_classifier is not None else {}

    if rf_classifier is not None:
        features = _feature_vector(tokens, score)
        probs = rf_classifier.predict_proba(features.reshape(1, -1))[0]
        classes = list(rf_classifier.classes_)
        top_idx = int(np.argmax(probs))
        rf_label = str(classes[top_idx])
        rf_conf = float(probs[top_idx])

        if score_ranges:
            range_label = _range_based_label(score, score_ranges)
            range_mean = float(score_ranges[range_label]["mean"])
            range_std = max(float(score_ranges[range_label]["std"]), 1e-8)
            range_conf = float(np.exp(-0.5 * abs(score - range_mean) / range_std))

            if rf_label == range_label:
                return rf_label, float(max(rf_conf, range_conf))

            if 0.65 * rf_conf >= 0.35 * range_conf:
                return rf_label, rf_conf
            return range_label, range_conf

        return rf_label, rf_conf

    # Fallback heuristic when RF model is unavailable.
    delta = score - threshold
    if delta > threshold * 0.9:
        label = "DoS"
    elif delta > threshold * 0.6:
        label = "Fuzzy"
    elif delta > threshold * 0.35:
        label = "Gear"
    else:
        label = "RPM"

    confidence = min(0.99, 0.55 + delta / max(threshold, 1e-8) * 0.35)
    return label, float(confidence)


def predict_from_tokens(tokens: Sequence[int], unknown_count: int, return_details: bool) -> Dict[str, Any]:
    start = time.perf_counter()
    threshold = float(runtime_state.threshold_data.get("threshold", 0.0))

    token_tensor = torch.tensor([tokens], dtype=torch.long)
    anomaly_score = float(score_token_tensor(token_tensor)[0])

    attack_type, confidence = classify_attack(anomaly_score, tokens, threshold)
    is_attack = anomaly_score > threshold
    label = "ATTACK" if is_attack else "NORMAL"

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    details = None
    if return_details:
        details = {
            "unknown_tokens": int(unknown_count),
            "unknown_ratio": round(unknown_count / WINDOW_SIZE, 6),
            "score_margin": round(anomaly_score - threshold, 6),
            "device": str(DEVICE),
        }

    result = {
        "anomaly_score": round(anomaly_score, 6),
        "threshold": round(threshold, 6),
        "label": label,
        "attack_type": attack_type,
        "confidence": round(float(confidence), 4),
        "is_attack": bool(is_attack),
        "processing_time_ms": round(elapsed_ms, 3),
        "details": details,
    }
    return result


def get_next_demo_tokens() -> List[int]:
    if runtime_state.demo_sequences is not None and len(runtime_state.demo_sequences) > 0:
        idx = runtime_state.demo_index % len(runtime_state.demo_sequences)
        runtime_state.demo_index += 1
        return runtime_state.demo_sequences[idx].tolist()

    vocab_values = [value for value in runtime_state.vocab.values() if value > UNK_ID]
    if not vocab_values:
        return [UNK_ID] * WINDOW_SIZE

    sampled = np.random.choice(vocab_values, size=WINDOW_SIZE, replace=True)
    return sampled.astype(np.int64).tolist()


def decode_tokens_to_can_ids(tokens: Sequence[int]) -> List[str]:
    reverse_vocab = {int(value): str(can_id) for can_id, value in runtime_state.vocab.items()}
    decoded: List[str] = []

    for token in tokens:
        token_int = int(token)
        decoded.append(reverse_vocab.get(token_int, format(token_int, "04x")))

    return decoded


# ---------------------------------------------------------------------------
# PART C — Startup Preflight Check
# Checks dataset files and model files listed in backend/config.py.
# Prints clear messages naming each missing file and the command to create it,
# and allows server startup without crashing.
# ---------------------------------------------------------------------------

_DATASET_PREFLIGHT = [
    (
        settings.VOCAB_PATH,
        "Vocabulary file (vocab.json)",
        "python -m src.preprocess",
    ),
    (
        settings.SEQUENCES_PATH,
        "Sequences tensor (sequences.pt)",
        "python -m src.preprocess",
    ),
    (
        settings.NORMAL_DATASET_PATH,
        "Normal driving dataset (normal_traffic_augmented.csv)",
        "Prepare normal traffic data in data/ folder",
    ),
    (
        settings.DOS_DATASET_PATH,
        "DoS attack dataset (DoS_dataset.csv)",
        "Extract Car Hacking DoS dataset to data/DoS_dataset.csv",
    ),
    (
        settings.FUZZY_DATASET_PATH,
        "Fuzzy attack dataset (Fuzzy_dataset.csv)",
        "Extract Car Hacking Fuzzy dataset to data/Fuzzy_dataset.csv",
    ),
    (
        settings.GEAR_DATASET_PATH,
        "Gear spoofing dataset (gear_dataset.csv)",
        "Extract Car Hacking Gear dataset to data/gear_dataset.csv",
    ),
    (
        settings.RPM_DATASET_PATH,
        "RPM spoofing dataset (RPM_dataset.csv)",
        "Extract Car Hacking RPM dataset to data/RPM_dataset.csv",
    ),
]

_MODEL_PREFLIGHT = [
    (
        settings.MODEL_PATH,
        "DistilBERT model checkpoint (best_model.pt)",
        "python -m src.train",
    ),
    (
        settings.THRESHOLD_PATH,
        "Detection threshold file (threshold.json)",
        "python -m src.evaluate",
    ),
    (
        settings.MULTICLASS_MODEL_PATH,
        "Multiclass attack classifier (multiclass_model.pkl)",
        "python -m src.train_multiclass",
    ),
    (
        settings.MULTICLASS_VECTORIZER_PATH,
        "Multiclass vectorizer (vectorizer.pkl)",
        "python -m src.train_multiclass",
    ),
]


def preflight_check() -> bool:
    """
    Check dataset and model files listed in backend/config.py.
    Prints status for every item, naming missing files and the exact command
    to create them. Never crashes the server.
    Returns True if core runtime files are present, False otherwise.
    """
    banner = "=" * 68
    print(f"\n{banner}")
    print("  STARF -- Smart Traffic Anomaly Recognition Kit")
    print("  Startup Preflight Check (backend/config.py file inventory)")
    print(f"{banner}")

    print("\n  [DATASET FILES]")
    dataset_missing = 0
    for path, label, cmd in _DATASET_PREFLIGHT:
        if path.exists():
            print(f"  [OK] Found: {label}")
            print(f"       Path   : {path}")
        else:
            dataset_missing += 1
            print(f"  [MISSING] {label}")
            print(f"       Path   : {path}")
            print(f"       Command: {cmd}")

    print("\n  [MODEL FILES]")
    model_missing = 0
    for path, label, cmd in _MODEL_PREFLIGHT:
        if path.exists():
            print(f"  [OK] Found: {label}")
            print(f"       Path   : {path}")
        else:
            model_missing += 1
            print(f"  [MISSING] {label}")
            print(f"       Path   : {path}")
            print(f"       Command: {cmd}")

    core_ready = settings.VOCAB_PATH.exists() and settings.MODEL_PATH.exists() and settings.THRESHOLD_PATH.exists()

    print(f"\n{'-' * 68}")
    if core_ready:
        print("  [STATUS] All core inference assets are available. Loading models...")
    else:
        print("  [STATUS] Missing required assets for real-time inference.")
        print("           Server will boot in DEMO / STANDBY mode without crashing.")
        print("           Run the commands above to generate missing assets.")
    print(f"{banner}\n")
    return core_ready



@asynccontextmanager
async def lifespan(_: FastAPI):

    preflight_check()
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        sync_report_artifacts()
        load_runtime_assets()
    except FileNotFoundError as exc:
        runtime_state.loaded = False
        runtime_state.model = None
        runtime_state.vocab = {}
        runtime_state.threshold_data = {}
        runtime_state.rf_classifier = None
        runtime_state.rf_meta = {}
        runtime_state.demo_sequences = None
        runtime_state.demo_index = 0
        # Still try the hybrid pipeline independently
        if _PIPELINE_AVAILABLE:
            try:
                runtime_state.pipeline = HybridPipeline()
                runtime_state.pipeline_loaded = True
                print("[STARF] HybridPipeline loaded in standalone mode.")
            except Exception:
                pass
        print(f"[STARF] Startup warning — model assets not available: {exc}")
        print("[STARF] The dashboard will run in demo mode until you train and evaluate the model.")
    except Exception as exc:  # pragma: no cover - startup hard failure path
        raise RuntimeError(f"STARF backend startup failed: {exc}") from exc

    yield

    if runtime_state.model is not None:
        del runtime_state.model
        runtime_state.model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="STARF — Smart Traffic Anomaly Recognition Kit",
    version="1.0.0",
    description="FastAPI backend for the STARF hybrid CAN bus intrusion detection system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/reports", StaticFiles(directory=str(REPORTS_DIR)), name="static-reports")


@app.get("/")
async def root() -> Dict[str, Any]:
    pipeline_loaded = runtime_state.pipeline_loaded and runtime_state.pipeline is not None
    return {
        "status": "online",
        "project": "STARF — Smart Traffic Anomaly Recognition Kit",
        "model": "Random Forest + rules" if pipeline_loaded else ("DistilBERT" if runtime_state.loaded else "not_loaded"),
        "pipeline_loaded": pipeline_loaded,
        "device": str(DEVICE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/status")
async def health() -> Dict[str, Any]:
    pipeline_loaded = runtime_state.pipeline_loaded and runtime_state.pipeline is not None
    is_online = pipeline_loaded or runtime_state.loaded

    if not is_online:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model and pipeline not loaded")

    # Load model name from multiclass_meta.json if present
    model_name = "Random Forest"
    meta_path = MODELS_DIR / "multiclass_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                mtype = meta.get("model_type", "")
                if "RandomForest" in mtype:
                    model_name = "Random Forest"
                elif mtype:
                    model_name = mtype.split("(")[0]
        except Exception:
            pass

    rules_count = 0
    shap_available = False
    if pipeline_loaded:
        rules_count = len(runtime_state.pipeline.whitelist) if hasattr(runtime_state.pipeline, "whitelist") else 0
        shap_available = getattr(runtime_state.pipeline, "shap_explainer", None) is not None

    return {
        "status": "online",
        "pipeline_loaded": pipeline_loaded,
        "model": "Random Forest + rules" if pipeline_loaded else "DistilBERT",
        "model_name": model_name,
        "rules_count": rules_count,
        "shap_available": shap_available,
        "device": str(DEVICE),
        "version": app.version,
    }


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    if not (runtime_state.loaded or runtime_state.pipeline_loaded):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    return stats_state.snapshot()


# ---------------------------------------------------------------------------
# /replay — Dataset replay from data/prepared.parquet (held-out test frames)
# ---------------------------------------------------------------------------
_REPLAY_TEST_CACHE: Dict[str, Any] = {}


def _get_replay_test_block(class_name: str) -> pd.DataFrame:
    global _REPLAY_TEST_CACHE
    if class_name in _REPLAY_TEST_CACHE:
        return _REPLAY_TEST_CACHE[class_name]

    prepared_path = DATA_DIR / "prepared.parquet"
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found at: {prepared_path}")

    df = pd.read_parquet(prepared_path)
    cls_df = df[df["label"] == class_name].sort_values("timestamp").reset_index(drop=True)
    if len(cls_df) == 0:
        raise ValueError(f"No frames found for class '{class_name}'.")

    # Chronological 80/20 held-out test block (last 20% in timestamp order)
    split_idx = int(len(cls_df) * 0.80)
    test_block = cls_df.iloc[split_idx:].reset_index(drop=True)
    _REPLAY_TEST_CACHE[class_name] = test_block
    return test_block


@app.get("/replay")
async def replay_frames(
    class_name: str = Query(default="Normal", alias="class"),
    start: int = Query(default=0, ge=0),
    count: int = Query(default=50, ge=1, le=1000),
) -> Dict[str, Any]:
    """Return real held-out test frames from data/prepared.parquet (last 20% split)."""
    valid_classes = ["Normal", "DoS", "Fuzzy", "Spoof_Gear", "Spoof_RPM"]
    if class_name not in valid_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class '{class_name}'. Must be one of: {valid_classes}",
        )

    try:
        test_block = _get_replay_test_block(class_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    total_test_frames = len(test_block)
    start_clamped = min(start, max(0, total_test_frames - 1)) if total_test_frames > 0 else 0
    end_clamped = min(start_clamped + count, total_test_frames)
    slice_df = test_block.iloc[start_clamped:end_clamped]

    def _parse_b(val: Any) -> int:
        if pd.isna(val) or val == "" or val is None:
            return 0
        if isinstance(val, (int, np.integer)):
            return int(val) & 0xFF
        val_str = str(val).strip().lower()
        if not val_str:
            return 0
        try:
            return int(val_str, 16) if len(val_str) <= 2 else int(val_str)
        except ValueError:
            return 0

    def _parse_cid(val: Any) -> int:
        if pd.isna(val) or val == "" or val is None:
            return 0
        if isinstance(val, (int, np.integer)):
            return int(val)
        val_str = str(val).strip()
        try:
            if val_str.startswith("0x") or val_str.startswith("0X"):
                return int(val_str, 16)
            return int(val_str)
        except ValueError:
            try:
                return int(val_str, 16)
            except ValueError:
                return 0

    frames = []
    for _, row in slice_df.iterrows():
        cid = _parse_cid(row["can_id"])
        data_bytes = [_parse_b(row.get(f"d{i}", 0)) for i in range(8)]
        frames.append({
            "timestamp": float(row["timestamp"]),
            "can_id": cid,
            "can_id_hex": f"0x{cid:03X}",
            "dlc": int(row["dlc"]) if not pd.isna(row.get("dlc")) else len(data_bytes),
            "data_bytes": data_bytes,
            "label": str(row["label"]),
            "simulated": False,
        })

    return {
        "class": class_name,
        "start": start_clamped,
        "count": len(frames),
        "total_test_frames": total_test_frames,
        "frames": frames,
    }


@app.get("/threshold")
async def get_threshold() -> Dict[str, Any]:
    return runtime_state.threshold_data or {"threshold": 0.75}


@app.post("/threshold")
async def update_threshold(payload: ThresholdUpdateRequest) -> Dict[str, Any]:
    if not runtime_state.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    runtime_state.threshold_data["threshold"] = float(payload.threshold)
    runtime_state.threshold_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    if payload.persist:
        try:
            save_threshold_data(THRESHOLD_PATH, runtime_state.threshold_data)
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist threshold: {exc}",
            ) from exc

    return {
        "threshold": runtime_state.threshold_data["threshold"],
        "persisted": payload.persist,
        "threshold_data": runtime_state.threshold_data,
    }


@app.get("/stream/config")
async def get_stream_config() -> Dict[str, Any]:
    return {
        "interval_ms": runtime_state.stream_interval_ms,
        "frames_per_second": round(1000.0 / runtime_state.stream_interval_ms, 3),
    }


@app.post("/stream/config")
async def update_stream_config(payload: StreamConfigUpdateRequest) -> Dict[str, Any]:
    runtime_state.stream_interval_ms = int(payload.interval_ms)
    return {
        "interval_ms": runtime_state.stream_interval_ms,
        "frames_per_second": round(1000.0 / runtime_state.stream_interval_ms, 3),
    }


@app.get("/vocab/size")
async def get_vocab_size() -> Dict[str, int]:
    if not runtime_state.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    return {"vocab_size": len(runtime_state.vocab)}


@app.get("/reports/manifest")
async def get_reports_manifest() -> Dict[str, Any]:
    sync_report_artifacts()

    expected = [
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "precision_recall_curve.png",
        "score_distribution.png",
        "metrics_summary.png",
        "model_evaluation_report.pdf",
        "multiclass_confusion_matrix.png",
        "multiclass_report.pdf",
    ]

    available = []
    for name in expected:
        path = REPORTS_DIR / name
        if path.exists():
            available.append(
                {
                    "name": name,
                    "url": f"/static/reports/{name}",
                    "size_bytes": path.stat().st_size,
                }
            )

    metrics = {}
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)

    multiclass_metrics = {}
    multiclass_path = REPORTS_DIR / "multiclass_metrics.json"
    if multiclass_path.exists():
        multiclass_metrics = load_json(multiclass_path)

    return {
        "available_files": available,
        "metrics": metrics,
        "multiclass_metrics": multiclass_metrics,
    }


# Shared REST session state (one StreamState + RuleEngine per server process)
_rest_stream_state: Optional[Any] = None  # StreamState | None
_rest_rule_engine: Optional[Any] = None   # RuleEngine | None


def _get_rest_pipeline_state():
    """Lazily create and return the shared REST session state objects."""
    global _rest_stream_state, _rest_rule_engine
    if runtime_state.pipeline is not None and _rest_stream_state is None:
        _rest_stream_state = runtime_state.pipeline.new_stream_state()
        _rest_rule_engine = runtime_state.pipeline.new_rule_state()
    return _rest_stream_state, _rest_rule_engine


def _run_hybrid_prediction(
    timestamp: float,
    can_id: int,
    dlc: int = 8,
    data_bytes: Optional[List[int]] = None,
    simulated: bool = False,
) -> Dict[str, Any]:
    """Common HybridPipeline runner for REST endpoints."""
    if not runtime_state.pipeline_loaded or runtime_state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HybridPipeline not loaded. Run 'python -m src.train_multiclass' first.",
        )

    stream_state, rule_engine = _get_rest_pipeline_state()
    result = runtime_state.pipeline.predict_frame(
        timestamp=timestamp,
        can_id=can_id,
        dlc=dlc,
        data_bytes=data_bytes or [0] * 8,
        state=stream_state,
        rule_engine=rule_engine,
        simulated=simulated,
    )
    stats_state.update(result)
    return result


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    """REST prediction endpoint supporting both raw CAN frames and sequences.

    Unified schema includes: decision, stage, attack_type, confidence, rule_id,
    reason, severity, simulated, and top_features.
    
    Legacy fields note:
      anomaly_score and threshold are legacy fields maintained for backwards
      compatibility with earlier dashboard clients and components.

    When HybridPipeline is loaded:
      - Raw CAN frame inputs (timestamp, can_id) run through the hybrid pipeline.
      - Legacy sequence inputs are streamed through the hybrid pipeline.
    Fallback:
      - If HybridPipeline is not loaded, legacy DistilBERT token scoring is used.
    """
    # 1. Frame-based prediction (timestamp or can_id provided)
    if payload.can_id is not None or (payload.sequence is None and payload.timestamp is not None):
        cid = payload.can_id if payload.can_id is not None else 0
        if isinstance(cid, str):
            cid = int(cid, 16) if cid.startswith(("0x", "0X")) or not cid.isdigit() else int(cid)
        ts = payload.timestamp if payload.timestamp is not None else time.time()
        dbytes = payload.data_bytes if payload.data_bytes is not None else [0] * 8
        result = _run_hybrid_prediction(
            timestamp=ts,
            can_id=cid,
            dlc=payload.dlc if payload.dlc is not None else 8,
            data_bytes=dbytes,
            simulated=payload.simulated or False,
        )
        return PredictResponse(**result)

    # 2. Sequence-based prediction (legacy sequence provided)
    if payload.sequence is not None:
        if runtime_state.pipeline_loaded and runtime_state.pipeline is not None:
            last_res = None
            base_ts = time.time() - len(payload.sequence) * 0.001
            for i, token_str in enumerate(payload.sequence):
                try:
                    cid = int(token_str, 16) if token_str.startswith(("0x", "0X")) or not token_str.isdigit() else int(token_str)
                except ValueError:
                    cid = 0
                ts = base_ts + i * 0.001
                last_res = _run_hybrid_prediction(
                    timestamp=ts,
                    can_id=cid,
                    dlc=payload.dlc if payload.dlc is not None else 8,
                    data_bytes=[0] * 8,
                    simulated=payload.simulated or False,
                )
            if last_res:
                return PredictResponse(**last_res)

        # Fallback to DistilBERT legacy path only when hybrid pipeline is not loaded
        if runtime_state.loaded:
            try:
                tokens, unknown_count = tokenize_sequence(payload.sequence)
                result = predict_from_tokens(tokens, unknown_count, payload.return_details)
                stats_state.update(result)
                return PredictResponse(**result)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neither HybridPipeline nor DistilBERT model is loaded.",
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Request must contain either frame fields ('timestamp', 'can_id') or 'sequence'.",
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    if not runtime_state.loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    predictions: List[PredictResponse] = []

    try:
        for sequence in payload.sequences:
            tokens, unknown_count = tokenize_sequence(sequence)
            result = predict_from_tokens(tokens, unknown_count, payload.return_details)
            stats_state.update(result)
            predictions.append(PredictResponse(**result))

        attacks = sum(1 for item in predictions if item.is_attack)
        normals = len(predictions) - attacks
        average_score = float(np.mean([item.anomaly_score for item in predictions])) if predictions else 0.0

        return BatchPredictResponse(
            total=len(predictions),
            attacks=attacks,
            normals=normals,
            average_score=round(average_score, 6),
            predictions=predictions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# /predict/frame  — Hybrid pipeline endpoint (rules + ML multiclass)
# Accepts a single raw CAN frame and returns the unified pipeline response.
# ---------------------------------------------------------------------------

class FramePredictRequest(BaseModel):
    timestamp: float = Field(..., description="Frame arrival timestamp in seconds")
    can_id: Union[int, str] = Field(..., description="CAN arbitration ID (integer or hex string)")
    dlc: int = Field(default=8, ge=0, le=8, description="Data Length Code")
    data_bytes: List[int] = Field(
        default_factory=lambda: [0] * 8,
        min_length=0,
        max_length=8,
        description="Up to 8 payload bytes [0-255]",
    )
    simulated: bool = Field(default=False, description="Tag response as simulated data")

    @field_validator("can_id", mode="before")
    @classmethod
    def parse_can_id(cls, v: Any) -> int:
        if isinstance(v, str):
            return int(v, 16) if v.startswith(("0x", "0X")) or not v.isdigit() else int(v)
        return int(v)


@app.post("/predict/frame")
async def predict_frame(payload: FramePredictRequest) -> Dict[str, Any]:
    """Hybrid pipeline prediction for a single raw CAN bus frame."""
    return _run_hybrid_prediction(
        timestamp=payload.timestamp,
        can_id=payload.can_id,
        dlc=payload.dlc,
        data_bytes=payload.data_bytes,
        simulated=payload.simulated,
    )


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Live streaming WebSocket endpoint.

    When HybridPipeline is loaded: uses per-connection StreamState + RuleEngine
    for fully isolated streaming inference.
    Fallback: existing DistilBERT token scoring path.
    Both paths emit the same JSON schema to the frontend.
    """
    await websocket.accept()

    # Per-connection state for the hybrid pipeline
    conn_stream_state: Optional[Any] = None
    conn_rule_engine: Optional[Any] = None
    use_pipeline = runtime_state.pipeline_loaded and runtime_state.pipeline is not None

    if use_pipeline:
        conn_stream_state = runtime_state.pipeline.new_stream_state()
        conn_rule_engine  = runtime_state.pipeline.new_rule_state()

    try:
        while True:
            if use_pipeline:
                # Check for incoming client frame or timeout to send next autonomous stream frame
                incoming_frame = None
                try:
                    raw_text = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=runtime_state.stream_interval_ms / 1000.0,
                    )
                    if raw_text:
                        try:
                            incoming_frame = json.loads(raw_text)
                        except Exception:
                            incoming_frame = None
                except asyncio.TimeoutError:
                    pass

                if incoming_frame and isinstance(incoming_frame, dict):
                    ts = float(incoming_frame.get("timestamp", time.time()))
                    raw_cid = incoming_frame.get("can_id", 0)
                    if isinstance(raw_cid, str):
                        try:
                            cid = int(raw_cid, 16) if raw_cid.startswith(("0x", "0X")) or not raw_cid.isdigit() else int(raw_cid)
                        except ValueError:
                            cid = 0
                    else:
                        cid = int(raw_cid)
                    dlc = int(incoming_frame.get("dlc", 8))
                    dbytes = incoming_frame.get("data_bytes", [0] * 8)
                    sim = bool(incoming_frame.get("simulated", False))
                else:
                    tokens = get_next_demo_tokens()
                    ts = time.time()
                    raw_ids = decode_tokens_to_can_ids(tokens)
                    mid = len(raw_ids) // 2
                    try:
                        cid = int(raw_ids[mid], 16)
                    except (ValueError, TypeError):
                        cid = tokens[mid]
                    dlc = 8
                    dbytes = [0] * 8
                    sim = True

                result = runtime_state.pipeline.predict_frame(
                    timestamp=ts,
                    can_id=cid,
                    dlc=dlc,
                    data_bytes=dbytes,
                    state=conn_stream_state,
                    rule_engine=conn_rule_engine,
                    simulated=sim,
                )
                stats_state.update(result)

                payload_out = {
                    # Frame fields
                    "can_id":          cid,
                    "can_id_hex":      f"0x{cid:03X}",
                    "dlc":             dlc,
                    "data_bytes":      dbytes,
                    # Legacy fields (frontend compatibility)
                    "score":           result["anomaly_score"],
                    "is_attack":       result["is_attack"],
                    "label":           result["label"],
                    "attack_type":     result["attack_type"],
                    "confidence":      result["confidence"],
                    "threshold":       result["threshold"],
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                    "pipeline_path":   result["pipeline_path"],
                    "rules_triggered": result["rules_triggered"],
                    # New unified fields
                    "decision":        result["decision"],
                    "stage":           result["stage"],
                    "rule_id":         result["rule_id"],
                    "reason":          result["reason"],
                    "severity":        result["severity"],
                    "simulated":       result["simulated"],
                    "top_features":    result["top_features"],
                }
            else:
                # ---- Legacy DistilBERT path (only when pipeline not loaded) ----
                tokens = get_next_demo_tokens()
                unknown_count = int(np.sum(np.asarray(tokens) == UNK_ID))
                result = predict_from_tokens(tokens, unknown_count, return_details=False)

                if result["is_attack"]:
                    raw_can_ids = decode_tokens_to_can_ids(tokens)
                    result["attack_type"] = predict_attack_type(tokens, raw_can_ids)
                else:
                    result["attack_type"] = None

                stats_state.update(result)

                payload_out = {
                    "score":           result["anomaly_score"],
                    "is_attack":       result["is_attack"],
                    "label":           result["label"],
                    "attack_type":     result["attack_type"],
                    "confidence":      result["confidence"],
                    "threshold":       result["threshold"],
                    "timestamp":       datetime.now(timezone.utc).isoformat(),
                    "pipeline_path":   "legacy",
                    "rules_triggered": [],
                    # New fields with safe defaults
                    "decision":        "attack" if result["is_attack"] else "normal",
                    "stage":           "legacy",
                    "rule_id":         None,
                    "reason":          None,
                    "severity":        None,
                    "simulated":       True,
                    "top_features":    [],
                }

            await websocket.send_json(payload_out)
            await asyncio.sleep(runtime_state.stream_interval_ms / 1000.0)

    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            return


@app.exception_handler(Exception)
async def global_exception_handler(_, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": type(exc).__name__})


if __name__ == "__main__":
    uvicorn.run("backend.main:app", **UVICORN_CONFIG)
