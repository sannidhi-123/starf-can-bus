"""Configuration for backend runtime paths and environment overrides."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-backed settings for backend model asset paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_ROOT: Path = PROJECT_ROOT

    # Dataset files (data/ directory)
    VOCAB_PATH: Path = Field(default=PROJECT_ROOT / "data" / "vocab.json")
    SEQUENCES_PATH: Path = Field(default=PROJECT_ROOT / "data" / "sequences.pt")
    NORMAL_DATASET_PATH: Path = Field(default=PROJECT_ROOT / "data" / "normal_traffic_augmented.csv")
    DOS_DATASET_PATH: Path = Field(default=PROJECT_ROOT / "data" / "DoS_dataset.csv")
    FUZZY_DATASET_PATH: Path = Field(default=PROJECT_ROOT / "data" / "Fuzzy_dataset.csv")
    GEAR_DATASET_PATH: Path = Field(default=PROJECT_ROOT / "data" / "gear_dataset.csv")
    RPM_DATASET_PATH: Path = Field(default=PROJECT_ROOT / "data" / "RPM_dataset.csv")

    # Model files (models/ directory)
    MODEL_PATH: Path = Field(default=PROJECT_ROOT / "models" / "best_model.pt")
    THRESHOLD_PATH: Path = Field(default=PROJECT_ROOT / "models" / "threshold.json")
    MULTICLASS_MODEL_PATH: Path = Field(default=PROJECT_ROOT / "models" / "multiclass_model.pkl")
    MULTICLASS_VECTORIZER_PATH: Path = Field(default=PROJECT_ROOT / "models" / "vectorizer.pkl")
    XGB_MODEL_PATH: Path = Field(default=PROJECT_ROOT / "models" / "xgb_model.pkl")
    XGB_META_PATH: Path = Field(default=PROJECT_ROOT / "models" / "xgb_meta.json")

    @field_validator(
        "VOCAB_PATH",
        "SEQUENCES_PATH",
        "NORMAL_DATASET_PATH",
        "DOS_DATASET_PATH",
        "FUZZY_DATASET_PATH",
        "GEAR_DATASET_PATH",
        "RPM_DATASET_PATH",
        "MODEL_PATH",
        "THRESHOLD_PATH",
        "MULTICLASS_MODEL_PATH",
        "MULTICLASS_VECTORIZER_PATH",
        "XGB_MODEL_PATH",
        "XGB_META_PATH",
        mode="before",
    )
    @classmethod
    def resolve_project_relative_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


settings = Settings()


# ---------------------------------------------------------------------------
# MODEL_BACKEND: "xgboost" | "random_forest"
# Default to xgboost when models/xgb_model.pkl exists, else fall back to RF.
# ---------------------------------------------------------------------------

def get_model_backend() -> str:
    """Return 'xgboost' if xgb_model.pkl exists, else 'random_forest'."""
    if settings.XGB_MODEL_PATH.exists():
        return "xgboost"
    logging.getLogger("starf").warning(
        "XGB model not found at %s — falling back to random_forest",
        settings.XGB_MODEL_PATH,
    )
    return "random_forest"


MODEL_BACKEND: str = get_model_backend()

# ---------------------------------------------------------------------------
# Severity classification (Part B)
# Maps attack/rule types to severity levels: High, Medium, None
# ---------------------------------------------------------------------------
SEVERITY_MAP: dict = {
    "DoS":        "High",
    "Fuzzy":      "Medium",
    "Spoof_Gear": "High",
    "Spoof_RPM":  "High",
    "Normal":     None,
}

# Rule-triggered severity by rule_id
RULE_SEVERITY_MAP: dict = {
    "INVALID_ID": "Medium",
    "FREQUENCY":  "High",
    "TIMING":     "Medium",
}
