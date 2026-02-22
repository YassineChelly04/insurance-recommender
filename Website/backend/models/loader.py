"""
Model loader — singleton pattern.
The artifact is loaded once at application startup and reused for every request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    """
    Typed wrapper around the joblib artifact produced by the training notebook.

    The artifact dict contains:
        model          — trained XGBClassifier
        feature_names  — list[str] of column names expected by the model
        label_encoder  — sklearn LabelEncoder (maps encoded int → original class)
        region_map     — dict[str, float] for target-encoding Region_Code
        global_mean    — float fallback for unseen regions
        medians        — dict[str, float] for numeric imputation
        cat_cols       — list[str] of one-hot-encoded categorical columns
        num_classes    — int (10)
    """

    model: Any
    feature_names: list[str]
    label_encoder: Any
    region_map: dict[str, float]
    global_mean: float
    medians: dict[str, float]
    cat_cols: list[str]
    num_classes: int

    version: str = field(default="1.0.0", repr=False)


# Module-level singleton
_artifact: ModelArtifact | None = None


def load_model(model_path: Path) -> ModelArtifact:
    """
    Load the model artifact from disk.
    Idempotent — subsequent calls return the cached instance.
    """
    global _artifact

    if _artifact is not None:
        logger.debug("Model already loaded — returning cached artifact.")
        return _artifact

    logger.info("Loading model artifact from: %s", model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"model.joblib not found at '{model_path}'. "
            "Place the trained artifact at that path or set MODEL_PATH in .env."
        )

    raw: dict = joblib.load(model_path)

    _artifact = ModelArtifact(
        model=raw["model"],
        feature_names=raw["feature_names"],
        label_encoder=raw["label_encoder"],
        region_map=raw["region_map"],
        global_mean=float(raw["global_mean"]),
        medians=raw["medians"],
        cat_cols=raw["cat_cols"],
        num_classes=int(raw["num_classes"]),
    )

    logger.info(
        "Model loaded — %d features, %d classes.",
        len(_artifact.feature_names),
        _artifact.num_classes,
    )
    return _artifact


def get_artifact() -> ModelArtifact:
    """FastAPI dependency — returns already-loaded artifact."""
    if _artifact is None:
        raise RuntimeError("Model has not been loaded yet. Call load_model() at startup.")
    return _artifact
