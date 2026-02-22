"""
Inference service.
Runs model prediction with optional in-memory LRU caching.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from backend.core.config import settings
from backend.schemas.customer import BUNDLE_NAMES, BundleScore, PredictionResponse
from backend.services.preprocessor import preprocess

if TYPE_CHECKING:
    from backend.models.loader import ModelArtifact
    from backend.schemas.customer import CustomerProfile

logger = logging.getLogger(__name__)


def _profile_cache_key(profile: "CustomerProfile") -> str:
    """Deterministic cache key derived from the profile JSON."""
    return hashlib.sha256(
        profile.model_dump_json(exclude_none=False).encode()
    ).hexdigest()


def _key_factors_from_profile(profile: "CustomerProfile") -> list[str]:
    """
    Generate human-readable explanation strings based on dominant profile attributes.
    This is a lightweight rule-based explainer (no SHAP dependency).
    """
    factors: list[str] = []

    total_deps = profile.adult_dependents + profile.child_dependents + profile.infant_dependents
    if total_deps >= 3:
        factors.append(f"Large family ({int(total_deps)} dependents) suggests comprehensive coverage need")
    elif total_deps == 0:
        factors.append("No dependents — individual coverage prioritized")

    if profile.estimated_annual_income >= 90000:
        factors.append("High income bracket — premium bundles are accessible")
    elif profile.estimated_annual_income < 30000:
        factors.append("Lower income bracket — cost-effective bundles preferred")

    if profile.broker_id is not None:
        factors.append("Broker-assisted purchase — agent-recommended bundle weighted more")

    if profile.years_without_claims and profile.years_without_claims >= 5:
        factors.append(f"{int(profile.years_without_claims)} years claim-free — low-risk profile")

    if profile.vehicles_on_policy and profile.vehicles_on_policy >= 2:
        factors.append(f"{int(profile.vehicles_on_policy)} vehicles on policy — multi-vehicle discount bundle")

    if profile.deductible_tier:
        factors.append(f"Deductible preference: {profile.deductible_tier.replace('_', ' ')}")

    return factors[:4]  # Keep top 4 for display


class InferenceService:
    """
    Stateless service that wraps preprocessing + model inference.
    An LRU cache is applied at the instance level to avoid re-computing
    identical predictions for the same input.
    """

    def __init__(self, artifact: "ModelArtifact") -> None:
        self.artifact = artifact
        self._cache: dict[str, PredictionResponse] = {}
        self._cache_max = settings.cache_max_size
        self._cache_enabled = settings.cache_enabled

    def predict(self, profile: "CustomerProfile") -> PredictionResponse:
        cache_key = _profile_cache_key(profile)

        if self._cache_enabled and cache_key in self._cache:
            logger.debug("Cache HIT for key %.8s…", cache_key)
            return self._cache[cache_key]

        # ── Preprocess ─────────────────────────────────────────────────────
        X = preprocess(profile, self.artifact)

        # ── Inference ──────────────────────────────────────────────────────
        proba: np.ndarray = self.artifact.model.predict_proba(X.values)[0]  # shape (10,)

        # label_encoder maps encoded int 0..N-1 → original class labels 0..9
        le = self.artifact.label_encoder
        classes = le.classes_  # e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Sort by probability descending
        sorted_idx = np.argsort(proba)[::-1]

        top_bundle_encoded = int(sorted_idx[0])
        top_bundle_original = int(le.inverse_transform([top_bundle_encoded])[0])
        top_confidence = float(proba[sorted_idx[0]])

        top_3 = [
            BundleScore(
                bundle_id=int(le.inverse_transform([int(sorted_idx[i])])[0]),
                bundle_name=BUNDLE_NAMES.get(int(le.inverse_transform([int(sorted_idx[i])])[0]), f"Bundle {i}"),
                confidence=round(float(proba[sorted_idx[i]]), 4),
            )
            for i in range(min(3, len(sorted_idx)))
        ]

        key_factors = _key_factors_from_profile(profile)

        response = PredictionResponse(
            predicted_bundle=top_bundle_original,
            predicted_bundle_name=BUNDLE_NAMES.get(top_bundle_original, f"Bundle {top_bundle_original}"),
            confidence=round(top_confidence, 4),
            top_3=top_3,
            key_factors=key_factors,
            model_version="1.0.0",
        )

        # ── Cache eviction (simple FIFO when full) ─────────────────────────
        if self._cache_enabled:
            if len(self._cache) >= self._cache_max:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[cache_key] = response
            logger.debug("Cache MISS — stored key %.8s… (size=%d)", cache_key, len(self._cache))

        return response


# Module-level singleton (created after model load)
_service: InferenceService | None = None


def create_inference_service(artifact: "ModelArtifact") -> InferenceService:
    global _service
    _service = InferenceService(artifact)
    return _service


def get_inference_service() -> InferenceService:
    if _service is None:
        raise RuntimeError("InferenceService has not been initialised. Call create_inference_service() at startup.")
    return _service
