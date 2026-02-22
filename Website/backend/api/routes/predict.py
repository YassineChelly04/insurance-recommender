"""
POST /predict — Run model inference on a customer profile.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from backend.schemas.customer import CustomerProfile, PredictionResponse
from backend.services.inference import get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict insurance bundle",
    description=(
        "Submit a customer profile and receive the predicted optimal insurance bundle "
        "alongside top-3 recommendations with confidence scores."
    ),
    tags=["Inference"],
)
async def predict(profile: CustomerProfile) -> PredictionResponse:
    """
    Run the full ML pipeline on the submitted customer profile.

    - Validates input via Pydantic
    - Preprocesses features (mirrors training pipeline)
    - Runs XGBoost inference
    - Returns predicted bundle + top-3 with confidence
    """
    t0 = time.perf_counter()

    try:
        service = get_inference_service()
        result = service.predict(profile)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Prediction: bundle=%d confidence=%.3f latency=%.1fms",
        result.predicted_bundle,
        result.confidence,
        elapsed_ms,
    )
    return result
