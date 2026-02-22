"""
GET /health — Liveness and readiness check.
"""

from fastapi import APIRouter

from backend.core.config import settings
from backend.models.loader import get_artifact
from backend.schemas.customer import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Returns the API status and model load state. Use this to verify the service is ready.",
    tags=["Health"],
)
async def health() -> HealthResponse:
    """Liveness + readiness check — returns 200 when the model is loaded."""
    try:
        artifact = get_artifact()
        model_loaded = True
        num_classes = artifact.num_classes
    except RuntimeError:
        model_loaded = False
        num_classes = 0

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_version="1.0.0",
        num_classes=num_classes,
        api_version=settings.app_version,
    )
