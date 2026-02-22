"""Router aggregator — collects all route modules into a single APIRouter."""

from fastapi import APIRouter

from backend.api.routes import health, predict

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(predict.router)
