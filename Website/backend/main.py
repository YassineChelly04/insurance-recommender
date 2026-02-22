"""
FastAPI application entry point.

Startup sequence:
  1. Configure logging
  2. Load model artifact from disk (once)
  3. Create inference service singleton
  4. Mount API router

Run locally:
    cd Website
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.router import api_router
from backend.core.config import settings
from backend.core.logging_config import configure_logging
from backend.models.loader import load_model
from backend.services.inference import create_inference_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — load model on startup, nothing on shutdown."""
    configure_logging()
    logger.info("Starting %s v%s", settings.app_title, settings.app_version)

    artifact = load_model(settings.model_path)
    create_inference_service(artifact)

    logger.info("Application ready — %d feature model loaded.", len(artifact.feature_names))
    yield
    logger.info("Shutting down application.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description=settings.app_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Routes ─────────────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


app = create_app()
