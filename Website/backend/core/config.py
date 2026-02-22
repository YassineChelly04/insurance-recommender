"""
Application configuration — driven by environment variables or .env file.
All settings can be overridden at runtime without changing source code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),  # Silence model_path namespace warning
    )

    # ── API metadata ────────────────────────────────────────────────────────
    app_title: str = "Insurance Bundle Recommendation API"
    app_version: str = "1.0.0"
    app_description: str = (
        "Production-ready ML API that predicts optimal insurance bundle "
        "recommendations for customer profiles."
    )

    # ── Server ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    # ── Model ────────────────────────────────────────────────────────────────
    # Resolved relative to the project root at runtime
    model_path: Path = Path(__file__).resolve().parents[3] / "model.joblib"

    # ── Caching ──────────────────────────────────────────────────────────────
    cache_enabled: bool = True
    cache_max_size: int = 512   # maximum number of cached predictions


# Module-level singleton — imported everywhere
settings = Settings()
