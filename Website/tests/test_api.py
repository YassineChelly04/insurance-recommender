"""
API integration tests.
Uses httpx AsyncClient with the FastAPI ASGI app directly — no running server needed.

Run:
    cd Website
    pytest tests/ -v
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def client():
    """Shared AsyncClient for the entire test session."""
    from backend.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Minimal valid payload ───────────────────────────────────────────────────────
VALID_PAYLOAD = {
    "estimated_annual_income": 75000,
    "adult_dependents": 1,
    "child_dependents": 2,
    "infant_dependents": 0,
    "previous_policy_duration_months": 24,
    "grace_period_extensions": 1,
    "years_without_claims": 3,
    "policy_amendments_count": 2,
    "vehicles_on_policy": 1,
    "custom_riders_requested": 1,
    "days_since_quote": 10,
    "policy_start_month": "March",
    "policy_start_year": 2024,
    "policy_start_week": 12,
    "broker_agency_type": "National_Corporate",
    "acquisition_channel": "Direct_Website",
    "payment_schedule": "Monthly",
    "employment_status": "Employed_FullTime",
    "region_code": "US-CA",
    "deductible_tier": "Tier_2_Mid_Ded",
}


@pytest.mark.anyio
async def test_health_ok(client):
    """GET /health should return 200 with model_loaded=True."""
    res = await client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["num_classes"] == 10


@pytest.mark.anyio
async def test_predict_valid(client):
    """POST /predict with a full valid payload should return a prediction."""
    res = await client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200, f"Unexpected status: {res.text}"
    body = res.json()

    assert "predicted_bundle" in body
    assert isinstance(body["predicted_bundle"], int)
    assert 0 <= body["predicted_bundle"] <= 9

    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0

    assert "top_3" in body
    assert len(body["top_3"]) == 3

    assert "key_factors" in body
    assert isinstance(body["key_factors"], list)


@pytest.mark.anyio
async def test_predict_missing_optional_fields(client):
    """POST /predict with only required fields should succeed (graceful defaults)."""
    minimal = {
        "estimated_annual_income": 40000,
        "adult_dependents": 0,
        "policy_start_month": "January",
    }
    res = await client.post("/predict", json=minimal)
    assert res.status_code == 200
    body = res.json()
    assert "predicted_bundle" in body


@pytest.mark.anyio
async def test_predict_invalid_income(client):
    """POST /predict with negative income should fail Pydantic validation."""
    bad = dict(VALID_PAYLOAD)
    bad["estimated_annual_income"] = -1
    res = await client.post("/predict", json=bad)
    assert res.status_code == 422


@pytest.mark.anyio
async def test_predict_caching(client):
    """Same payload submitted twice should return identical results (cache hit)."""
    res1 = await client.post("/predict", json=VALID_PAYLOAD)
    res2 = await client.post("/predict", json=VALID_PAYLOAD)
    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["predicted_bundle"] == res2.json()["predicted_bundle"]
    assert res1.json()["confidence"] == res2.json()["confidence"]
