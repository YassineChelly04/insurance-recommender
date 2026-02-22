"""
Pydantic schemas for request validation and response serialization.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Bundle label names (for human-readable output) ──────────────────────────
BUNDLE_NAMES: dict[int, str] = {
    0: "Basic Plan",
    1: "Essential Plan",
    2: "Standard Plan",
    3: "Enhanced Plan",
    4: "Premium Plan",
    5: "Elite Plan",
    6: "Family Plan",
    7: "Comprehensive Plan",
    8: "Enterprise Plan",
    9: "Ultimate Plan",
}


class CustomerProfile(BaseModel):
    """
    Customer profile submitted for bundle prediction.
    All optional fields default to sentinel values that trigger
    the same imputation logic used during training.
    """

    model_config = {"json_schema_extra": {"example": {
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
        "broker_id": None,
        "deductible_tier": "Tier_2_Mid_Ded",
    }}}

    # ── Numeric features ─────────────────────────────────────────────────────
    estimated_annual_income: float = Field(50000.0, ge=0, description="Customer's estimated annual income")
    adult_dependents: int = Field(0, ge=0, description="Number of adult dependents")
    child_dependents: float = Field(0.0, ge=0, description="Number of child dependents")
    infant_dependents: float = Field(0.0, ge=0, description="Number of infant dependents")
    previous_policy_duration_months: float = Field(12.0, ge=0, description="Duration of previous policy in months")
    grace_period_extensions: float = Field(0.0, ge=0, description="Number of grace period extensions used")
    years_without_claims: float = Field(1.0, ge=0, description="Years since last insurance claim")
    policy_amendments_count: float = Field(0.0, ge=0, description="Number of amendments made to policy")
    vehicles_on_policy: float = Field(1.0, ge=0, description="Number of vehicles covered")
    custom_riders_requested: float = Field(0.0, ge=0, description="Number of custom riders added")
    days_since_quote: float = Field(7.0, ge=0, description="Days between quote and purchase")
    policy_start_year: float = Field(2024.0, description="Year the policy starts")
    policy_start_week: float = Field(1.0, ge=1, le=53, description="ISO week number of policy start")

    # ── Categorical features ──────────────────────────────────────────────────
    policy_start_month: str = Field("January", description="Month the policy starts")
    broker_agency_type: Optional[str] = Field(None, description="Type of broker agency")
    acquisition_channel: Optional[str] = Field(None, description="How the customer was acquired")
    payment_schedule: Optional[str] = Field(None, description="Payment frequency")
    employment_status: Optional[str] = Field(None, description="Customer employment status")
    region_code: Optional[str] = Field(None, description="Customer region code")
    broker_id: Optional[float] = Field(None, description="Broker identifier (null if no broker)")
    deductible_tier: Optional[str] = Field(None, description="Deductible tier selected")


class BundleScore(BaseModel):
    bundle_id: int
    bundle_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Full prediction response returned to the frontend."""

    predicted_bundle: int = Field(..., description="Top predicted bundle ID (0–9)")
    predicted_bundle_name: str = Field(..., description="Human-readable bundle name")
    confidence: float = Field(..., description="Confidence of top prediction (0–1)")
    top_3: list[BundleScore] = Field(..., description="Top 3 bundle recommendations with scores")

    # Explanation fields
    key_factors: list[str] = Field(
        default_factory=list,
        description="Key customer attributes that influenced this prediction",
    )

    model_version: str = Field("1.0.0", description="Model artifact version")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    num_classes: int
    api_version: str
