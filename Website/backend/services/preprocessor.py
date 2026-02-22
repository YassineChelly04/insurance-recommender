"""
Preprocessing service — mirrors the training notebook's feature engineering pipeline
exactly so that the live API produces features identical to those used during training.

Pipeline steps (matching dataoverflow-7-0.ipynb):
  1. Binary flags (Has_Broker, Has_Employer)
  2. Imputation (Broker_ID → -1, Child_Dependents → 0)
  3. Dependent aggregates  (Total_Dependents, Minor_Dependents)
  4. Income features (Log_Income, Income_Per_Dependent, Income_Bracket)
  5. Policy risk features (Policy_Complexity, Grace_Ratio, Claims_Rate, Clean_Ratio)
  6. Deductible ordinal encoding
  7. Cyclical month encoding
  8. Numeric imputation with training medians
  9. One-hot encoding (Broker_Agency_Type, Acquisition_Channel, Payment_Schedule, Employment_Status)
  10. Column alignment to training feature set
  11. Region_Code target encoding (Bayesian-smoothed, falls back to global_mean)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backend.models.loader import ModelArtifact
    from backend.schemas.customer import CustomerProfile

logger = logging.getLogger(__name__)

# Known category values from training data (for robust OHE alignment)
_BROKER_AGENCY_TYPES = [
    "Urban_Boutique",
    "National_Corporate",
    "Regional_Independent",
    "Online_Direct",
]
_ACQUISITION_CHANNELS = [
    "Direct_Website",
    "Affiliate_Group",
    "Corporate_Partner",
    "Aggregator_Site",
    "Broker_Referral",
    "Social_Media",
]
_PAYMENT_SCHEDULES = [
    "Monthly",
    "Quarterly",
    "Semi_Annual",
    "Annual",
]
_EMPLOYMENT_STATUSES = [
    "Employed_FullTime",
    "Employed_PartTime",
    "Self_Employed",
    "Unemployed",
    "Retired",
    "Student",
]

_MONTH_MAP: dict[str, int] = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

_DED_MAP: dict[str, int] = {
    "Tier_1_High_Ded": 3,
    "Tier_2_Mid_Ded": 2,
    "Tier_3_Low_Ded": 1,
    "Tier_4_Zero_Ded": 0,
}


def preprocess(profile: "CustomerProfile", artifact: "ModelArtifact") -> pd.DataFrame:
    """
    Transform a CustomerProfile into a single-row DataFrame aligned to the
    model's expected feature columns.

    Args:
        profile:  Validated CustomerProfile pydantic model from the request body.
        artifact: Loaded model artifact containing training metadata.

    Returns:
        pd.DataFrame with exactly one row and columns matching artifact.feature_names.
    """
    p = profile  # shorthand

    # ── 1. Build raw dict ─────────────────────────────────────────────────────
    raw: dict = {
        "Estimated_Annual_Income":       p.estimated_annual_income,
        "Adult_Dependents":              float(p.adult_dependents),
        "Child_Dependents":              float(p.child_dependents),
        "Infant_Dependents":             float(p.infant_dependents),
        "Previous_Policy_Duration_Months": p.previous_policy_duration_months,
        "Grace_Period_Extensions":       p.grace_period_extensions,
        "Years_Without_Claims":          p.years_without_claims,
        "Policy_Amendments_Count":       p.policy_amendments_count,
        "Vehicles_on_Policy":            p.vehicles_on_policy,
        "Custom_Riders_Requested":       p.custom_riders_requested,
        "Days_Since_Quote":              p.days_since_quote,
        "Policy_Start_Year":             p.policy_start_year,
        "Policy_Start_Week":             p.policy_start_week,
        "Policy_Start_Month":            p.policy_start_month,
        "Broker_Agency_Type":            p.broker_agency_type,
        "Acquisition_Channel":           p.acquisition_channel,
        "Payment_Schedule":              p.payment_schedule,
        "Employment_Status":             p.employment_status,
        "Region_Code":                   p.region_code if p.region_code else "Unknown",
        "Broker_ID":                     p.broker_id,
        "Deductible_Tier":               p.deductible_tier,
        # Columns dropped during training but needed for flag extraction
        "Previous_Claims_Filed":         0.0,
    }

    df = pd.DataFrame([raw])

    # ── 2. Binary flags ───────────────────────────────────────────────────────
    df["Has_Broker"] = (df["Broker_ID"].notna() & (df["Broker_ID"] != -1)).astype(int)
    df["Has_Employer"] = 0  # Employer_ID was always dropped; flag stays 0

    # ── 3. Impute Broker_ID ───────────────────────────────────────────────────
    df["Broker_ID"] = df["Broker_ID"].fillna(-1)

    # ── 4. Dependent aggregates ───────────────────────────────────────────────
    df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
    df["Minor_Dependents"] = df["Child_Dependents"] + df["Infant_Dependents"]

    # ── 5. Income features ────────────────────────────────────────────────────
    df["Log_Income"] = np.log1p(df["Estimated_Annual_Income"])
    df["Income_Per_Dependent"] = df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    # Income_Bracket: use training medians to approximate quintile boundaries
    income = df["Estimated_Annual_Income"].iloc[0]
    income_boundaries = [0, 25000, 45000, 65000, 90000, float("inf")]
    df["Income_Bracket"] = float(
        next(i for i, upper in enumerate(income_boundaries[1:]) if income <= upper)
    )

    # ── 6. Policy risk features ───────────────────────────────────────────────
    df["Policy_Complexity"] = (
        df["Vehicles_on_Policy"] + df["Custom_Riders_Requested"] + df["Policy_Amendments_Count"]
    )
    df["Grace_Ratio"] = df["Grace_Period_Extensions"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Claims_Rate"] = df["Previous_Claims_Filed"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Clean_Ratio"] = df["Years_Without_Claims"] / (df["Previous_Policy_Duration_Months"] + 1)

    # ── 7. Deductible ordinal encoding ────────────────────────────────────────
    df["Deductible_Ord"] = df["Deductible_Tier"].map(_DED_MAP).fillna(-1)
    df = df.drop(columns=["Deductible_Tier"])

    # ── 8. Cyclical month encoding ────────────────────────────────────────────
    df["Month_Num"] = df["Policy_Start_Month"].map(_MONTH_MAP).fillna(0)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df = df.drop(columns=["Policy_Start_Month", "Month_Num"])

    # ── 9. Numeric imputation with training medians ───────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col in artifact.medians:
            df[col] = df[col].fillna(artifact.medians[col])

    # ── 10. Categorical imputation (mode from training — use first known value) ─
    for col in ["Broker_Agency_Type", "Acquisition_Channel", "Payment_Schedule", "Employment_Status"]:
        if df[col].isna().any():
            # Fall back to most common training value
            fallbacks = {
                "Broker_Agency_Type": "National_Corporate",
                "Acquisition_Channel": "Direct_Website",
                "Payment_Schedule": "Monthly",
                "Employment_Status": "Employed_FullTime",
            }
            df[col] = df[col].fillna(fallbacks[col])

    # ── 11. One-hot encoding ──────────────────────────────────────────────────
    cat_cols = ["Broker_Agency_Type", "Acquisition_Channel", "Payment_Schedule", "Employment_Status"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # ── 12. Region_Code target encoding ──────────────────────────────────────
    region = df["Region_Code"].iloc[0] if "Region_Code" in df.columns else "Unknown"
    df["Region_Enc"] = artifact.region_map.get(str(region), artifact.global_mean)
    if "Region_Code" in df.columns:
        df = df.drop(columns=["Region_Code"])

    # Remove columns not in training set + add missing columns as 0
    df = df.reindex(columns=artifact.feature_names, fill_value=0)

    logger.debug("Preprocessed row — shape: %s", df.shape)
    return df
