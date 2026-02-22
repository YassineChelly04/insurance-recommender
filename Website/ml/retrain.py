"""
Retraining script skeleton — MLOps ready.
Run this to retrain the XGBoost model from a fresh CSV.

Usage:
    python ml/retrain.py --train data/train.csv --output model.joblib

Arguments:
    --train    Path to training CSV (same format as original train.csv)
    --output   Output path for the saved model artifact
    --seed     Random seed (default: 42)
    --n-est    Max XGBoost estimators (default: 1000)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain XGBoost Insurance Bundle Classifier")
    parser.add_argument("--train",  default="train.csv",    help="Path to training CSV")
    parser.add_argument("--output", default="model.joblib", help="Output artifact path")
    parser.add_argument("--seed",   type=int, default=42,   help="Random seed")
    parser.add_argument("--n-est",  type=int, default=1000, help="XGBoost max estimators")
    return parser.parse_args()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — identical to training notebook."""
    df = df.copy()

    df["Has_Broker"]   = df["Broker_ID"].notna().astype(int)
    df["Has_Employer"] = 0  # Employer_ID always dropped
    df["Broker_ID"]    = df["Broker_ID"].fillna(-1)

    df["Child_Dependents"]   = df["Child_Dependents"].fillna(0)
    df["Total_Dependents"]   = df["Adult_Dependents"] + df["Child_Dependents"] + df["Infant_Dependents"]
    df["Minor_Dependents"]   = df["Child_Dependents"] + df["Infant_Dependents"]

    df["Log_Income"]          = np.log1p(df["Estimated_Annual_Income"])
    df["Income_Per_Dependent"]= df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    df["Income_Bracket"]      = pd.qcut(
        df["Estimated_Annual_Income"], q=5, labels=False, duplicates="drop"
    ).astype(float)

    df["Policy_Complexity"] = df["Vehicles_on_Policy"] + df["Custom_Riders_Requested"] + df["Policy_Amendments_Count"]
    df["Grace_Ratio"]       = df["Grace_Period_Extensions"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Claims_Rate"]       = df["Previous_Claims_Filed"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Clean_Ratio"]       = df["Years_Without_Claims"] / (df["Previous_Policy_Duration_Months"] + 1)

    ded_map = {"Tier_1_High_Ded": 3, "Tier_2_Mid_Ded": 2, "Tier_3_Low_Ded": 1, "Tier_4_Zero_Ded": 0}
    df["Deductible_Ord"] = df["Deductible_Tier"].map(ded_map).fillna(-1)
    df = df.drop(columns=["Deductible_Tier"])

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
    }
    df["Month_Num"] = df["Policy_Start_Month"].map(month_map).fillna(0)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df = df.drop(columns=["Policy_Start_Month", "Month_Num"])

    return df


def main() -> None:
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    train_path = Path(args.train)
    if not train_path.exists():
        logger.error("Training file not found: %s", train_path)
        sys.exit(1)

    logger.info("Loading training data from %s", train_path)
    df = pd.read_csv(train_path)
    TARGET = "Purchased_Coverage_Bundle"
    DROP_COLS = ["User_ID", "Employer_ID", "Policy_Start_Week", "Policy_Start_Day"]
    df = df.drop(columns=DROP_COLS, errors="ignore")
    logger.info("Raw shape: %s", df.shape)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Imputation & encoding ──────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET, errors="ignore").tolist()
    medians = df[num_cols].median().to_dict()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if c not in [TARGET, "Region_Code"]]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # Target encoding for Region_Code
    df["Region_Code"] = df["Region_Code"].fillna("Unknown")
    global_mean = df[TARGET].mean()
    smoothing   = 20
    region_stats = df.groupby("Region_Code")[TARGET].agg(["mean", "count"])
    region_stats["encoded"] = (
        (region_stats["mean"] * region_stats["count"] + global_mean * smoothing)
        / (region_stats["count"] + smoothing)
    )
    region_map = region_stats["encoded"].to_dict()
    df["Region_Enc"] = df["Region_Code"].map(region_map).fillna(global_mean)
    df = df.drop(columns=["Region_Code"])

    # ── 4. Prepare X / y ──────────────────────────────────────────────────────
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values.astype(int)
    feature_names = list(X.columns)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    NUM_CLASSES = len(le.classes_)
    logger.info("Classes: %s (%d total)", le.classes_, NUM_CLASSES)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X.values, y_enc, test_size=0.1, random_state=args.seed, stratify=y_enc
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    import xgboost as xgb

    logger.info("Training XGBoost (max %d estimators)…", args.n_est)
    t0 = time.perf_counter()

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric=["mlogloss", "merror"],
        n_estimators=args.n_est,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=args.seed,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1fs. Best iteration: %d", elapsed, model.best_iteration)

    # ── 6. Save artifact ──────────────────────────────────────────────────────
    artifact = {
        "model":         model,
        "feature_names": feature_names,
        "label_encoder": le,
        "region_map":    region_map,
        "global_mean":   global_mean,
        "medians":       medians,
        "cat_cols":      cat_cols,
        "num_classes":   NUM_CLASSES,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    logger.info("Artifact saved → %s", output_path)


if __name__ == "__main__":
    main()
