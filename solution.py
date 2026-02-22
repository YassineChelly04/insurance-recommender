# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
# Do NOT change the function signatures.
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import time


def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'User_ID' column here.
    # It will be used in the predict function to return the final predictions.

    df = df.copy()

    # ── 1. DROP LOW-SIGNAL COLUMNS (keep User_ID) ──────────────────────────
    DROP_COLS = [
        "Employer_ID",
        "Previous_Claims_Filed",
        "Existing_Policyholder",
        "Underwriting_Processing_Days",
        "Infant_Dependents",
        "Policy_Start_Day",
    ]
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # ── 2. MISSING VALUE HANDLING ───────────────────────────────────────────
    df["Child_Dependents"]    = df["Child_Dependents"].fillna(0)
    df["Has_Broker"]          = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"]           = df["Broker_ID"].fillna(-1)
    df["Region_Code"]         = df["Region_Code"].fillna("Unknown")
    df["Deductible_Tier"]     = df["Deductible_Tier"].fillna("Unknown")
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna("Unknown")

    # ── 3. FEATURE ENGINEERING ─────────────────────────────────────────────
    df["Total_Dependents"]        = df["Adult_Dependents"] + df["Child_Dependents"]
    df["Income_Per_Dependent"]    = df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    df["Grace_To_Duration_Ratio"] = df["Grace_Period_Extensions"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Log_Income"]              = np.log1p(df["Estimated_Annual_Income"])
    df["Log_Days_Since_Quote"]    = np.log1p(df["Days_Since_Quote"])

    # Cyclical month encoding
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    df["Month_Num"] = pd.Categorical(df["Policy_Start_Month"], categories=month_order, ordered=True).codes + 1
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df.drop(columns=["Policy_Start_Month", "Month_Num"], inplace=True)

    # ── 4. ENCODING ─────────────────────────────────────────────────────────
    deductible_map = {
        "Tier_1_High_Ded": 3,
        "Tier_2_Mid_Ded":  2,
        "Tier_3_Low_Ded":  1,
        "Tier_4_Zero_Ded": 0,
        "Unknown":        -1,
    }
    df["Deductible_Tier"] = df["Deductible_Tier"].map(deductible_map)

    # Target encoding: Region_Code — on test set, fall back to 0.0
    if "Purchased_Coverage_Bundle" in df.columns:
        global_mean = df["Purchased_Coverage_Bundle"].mean()
        smoothing   = 10
        stats = df.groupby("Region_Code")["Purchased_Coverage_Bundle"].agg(["mean", "count"])
        stats["encoded"] = (
            (stats["mean"] * stats["count"] + global_mean * smoothing) /
            (stats["count"] + smoothing)
        )
        df["Region_Code_Encoded"] = df["Region_Code"].map(stats["encoded"]).fillna(global_mean)
    else:
        df["Region_Code_Encoded"] = 0.0
    df.drop(columns=["Region_Code"], inplace=True)

    # One-hot encoding: low-cardinality categoricals
    OHE_COLS = ["Broker_Agency_Type", "Acquisition_Channel", "Payment_Schedule", "Employment_Status"]
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)

    # ── 5. STANDARDISATION ──────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    STD_COLS = [c for c in [
        "Estimated_Annual_Income", "Log_Income", "Log_Days_Since_Quote",
        "Income_Per_Dependent", "Grace_To_Duration_Ratio", "Days_Since_Quote",
        "Previous_Policy_Duration_Months", "Policy_Start_Year",
        "Policy_Start_Week", "Broker_ID", "Region_Code_Encoded",
    ] if c in df.columns]
    df[STD_COLS] = StandardScaler().fit_transform(df[STD_COLS])

    # ── 6. NORMALISATION ─────────────────────────────────────────────────────
    MM_COLS = [c for c in [
        "Adult_Dependents", "Child_Dependents", "Total_Dependents",
        "Grace_Period_Extensions", "Years_Without_Claims",
        "Policy_Amendments_Count", "Vehicles_on_Policy",
        "Custom_Riders_Requested", "Deductible_Tier",
        "Month_Sin", "Month_Cos",
    ] if c in df.columns]
    df[MM_COLS] = MinMaxScaler().fit_transform(df[MM_COLS])

    return df


def load_model():
    # ------------------ MODEL LOADING LOGIC ------------------
    import catboost  # Required so joblib can deserialize the CatBoostClassifier
    model = joblib.load("model.joblib")
    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    # ------------------ PREDICTION LOGIC ------------------
    # df has already been passed through preprocess().
    # User_ID is retained in df for the final output.

    user_ids = df["User_ID"]
    X = df.drop(columns=["User_ID"], errors="ignore")
    X = X.drop(columns=["Purchased_Coverage_Bundle"], errors="ignore")

    # Align columns to training feature set using CatBoost's native attribute
    train_features = model.feature_names_
    X = X.reindex(columns=train_features, fill_value=0)

    # Flatten to 1D — CatBoost.predict() returns 2D for multi-class
    preds = model.predict(X).flatten().astype(int)

    predictions = pd.DataFrame({
        "User_ID":                   user_ids.values,
        "Purchased_Coverage_Bundle": preds,
    })
    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------

def run(df):
    df_processed = preprocess(df)
    model        = load_model()

    import os
    size = os.path.getsize("model.joblib") / (1024 * 1024)

    start       = time.perf_counter()
    predictions = predict(df_processed, model)
    duration    = time.perf_counter() - start

    return size, 0.0, duration


def get_model_size(model):
    import pickle
    return len(pickle.dumps(model)) / (1024 * 1024)


def get_model_accuracy(predictions):
    pass
