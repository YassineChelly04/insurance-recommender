"""
Inference module - loads model and runs predictions
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

_MODEL = None
_MODEL_PATH = "catboost_model.joblib"

def load_model():
    """Lazy-load model on first call"""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {_MODEL_PATH}")
        _MODEL = joblib.load(_MODEL_PATH)
    return _MODEL


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess DataFrame following the training pipeline.
    """
    df = df.copy()
    
    numeric_defaults = {
        "Adult_Dependents": 0,
        "Child_Dependents": 0,
        "Estimated_Annual_Income": 0,
        "Grace_Period_Extensions": 0,
        "Previous_Policy_Duration_Months": 0,
        "Days_Since_Quote": 0,
        "Years_Without_Claims": 0,
        "Policy_Amendments_Count": 0,
        "Vehicles_on_Policy": 0,
        "Custom_Riders_Requested": 0,
        "Policy_Start_Year": 0,
        "Policy_Start_Week": 0,
    }
    text_defaults = {
        "User_ID": "",
        "Broker_ID": -1,
        "Region_Code": "Unknown",
        "Deductible_Tier": "Unknown",
        "Acquisition_Channel": "Unknown",
        "Broker_Agency_Type": "Unknown",
        "Payment_Schedule": "Unknown",
        "Employment_Status": "Unknown",
        "Policy_Start_Month": "Unknown",
    }

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    for col, default in text_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    DROP_COLS = [
        "Employer_ID", "Previous_Claims_Filed", "Existing_Policyholder",
        "Underwriting_Processing_Days", "Infant_Dependents", "Policy_Start_Day",
    ]
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    df["Child_Dependents"] = df["Child_Dependents"].fillna(0)
    df["Has_Broker"] = df["Broker_ID"].notna().astype(int)
    df["Broker_ID"] = df["Broker_ID"].fillna(-1)
    df["Region_Code"] = df["Region_Code"].fillna("Unknown")
    df["Deductible_Tier"] = df["Deductible_Tier"].fillna("Unknown")
    df["Acquisition_Channel"] = df["Acquisition_Channel"].fillna("Unknown")

    df["Total_Dependents"] = df["Adult_Dependents"] + df["Child_Dependents"]
    df["Income_Per_Dependent"] = df["Estimated_Annual_Income"] / (df["Total_Dependents"] + 1)
    df["Grace_To_Duration_Ratio"] = df["Grace_Period_Extensions"] / (df["Previous_Policy_Duration_Months"] + 1)
    df["Log_Income"] = np.log1p(df["Estimated_Annual_Income"])
    df["Log_Days_Since_Quote"] = np.log1p(df["Days_Since_Quote"])

    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    df["Month_Num"] = pd.Categorical(df["Policy_Start_Month"], categories=month_order, ordered=True).codes + 1
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    df.drop(columns=[c for c in ["Policy_Start_Month", "Month_Num"] if c in df.columns], inplace=True)

    deductible_map = {"Tier_1_High_Ded": 3, "Tier_2_Mid_Ded": 2, "Tier_3_Low_Ded": 1, "Tier_4_Zero_Ded": 0, "Unknown": -1}
    df["Deductible_Tier"] = df["Deductible_Tier"].map(deductible_map).fillna(-1)

    if "Purchased_Coverage_Bundle" in df.columns:
        global_mean = df["Purchased_Coverage_Bundle"].mean()
        smoothing = 10
        stats = df.groupby("Region_Code")["Purchased_Coverage_Bundle"].agg(["mean", "count"])
        stats["encoded"] = ((stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing))
        df["Region_Code_Encoded"] = df["Region_Code"].map(stats["encoded"]).fillna(global_mean)
    else:
        df["Region_Code_Encoded"] = 0.0
    df.drop(columns=[c for c in ["Region_Code"] if c in df.columns], inplace=True)

    OHE_COLS = ["Broker_Agency_Type", "Acquisition_Channel", "Payment_Schedule", "Employment_Status"]
    df = pd.get_dummies(df, columns=[c for c in OHE_COLS if c in df.columns], drop_first=False, dtype=int)

    STD_COLS = [c for c in [
        "Estimated_Annual_Income", "Log_Income", "Log_Days_Since_Quote",
        "Income_Per_Dependent", "Grace_To_Duration_Ratio", "Days_Since_Quote",
        "Previous_Policy_Duration_Months", "Policy_Start_Year", "Policy_Start_Week", "Broker_ID", "Region_Code_Encoded",
    ] if c in df.columns]
    if STD_COLS:
        std_scaler = StandardScaler()
        df[STD_COLS] = std_scaler.fit_transform(df[STD_COLS])

    MM_COLS = [c for c in [
        "Adult_Dependents", "Child_Dependents", "Total_Dependents",
        "Grace_Period_Extensions", "Years_Without_Claims", "Policy_Amendments_Count",
        "Vehicles_on_Policy", "Custom_Riders_Requested", "Deductible_Tier", "Month_Sin", "Month_Cos",
    ] if c in df.columns]
    if MM_COLS:
        mm_scaler = MinMaxScaler()
        df[MM_COLS] = mm_scaler.fit_transform(df[MM_COLS])

    return df


def predict_bundle(user_data: dict) -> dict:
    """
    Predict insurance bundle for a single user.
    Input: dict with customer features
    Output: dict with prediction and confidence
    """
    model = load_model()
    
    df = pd.DataFrame([user_data])
    df_processed = preprocess(df)
    
    features = None
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    elif hasattr(model, "feature_names_"):
        features = list(model.feature_names_)
    
    if features is not None:
        X = df_processed.drop(columns=["User_ID"], errors="ignore").reindex(columns=features, fill_value=0)
    else:
        X = df_processed.drop(columns=["User_ID"], errors="ignore")
    
    pred = model.predict(X)[0]
    pred = int(pred)
    
    return {
        "predicted_bundle": pred,
        "bundle_class": f"Bundle_{pred}",
        "status": "success"
    }
