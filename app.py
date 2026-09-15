from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="UPI Fraud Detection API",
    description="Synthetic UPI-style fraud detection and risk scoring API",
    version="1.0"
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

fraud_model = joblib.load("lightgbm_fraud_model.pkl")
iso_model = joblib.load("isolation_forest_model.pkl")
anomaly_scaler = joblib.load("anomaly_scaler.pkl")
score_scaler = joblib.load("anomaly_score_scaler.pkl")


# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class Transaction(BaseModel):
    amount: float
    sender_age: float
    receiver_type: str
    transaction_type: str
    device_id: str
    location: str
    hour: int
    account_age_days: int
    transactions_last_24h: int
    avg_amount_30d: float
    new_receiver: int
    new_device: int
    failed_attempts: int
    distance_from_usual_location_km: float


# --------------------------------------------------
# Health Endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "API is running"
    }


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def create_features(df):

    df["amount_to_avg_ratio"] = (
        df["amount"] /
        df["avg_amount_30d"].replace(0, 1)
    )

    df["high_amount_flag"] = (
        df["amount_to_avg_ratio"] >= 3
    ).astype(int)

    df["velocity_flag"] = (
        df["transactions_last_24h"] >= 10
    ).astype(int)

    df["location_anomaly_flag"] = (
        df["distance_from_usual_location_km"] >= 20
    ).astype(int)

    df["device_risk"] = (
        (df["new_device"] == 1) &
        (df["failed_attempts"] > 0)
    ).astype(int)

    df["receiver_risk"] = (
        (df["new_receiver"] == 1) &
        (df["high_amount_flag"] == 1)
    ).astype(int)

    df["failed_attempt_risk"] = (
        df["failed_attempts"] >= 2
    ).astype(int)

    df["late_night_flag"] = (
        df["hour"].between(0, 5)
    ).astype(int)

    df["account_age_risk"] = (
        df["account_age_days"] <= 30
    ).astype(int)

    risk_cols = [
        "high_amount_flag",
        "velocity_flag",
        "location_anomaly_flag",
        "device_risk",
        "receiver_risk",
        "failed_attempt_risk",
        "late_night_flag",
        "account_age_risk"
    ]

    df["risk_signal_count"] = df[risk_cols].sum(axis=1)

    return df


# --------------------------------------------------
# Risk Reasons
# --------------------------------------------------

def get_reasons(row):

    reasons = []

    if row["high_amount_flag"] == 1:
        reasons.append("Unusually high amount")

    if row["velocity_flag"] == 1:
        reasons.append("High transaction velocity")

    if row["location_anomaly_flag"] == 1:
        reasons.append("Unusual location")

    if row["new_device"] == 1:
        reasons.append("New device")

    if row["new_receiver"] == 1:
        reasons.append("New receiver")

    if row["failed_attempts"] >= 2:
        reasons.append("Multiple failed attempts")

    if row["late_night_flag"] == 1:
        reasons.append("Late-night transaction")

    if row["account_age_risk"] == 1:
        reasons.append("New account")

    if len(reasons) == 0:
        reasons.append("No strong behavioral signal")

    return reasons


# --------------------------------------------------
# Risk Level
# --------------------------------------------------

def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 60:
        return "MEDIUM"

    else:
        return "HIGH"


# --------------------------------------------------
# Predict Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input to dataframe
    input_df = pd.DataFrame(
        [transaction.model_dump()]
    )

    # Feature Engineering
    feature_df = create_features(
        input_df.copy()
    )

    # --------------------------------------------------
    # Fraud Probability
    # --------------------------------------------------

    fraud_probability = fraud_model.predict_proba(
        feature_df
    )[:, 1][0]

    fraud_probability_100 = (
        fraud_probability * 100
    )


    # --------------------------------------------------
    # Isolation Forest Features
    # --------------------------------------------------

    anomaly_features = [
        "amount",
        "transactions_last_24h",
        "avg_amount_30d",
        "failed_attempts",
        "distance_from_usual_location_km",
        "amount_to_avg_ratio",
        "risk_signal_count"
    ]

    anomaly_input = feature_df[
        anomaly_features
    ]

    anomaly_scaled = anomaly_scaler.transform(
        anomaly_input
    )


    # --------------------------------------------------
    # Anomaly Prediction
    # --------------------------------------------------

    iso_prediction = iso_model.predict(
        anomaly_scaled
    )[0]

    anomaly_flag = (
        1 if iso_prediction == -1 else 0
    )


    # --------------------------------------------------
    # Raw Anomaly Score
    # --------------------------------------------------

    raw_anomaly_score = -iso_model.decision_function(
        anomaly_scaled
    )[0]


    # --------------------------------------------------
    # Normalize Anomaly Score 0-100
    # --------------------------------------------------

    anomaly_score_100 = score_scaler.transform(
        [[raw_anomaly_score]]
    )[0][0]

    anomaly_score_100 = np.clip(
        anomaly_score_100,
        0,
        100
    )


    # --------------------------------------------------
    # Behavioral Score
    # --------------------------------------------------

    risk_signal_count = int(
        feature_df["risk_signal_count"].iloc[0]
    )

    behavior_score = (
        risk_signal_count / 5
    ) * 100

    behavior_score = min(
        behavior_score,
        100
    )


    # --------------------------------------------------
    # Final Risk Score
    # --------------------------------------------------

    risk_score = (
        0.60 * fraud_probability_100 +
        0.25 * anomaly_score_100 +
        0.15 * behavior_score
    )

    risk_score = round(
        float(risk_score),
        2
    )


    # --------------------------------------------------
    # Risk Level
    # --------------------------------------------------

    risk_level = get_risk_level(
        risk_score
    )


    # --------------------------------------------------
    # Reasons
    # --------------------------------------------------

    reasons = get_reasons(
        feature_df.iloc[0]
    )


    # --------------------------------------------------
    # Response
    # --------------------------------------------------

    return {

        "fraud_probability": round(
            float(fraud_probability_100),
            2
        ),

        "anomaly_score": round(
            float(anomaly_score_100),
            2
        ),

        "anomaly_flag": int(
            anomaly_flag
        ),

        "risk_signal_count": risk_signal_count,

        "behavior_score": round(
            float(behavior_score),
            2
        ),

        "risk_score": risk_score,

        "risk_level": risk_level,

        "reasons": reasons
    }