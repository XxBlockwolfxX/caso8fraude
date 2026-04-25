from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import json

from app.backend.train import load_best_model
from app.backend.threshold import evaluate_threshold
from app.backend.data_loader import load_data
from app.backend.preprocess import split_data, scale_time_amount

app = FastAPI(title="Fraud Detection API")


class TransactionInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def home():
    return {"message": "API de detección de fraude activa"}


@app.post("/transactions/evaluate")
def evaluate_transaction(transaction: TransactionInput):
    if not os.path.exists("models/best_model.pkl"):
        raise HTTPException(status_code=404, detail="No existe un modelo entrenado.")

    model, scaler, metadata = load_best_model()

    input_dict = transaction.dict()
    df = pd.DataFrame([input_dict])

    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    probability = float(model.predict_proba(df)[0][1])
    threshold = metadata.get("selected_threshold", 0.5)
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability_fraud": round(probability, 6),
        "threshold_used": threshold
    }


@app.get("/transactions/threshold")
def threshold_projection(value: float = 0.5):
    if value < 0 or value > 1:
        raise HTTPException(status_code=400, detail="El umbral debe estar entre 0 y 1.")

    if not os.path.exists("models/best_model.pkl"):
        raise HTTPException(status_code=404, detail="No existe un modelo entrenado.")

    model, scaler, metadata = load_best_model()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_time_amount(X_train, X_test)

    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    metrics = evaluate_threshold(y_test, y_probs, value)

    return metrics