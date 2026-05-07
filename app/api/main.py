import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.backend.data_loader import load_data
from app.backend.preprocess import build_manual_input_dataframe, split_data
from app.backend.threshold import evaluate_threshold
from app.backend.train import load_best_model

app = FastAPI(title="Fraud Detection API")


class TransactionInput(BaseModel):
    TransactionDT: float
    TransactionAmt: float
    ProductCD: str = "W"
    card4: str = "visa"
    card6: str = "debit"
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None


@app.get("/")
def home():
    return {"message": "API de detección de fraude activa"}


@app.post("/transactions/evaluate")
def evaluate_transaction(transaction: TransactionInput):
    if not os.path.exists("models/best_model.pkl"):
        raise HTTPException(status_code=404, detail="No existe un modelo entrenado.")

    model, metadata = load_best_model()
    df = load_data()

    overrides = transaction.model_dump() if hasattr(transaction, "model_dump") else transaction.dict()
    input_df = build_manual_input_dataframe(df, overrides)

    probability = float(model.predict_proba(input_df)[0][1])
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

    model, _ = load_best_model()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    y_probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_threshold(y_test, y_probs, value)

    return metrics