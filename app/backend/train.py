import json
import os
import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, cross_validate

from app.backend.evaluate import evaluate_model
from app.backend.models import get_model
from app.backend.preprocess import build_preprocessor


def build_pipeline(model_name: str, strategy: str, X_train: pd.DataFrame):
    preprocessor = build_preprocessor(X_train)
    model = get_model(model_name, strategy)

    if strategy == "undersample":
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("undersample", RandomUnderSampler(random_state=42)),
            ("model", model)
        ])
    else:
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

    return pipeline


def compare_models(X_train, y_train, X_test, y_test):
    """
    Quitamos SMOTE en este dataset porque consume demasiada memoria.
    Dejamos 4 combinaciones más ligeras:
    - Logistic Regression + class_weight
    - Logistic Regression + undersample
    - Random Forest + class_weight
    - Random Forest + undersample
    """
    combinations = [
        ("logistic_regression", "class_weight"),
        ("logistic_regression", "undersample"),
        ("random_forest", "class_weight"),
        ("random_forest", "undersample"),
    ]

    results = []
    trained_models = {}

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    for model_name, strategy in combinations:
        pipeline = build_pipeline(model_name, strategy, X_train)

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=1,   # más seguro en memoria
            return_train_score=False
        )

        pipeline.fit(X_train, y_train)
        test_metrics = evaluate_model(pipeline, X_test, y_test, threshold=0.5)

        result = {
            "model_name": model_name,
            "strategy": strategy,
            "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_results["test_pr_auc"])),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_precision": float(test_metrics["precision"]),
            "test_recall": float(test_metrics["recall"]),
            "test_f1": float(test_metrics["f1"]),
            "test_roc_auc": float(test_metrics["roc_auc"]),
            "test_pr_auc": float(test_metrics["pr_auc"]),
        }

        results.append(result)
        trained_models[f"{model_name}_{strategy}"] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="cv_pr_auc_mean",
        ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


def get_sampling_preview(y_train: pd.Series) -> pd.DataFrame:
    counts = y_train.value_counts().sort_index()
    class_0 = int(counts.get(0, 0))
    class_1 = int(counts.get(1, 0))

    preview_df = pd.DataFrame({
        "Original": [class_0, class_1],
        "Undersample": [min(class_0, class_1), min(class_0, class_1)],
    }, index=["Clase 0", "Clase 1"])

    return preview_df


def save_best_model(model, metadata: dict):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/best_model.pkl")

    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def load_best_model():
    model = joblib.load("models/best_model.pkl")

    metadata = {}
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, metadata