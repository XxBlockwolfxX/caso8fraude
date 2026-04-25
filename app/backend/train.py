import json
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from app.backend.models import get_model
from app.backend.evaluate import evaluate_model


def build_pipeline(model_name: str, strategy: str):
    """
    Construye el pipeline del modelo según la estrategia:
    - class_weight
    - smote
    - undersample
    """
    model = get_model(model_name, strategy)

    if strategy == "smote":
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])
    elif strategy == "undersample":
        pipeline = ImbPipeline([
            ("undersample", RandomUnderSampler(random_state=42)),
            ("model", model)
        ])
    else:
        pipeline = ImbPipeline([
            ("model", model)
        ])

    return pipeline


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compara 6 combinaciones:
    1. Logistic Regression + class_weight
    2. Logistic Regression + SMOTE
    3. Logistic Regression + undersample
    4. Random Forest + class_weight
    5. Random Forest + SMOTE
    6. Random Forest + undersample

    Ahora además usa StratifiedKFold para obtener promedios
    más confiables de Precision, Recall, F1 y PR-AUC.
    """
    combinations = [
        ("logistic_regression", "class_weight"),
        ("logistic_regression", "smote"),
        ("logistic_regression", "undersample"),
        ("random_forest", "class_weight"),
        ("random_forest", "smote"),
        ("random_forest", "undersample"),
    ]

    results = []
    trained_models = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    for model_name, strategy in combinations:
        pipeline = build_pipeline(model_name, strategy)

        # Validación cruzada estratificada
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        # Entrenamiento final para luego evaluar en test
        pipeline.fit(X_train, y_train)

        test_metrics = evaluate_model(pipeline, X_test, y_test, threshold=0.5)

        result = {
            "model_name": model_name,
            "strategy": strategy,

            # Promedios de validación cruzada
            "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(cv_results["test_pr_auc"])),

            # Desviación estándar opcional para mostrar estabilidad
            "cv_precision_std": float(np.std(cv_results["test_precision"])),
            "cv_recall_std": float(np.std(cv_results["test_recall"])),
            "cv_f1_std": float(np.std(cv_results["test_f1"])),
            "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
            "cv_pr_auc_std": float(np.std(cv_results["test_pr_auc"])),

            # Métricas en test final
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
        }

        results.append(result)
        trained_models[f"{model_name}_{strategy}"] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="cv_pr_auc_mean",
        ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


def get_sampling_preview(X_train, y_train):
    """
    Muestra cómo cambian las clases:
    - Original
    - Después de SMOTE
    - Después de Undersampling
    """
    original_counts = y_train.value_counts().sort_index()

    smote = SMOTE(random_state=42)
    _, y_smote = smote.fit_resample(X_train, y_train)
    smote_counts = pd.Series(y_smote).value_counts().sort_index()

    undersample = RandomUnderSampler(random_state=42)
    _, y_under = undersample.fit_resample(X_train, y_train)
    under_counts = pd.Series(y_under).value_counts().sort_index()

    preview_df = pd.DataFrame({
        "Original": original_counts,
        "SMOTE": smote_counts,
        "Undersample": under_counts
    }).fillna(0).astype(int)

    preview_df.index = ["Clase 0", "Clase 1"]
    return preview_df


def save_best_model(model, scaler, metadata: dict):
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)


def load_best_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    metadata = {}
    if os.path.exists("models/metadata.json"):
        with open("models/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, scaler, metadata