import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_threshold(y_true, y_probs, threshold: float):
    """
    Evalúa las métricas para un umbral específico.
    """
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alerts": int(tp + fp),
        "frauds_captured": int(tp),
        "missed_frauds": int(fn),
    }


def generate_threshold_table(y_true, y_probs, start=0.01, end=0.99, step=0.01):
    """
    Genera una tabla con métricas para muchos umbrales.
    """
    thresholds = np.arange(start, end + step, step)
    rows = []

    for threshold in thresholds:
        metrics = evaluate_threshold(y_true, y_probs, round(float(threshold), 2))
        rows.append(metrics)

    return pd.DataFrame(rows)


def get_threshold_recommendations(y_true, y_probs):
    """
    Devuelve recomendaciones automáticas de umbral para distintos enfoques:
    - mejor equilibrio general (F1)
    - máxima captura de fraude (Recall)
    - mayor precisión
    """
    table = generate_threshold_table(y_true, y_probs)

    best_f1 = table.sort_values(
        by=["f1", "recall", "precision"],
        ascending=[False, False, False]
    ).iloc[0]

    best_recall = table.sort_values(
        by=["recall", "precision"],
        ascending=[False, False]
    ).iloc[0]

    best_precision = table.sort_values(
        by=["precision", "recall"],
        ascending=[False, False]
    ).iloc[0]

    return {
        "best_f1": best_f1.to_dict(),
        "best_recall": best_recall.to_dict(),
        "best_precision": best_precision.to_dict(),
        "table": table
    }