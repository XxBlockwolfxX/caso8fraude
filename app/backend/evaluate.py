<<<<<<< HEAD
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evalúa un modelo usando un umbral configurable.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "pr_auc": average_precision_score(y_test, y_probs),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_probs": y_probs,
        "y_pred": y_pred,
    }

    return results


def get_precision_recall_data(y_test, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    return precision, recall, thresholds


def get_roc_data(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
=======
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evalúa un modelo usando un umbral configurable.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "pr_auc": average_precision_score(y_test, y_probs),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_probs": y_probs,
        "y_pred": y_pred,
    }

    return results


def get_precision_recall_data(y_test, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    return precision, recall, thresholds


def get_roc_data(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
>>>>>>> 9aba5e6f79a86c76dd8bcf8750187d97bc25e2c4
    return fpr, tpr, thresholds