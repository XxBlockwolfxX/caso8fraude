from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_model(model_name: str, strategy: str):
    """
    Retorna el modelo solicitado según el nombre y la estrategia.
    strategy:
        - "class_weight"
        - "smote"
    """
    if model_name == "logistic_regression":
        if strategy == "class_weight":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        return LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    if model_name == "random_forest":
        if strategy == "class_weight":
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=None,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

    raise ValueError(f"Modelo no soportado: {model_name}")