from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name: str, strategy: str):
    if model_name == "logistic_regression":
        if strategy == "class_weight":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=600,
                random_state=42
            )
        return LogisticRegression(
            max_iter=600,
            random_state=42
        )

    if model_name == "decision_tree":
        if strategy == "class_weight":
            return DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )

    if model_name == "random_forest":
        if strategy == "class_weight":
            return RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            )
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )

    raise ValueError(f"Modelo o estrategia no soportados: {model_name} - {strategy}")