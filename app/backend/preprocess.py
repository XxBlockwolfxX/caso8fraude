from typing import Tuple, List, Dict, Any
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"

# Reducimos columnas para evitar explosión de memoria
FEATURE_COLUMNS = [
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "dist1",
    "P_emaildomain",
    "C1", "C2", "C5", "C6",
    "D1", "D2", "D3", "D4", "D10", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6",
    "DeviceType",
    "id_01", "id_02", "id_05", "id_06", "id_11",
    "id_12", "id_15", "id_16", "id_28", "id_29"
]


def get_selected_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    selected_columns = get_selected_feature_columns(df)

    X = df[selected_columns].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def get_dataset_defaults(df: pd.DataFrame) -> Dict[str, Any]:
    selected_columns = get_selected_feature_columns(df)
    defaults: Dict[str, Any] = {}

    for col in selected_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median()) if not df[col].dropna().empty else 0.0
        else:
            mode_series = df[col].mode(dropna=True)
            defaults[col] = mode_series.iloc[0] if not mode_series.empty else "Missing"

    return defaults


def build_manual_input_dataframe(df: pd.DataFrame, overrides: Dict[str, Any]) -> pd.DataFrame:
    row = get_dataset_defaults(df)

    for key, value in overrides.items():
        if key in row:
            row[key] = value

    selected_columns = get_selected_feature_columns(df)
    return pd.DataFrame([[row[col] for col in selected_columns]], columns=selected_columns)