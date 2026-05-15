import os
import pandas as pd


# Columnas mínimas y manejables para el proyecto
TRANSACTION_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "dist1",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C5", "C6",
    "D1", "D2", "D3", "D4", "D10", "D15",
    "M1", "M2", "M3", "M4", "M5", "M6"
]

IDENTITY_COLS = [
    "TransactionID",
    "DeviceType",
    "id_01", "id_02", "id_05", "id_06", "id_11",
    "id_12", "id_15", "id_16", "id_28", "id_29"
]


def load_data(
    transaction_path: str = "data/train_transaction.csv",
    identity_path: str = "data/train_identity.csv"
) -> pd.DataFrame:
    """
    Carga los dos datasets y los une por TransactionID.
    Optimizado para Render:
    - lee solo columnas necesarias
    - permite limitar filas en producción
    """

    # En Render usa muestra para evitar errores de memoria
    running_on_render = os.getenv("RENDER") is not None

    # Puedes ajustar este número si quieres más o menos datos en Render
    transaction_nrows = 100_000 if running_on_render else None

    transaction_dtypes = {
        "TransactionID": "int32",
        "isFraud": "int8",
        "TransactionDT": "int32",
        "TransactionAmt": "float32",
        "card1": "float32",
        "card2": "float32",
        "card3": "float32",
        "card5": "float32",
        "addr1": "float32",
        "addr2": "float32",
        "dist1": "float32",
        "C1": "float32",
        "C2": "float32",
        "C5": "float32",
        "C6": "float32",
        "D1": "float32",
        "D2": "float32",
        "D3": "float32",
        "D4": "float32",
        "D10": "float32",
        "D15": "float32",
    }

    identity_dtypes = {
        "TransactionID": "int32",
        "id_01": "float32",
        "id_02": "float32",
        "id_05": "float32",
        "id_06": "float32",
        "id_11": "float32",
        "id_12": "object",
        "id_15": "object",
        "id_16": "object",
        "id_28": "object",
        "id_29": "object",
        "DeviceType": "object",
    }

    train_transaction = pd.read_csv(
        transaction_path,
        usecols=lambda c: c in TRANSACTION_COLS,
        dtype={k: v for k, v in transaction_dtypes.items() if k in TRANSACTION_COLS},
        nrows=transaction_nrows
    )

    train_identity = pd.read_csv(
        identity_path,
        usecols=lambda c: c in IDENTITY_COLS,
        dtype={k: v for k, v in identity_dtypes.items() if k in IDENTITY_COLS}
    )

    # Si en Render cargaste solo parte de transaction, filtra identity para ese subset
    if transaction_nrows is not None:
        valid_ids = set(train_transaction["TransactionID"].unique())
        train_identity = train_identity[train_identity["TransactionID"].isin(valid_ids)]

    df = train_transaction.merge(
        train_identity,
        on="TransactionID",
        how="left"
    )

    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Retorna información básica del dataset combinado.
    """
    total_rows = len(df)
    fraud_count = int(df["isFraud"].sum())
    legit_count = total_rows - fraud_count
    fraud_ratio = fraud_count / total_rows if total_rows > 0 else 0
    missing_ratio = float(df.isna().mean().mean())

    return {
        "rows": total_rows,
        "columns": df.shape[1],
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_ratio": fraud_ratio,
        "missing_ratio": missing_ratio,
    }