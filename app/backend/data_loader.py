import pandas as pd


def load_data(
    transaction_path: str = "data/train_transaction.csv",
    identity_path: str = "data/train_identity.csv"
) -> pd.DataFrame:
    """
    Carga los dos datasets y los une por TransactionID.
    """
    train_transaction = pd.read_csv(transaction_path)
    train_identity = pd.read_csv(identity_path)

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