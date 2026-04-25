import pandas as pd


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    """
    df = pd.read_csv(path)
    return df


def get_basic_info(df: pd.DataFrame) -> dict:
    """
    Retorna información básica del dataset.
    """
    total_rows = len(df)
    fraud_count = int(df["Class"].sum())
    legit_count = total_rows - fraud_count
    fraud_ratio = fraud_count / total_rows if total_rows > 0 else 0

    return {
        "rows": total_rows,
        "columns": df.shape[1],
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_ratio": fraud_ratio,
    }