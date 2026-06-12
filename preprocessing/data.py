"""Caricamento del dataset e separazione feature/target.

Isola l'unico accesso di rete del package (il CSV su S3) in un punto solo, così che i
test possano usare dataset sintetici senza dipendere dalla connettività.
"""

from __future__ import annotations

import pandas as pd

DATASET_URL = "https://proai-datasets.s3.eu-west-3.amazonaws.com/sample_dataset.csv"
TARGET_COLUMN = "target"


def load_dataset(url: str = DATASET_URL) -> pd.DataFrame:
    """Carica il dataset di rilevazione del tumore al seno da ``url``."""
    return pd.read_csv(url)


def split_features_target(
    df: pd.DataFrame, target: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa le feature (X) dalla colonna target (y).

    :raises KeyError: Se la colonna target non è presente nel DataFrame.
    """
    if target not in df.columns:
        raise KeyError(f"Colonna target '{target}' assente dal DataFrame.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
