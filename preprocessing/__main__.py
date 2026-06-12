"""Demo end-to-end del package: ``python -m preprocessing``.

Scarica il dataset, costruisce e addestra la pipeline completa, salva l'oggetto serializzato
e stampa un mini-report di sanità sull'output trasformato.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from .data import load_dataset, split_features_target
from .pipelines import build_full_pipeline

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "pipeline_completa.joblib"


def main() -> None:
    """Esegue la pipeline completa sul dataset e ne salva l'oggetto addestrato."""
    df = load_dataset()
    X, y = split_features_target(df)

    pipeline = build_full_pipeline()
    transformed = pipeline.fit_transform(X, y)
    joblib.dump(pipeline, OUTPUT_PATH)

    print(f"Input:  {X.shape[0]} righe x {X.shape[1]} colonne")
    print(f"Output: {transformed.shape[0]} righe x {transformed.shape[1]} colonne")
    print(f"Valori NaN nell'output: {int(transformed.isna().sum().sum())}")
    print(f"Nomi colonna unici:     {bool(transformed.columns.is_unique)}")
    print(f"Pipeline salvata in:    {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
