"""Libreria di preprocessing & feature engineering (progetto 6 del master).

Espone i due transformer scikit-learn custom, le factory che assemblano le tre pipeline
della consegna e la loro unione, più le utility di caricamento dati e visualizzazione.
"""

from __future__ import annotations

from sklearn import set_config

from .data import DATASET_URL, TARGET_COLUMN, load_dataset, split_features_target
from .pipelines import (
    build_full_pipeline,
    build_numeric_symmetrization_block,
    build_pipeline_1,
    build_pipeline_2,
    build_pipeline_3,
)
from .plotting import plot_distribution
from .transformers import PipelineWithRowFilter, SelectiveTransformerBySkewness

# I transformer custom e le pipeline ragionano per nome di colonna: l'output pandas
# garantisce che i nomi delle feature sopravvivano lungo tutta la composizione
# (ColumnTransformer -> Pipeline -> FeatureUnion).
set_config(transform_output="pandas")

__all__ = [
    "DATASET_URL",
    "TARGET_COLUMN",
    "PipelineWithRowFilter",
    "SelectiveTransformerBySkewness",
    "build_full_pipeline",
    "build_numeric_symmetrization_block",
    "build_pipeline_1",
    "build_pipeline_2",
    "build_pipeline_3",
    "load_dataset",
    "plot_distribution",
    "split_features_target",
]
