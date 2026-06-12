"""Costruzione delle tre pipeline di preprocessing e della pipeline completa.

Le factory di questo modulo assemblano, a partire dalle primitive scikit-learn e dai
transformer custom di :mod:`preprocessing.transformers`, le tre pipeline descritte
nella consegna e il loro ``FeatureUnion`` finale (il dataset *max-feature*).

Nota di design (skewness rivalutata per step). Negli step numerici la partizione
asimmetriche/simmetriche viene ricalcolata a ogni passo, ed è intenzionale: l'imputazione
sceglie media vs mediana in base alla skewness del dato *grezzo* (la mediana è robusta
sulle code), mentre la simmetrizzazione agisce sulle colonne ancora asimmetriche *dopo*
l'imputazione — cioè su ciò che ha davvero senso simmetrizzare.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

from .transformers import PipelineWithRowFilter, SelectiveTransformerBySkewness

# Selettori di colonna riusati: numeriche vs categoriche (per dtype).
_numeric = make_column_selector(dtype_include=np.number)
_categorical = make_column_selector(dtype_exclude=np.number)


def _imputation_block() -> list[tuple[str, BaseEstimator]]:
    """Imputazione selettiva: media per le simmetriche, mediana per le asimmetriche.

    ``keep_empty_features=True`` evita che una colonna interamente NaN venga eliminata
    (verrebbe riempita con 0), preservando l'allineamento delle colonne a valle.
    """
    return [
        (
            "impute_symmetric",
            SelectiveTransformerBySkewness(
                SimpleImputer(strategy="mean", keep_empty_features=True),
                apply_to="symmetric",
            ),
        ),
        (
            "impute_skewed",
            SelectiveTransformerBySkewness(
                SimpleImputer(strategy="median", keep_empty_features=True),
                apply_to="skewed",
            ),
        ),
    ]


def build_numeric_symmetrization_block() -> list[tuple[str, BaseEstimator]]:
    """Imputazione selettiva seguita dalla simmetrizzazione delle asimmetriche.

    È il sotto-blocco numerico comune alle Pipeline 1 e 3 (la 2 usa solo l'imputazione,
    poiché discretizza invece di simmetrizzare).
    """
    return [
        *_imputation_block(),
        (
            "symmetrize",
            SelectiveTransformerBySkewness(
                PowerTransformer(method="yeo-johnson", standardize=False),
                apply_to="skewed",
            ),
        ),
    ]


def build_pipeline_1() -> PipelineWithRowFilter:
    """Pipeline 1 — addestrata sui soli record positivi (target=1).

    Numeriche: imputazione selettiva, simmetrizzazione, standardizzazione.
    Categoriche: imputazione (moda) e one-hot encoding.
    L'intero ``ColumnTransformer`` è avvolto in :class:`PipelineWithRowFilter`, che lo
    addestra solo sui positivi ma lo applica a tutte le righe.
    """
    numeric = Pipeline(
        [
            *build_numeric_symmetrization_block(),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            ("numeric", numeric, _numeric),
            ("categorical", categorical, _categorical),
        ]
    )
    return PipelineWithRowFilter(pipeline=column_transformer, fit_target_value=1)


def build_pipeline_2(k: int = 5, ordinal_categories: Any = "auto") -> Pipeline:
    """Pipeline 2 — applicata a tutti i record.

    Numeriche: imputazione selettiva e discretizzazione in 20 bin.
    Categoriche: imputazione (moda) e encoding ordinale.
    In coda, selezione delle ``k`` feature più informative tramite ANOVA F.

    :param k: Numero di feature da selezionare (richiede ``y`` al fit).
    :param ordinal_categories: Categorie per l'``OrdinalEncoder``. ``'auto'`` le deriva
        dai dati; in alternativa una lista di liste (una per colonna categorica).
    """
    numeric = Pipeline(
        [
            *_imputation_block(),
            ("discretize", KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="uniform")),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    column_transformer = ColumnTransformer(
        [
            ("numeric", numeric, _numeric),
            ("categorical", categorical, _categorical),
        ]
    )
    return Pipeline(
        [
            ("preprocess", column_transformer),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def build_pipeline_3(pca_variance: float = 0.85) -> ColumnTransformer:
    """Pipeline 3 — solo variabili numeriche.

    Imputazione selettiva, PCA (a varianza spiegata cumulata), simmetrizzazione e
    normalizzazione campione per campione. Le colonne categoriche vengono scartate.

    :param pca_variance: Frazione di varianza da preservare nella PCA.
    """
    numeric = Pipeline(
        [
            *build_numeric_symmetrization_block(),
            ("pca", PCA(n_components=pca_variance)),
            ("normalize", Normalizer()),
        ]
    )
    return ColumnTransformer([("numeric", numeric, _numeric)])


def build_full_pipeline(
    k: int = 5,
    ordinal_categories: Any = "auto",
    pca_variance: float = 0.85,
) -> FeatureUnion:
    """Unisce le tre pipeline in un unico dataset *max-feature* (``FeatureUnion``).

    Il ``FeatureUnion`` prefissa i nomi di colonna con ``pipe1__``/``pipe2__``/``pipe3__``,
    garantendo l'unicità anche in caso di nomi originali coincidenti fra i rami.
    """
    return FeatureUnion(
        [
            ("pipe1", build_pipeline_1()),
            ("pipe2", build_pipeline_2(k=k, ordinal_categories=ordinal_categories)),
            ("pipe3", build_pipeline_3(pca_variance=pca_variance)),
        ]
    )
