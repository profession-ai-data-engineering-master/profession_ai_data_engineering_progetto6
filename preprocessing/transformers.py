"""Transformer scikit-learn custom per il preprocessing.

Due estensioni dell'API `BaseEstimator`/`TransformerMixin` che le primitive di
scikit-learn non coprono nativamente:

- :class:`PipelineWithRowFilter`: addestra una pipeline interna su un sottoinsieme
  di record (selezionato in base al target) ma trasforma *tutte* le righe.
- :class:`SelectiveTransformerBySkewness`: applica un transformer solo alle colonne
  numeriche selezionate in base alla loro asimmetria (skewness).
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted


def _as_dataframe(X: Any) -> pd.DataFrame:
    """Garantisce che l'input sia un ``DataFrame`` pandas.

    I transformer di questo modulo ragionano per nome di colonna (skewness,
    selezione), quindi lavorano su ``DataFrame``. Un ``ndarray`` viene avvolto con
    nomi di colonna posizionali.
    """
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


class PipelineWithRowFilter(BaseEstimator, TransformerMixin):
    """Addestra una pipeline interna su un sottoinsieme di record, trasforma tutto.

    La pipeline interna *apprende* i parametri esclusivamente dai record il cui
    target è uguale a ``fit_target_value`` (così i record esclusi non contaminano le
    statistiche apprese, ad es. media/scala). A ``transform`` la trasformazione viene
    poi applicata a **tutte** le righe in input.

    Da ciò discende che l'output non contiene NaN strutturali e che vale l'invariante
    ``fit_transform(X, y) == fit(X, y).transform(X)``.

    :param pipeline: Stimatore scikit-learn (tipicamente un ``ColumnTransformer`` o
        una ``Pipeline``) da addestrare sul sottoinsieme di record.
    :param fit_target_value: Valore di ``y`` che identifica i record su cui addestrare
        la pipeline. Se ``None`` (o se ``y`` non viene passato a ``fit``) la pipeline
        viene addestrata su tutte le righe.
    """

    def __init__(self, pipeline: BaseEstimator, fit_target_value: Any = None) -> None:
        self.pipeline = pipeline
        self.fit_target_value = fit_target_value

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Addestra la pipeline interna sui soli record selezionati dal target.

        :raises ValueError: Se il filtro sul target non seleziona alcun record.
        """
        X = _as_dataframe(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        if y is not None and self.fit_target_value is not None:
            y_ser = pd.Series(np.asarray(y), index=X.index)
            mask = y_ser == self.fit_target_value
            if not mask.any():
                raise ValueError(
                    f"Il filtro fit_target_value={self.fit_target_value!r} non seleziona "
                    "alcun record."
                )
            X_fit = X.loc[mask]
            y_fit: Any = y_ser.loc[mask]
        else:
            X_fit = X
            y_fit = y

        self.pipeline_ = clone(self.pipeline)
        self.pipeline_.fit(X_fit, y_fit)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applica la pipeline addestrata a tutte le righe in input."""
        check_is_fitted(self)
        X = _as_dataframe(X)
        return self.pipeline_.transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Delega i nomi delle feature di output alla pipeline interna."""
        check_is_fitted(self)
        return self.pipeline_.get_feature_names_out(input_features)


class SelectiveTransformerBySkewness(BaseEstimator, TransformerMixin):
    """Applica un transformer solo alle colonne selezionate per asimmetria.

    A ``fit`` calcola la skewness di ogni colonna numerica e partiziona le colonne in
    *asimmetriche* (``|skew| > skew_threshold``) e *simmetriche* (le restanti). A
    seconda di ``apply_to``, il transformer fornito viene addestrato e poi applicato a
    uno dei due gruppi; le colonne non selezionate passano invariate.

    Le colonne con skewness **non definita** (costanti o interamente NaN, per cui
    ``skew`` è ``NaN``) vengono trattate come *simmetriche*: così, con
    ``apply_to='symmetric'`` su uno step di imputazione, non sfuggono al riempimento
    dei valori mancanti (evita NaN residui che farebbero fallire step a valle come la
    PCA).

    :param transformer: Transformer scikit-learn da applicare alle colonne selezionate.
        Deve preservare il numero e l'ordine delle colonne (es. imputer, power transform).
    :param skew_threshold: Soglia assoluta di skewness per distinguere asimmetriche e
        simmetriche.
    :param apply_to: ``'skewed'`` per agire sulle asimmetriche, ``'symmetric'`` per le
        simmetriche.
    """

    def __init__(
        self,
        transformer: BaseEstimator,
        skew_threshold: float = 0.5,
        apply_to: str = "skewed",
    ) -> None:
        self.transformer = transformer
        self.skew_threshold = skew_threshold
        self.apply_to = apply_to

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """Seleziona le colonne per skewness e addestra il transformer su di esse.

        :raises ValueError: Se ``apply_to`` non è ``'skewed'`` né ``'symmetric'``.
        """
        if self.apply_to not in ("skewed", "symmetric"):
            raise ValueError("apply_to deve essere 'skewed' oppure 'symmetric'")

        X = _as_dataframe(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        skew = X.skew(numeric_only=True)
        # NaN > soglia == False: le colonne a skewness indefinita ricadono fra le
        # simmetriche, restando così coperte da un'eventuale imputazione.
        is_skewed = skew.abs() > self.skew_threshold
        selected = skew.index[is_skewed] if self.apply_to == "skewed" else skew.index[~is_skewed]
        self.selected_columns_ = list(selected)

        self.transformer_: BaseEstimator | None = None
        if self.selected_columns_:
            self.transformer_ = clone(self.transformer)
            self.transformer_.fit(X[self.selected_columns_], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Trasforma le colonne selezionate, lascia invariate le altre."""
        check_is_fitted(self)
        X = _as_dataframe(X).copy()
        if self.transformer_ is not None and self.selected_columns_:
            transformed = self.transformer_.transform(X[self.selected_columns_])
            if not isinstance(transformed, pd.DataFrame):
                transformed = pd.DataFrame(
                    transformed, index=X.index, columns=self.selected_columns_
                )
            X[self.selected_columns_] = transformed
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """I nomi delle colonne non cambiano: il transform è colonna-preservante."""
        check_is_fitted(self)
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)
