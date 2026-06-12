"""Test dei due transformer custom: oracoli espliciti + proprietà (Hypothesis)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.pipelines import _imputation_block
from preprocessing.transformers import PipelineWithRowFilter, SelectiveTransformerBySkewness

# Strategia: DataFrame numerico con valori finiti nel range, oppure NaN; almeno 20 righe.
_finite = st.one_of(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)


def _numeric_df(cols: tuple[str, ...] = ("a", "b", "c")) -> st.SearchStrategy:
    return data_frames(
        columns=[column(c, dtype=float, elements=_finite) for c in cols],
        index=range_indexes(min_size=20, max_size=60),
    )


# --- SelectiveTransformerBySkewness -----------------------------------------------------


@given(df=_numeric_df())
@settings(deadline=None, max_examples=50)
def test_selective_leaves_unselected_columns_untouched(df: pd.DataFrame) -> None:
    transformer = SelectiveTransformerBySkewness(
        SimpleImputer(strategy="mean", keep_empty_features=True), apply_to="skewed"
    )
    out = transformer.fit_transform(df)
    for col in df.columns:
        if col not in transformer.selected_columns_:
            pd.testing.assert_series_equal(out[col], df[col])


@given(df=_numeric_df())
@settings(deadline=None, max_examples=50)
def test_imputation_block_removes_all_nans(df: pd.DataFrame) -> None:
    """Regressione del bug skew=NaN: costanti/quasi-tutto-NaN restano comunque imputate."""
    pipe = Pipeline(_imputation_block())
    out = pipe.fit_transform(df)
    assert not out.isna().to_numpy().any()


def test_selective_invalid_apply_to_raises() -> None:
    with pytest.raises(ValueError, match="apply_to"):
        SelectiveTransformerBySkewness(SimpleImputer(), apply_to="nope").fit(
            pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        )


def test_selective_feature_names_preserved() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    transformer = SelectiveTransformerBySkewness(
        SimpleImputer(strategy="mean"), apply_to="symmetric"
    ).fit(df)
    assert list(transformer.get_feature_names_out()) == ["a", "b"]


def test_selective_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        SelectiveTransformerBySkewness(SimpleImputer(), apply_to="skewed").transform(
            pd.DataFrame({"a": [1.0]})
        )


# --- PipelineWithRowFilter --------------------------------------------------------------


def test_row_filter_transforms_all_rows() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]})
    y = pd.Series([1, 1, 0, 0])
    out = PipelineWithRowFilter(StandardScaler(), fit_target_value=1).fit(df, y).transform(df)
    assert len(out) == 4
    assert list(out.index) == [0, 1, 2, 3]


def test_row_filter_learns_only_from_selected_rows() -> None:
    """No-leakage: cambiare i record esclusi non altera i parametri appresi."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0]})
    y = pd.Series([1, 1, 1, 1, 0, 0])
    fitted_a = PipelineWithRowFilter(StandardScaler(), fit_target_value=1).fit(df, y)

    df_mod = df.copy()
    df_mod.loc[[4, 5], "a"] = -999.0  # modifico solo i record esclusi (target=0)
    fitted_b = PipelineWithRowFilter(StandardScaler(), fit_target_value=1).fit(df_mod, y)

    np.testing.assert_allclose(fitted_a.pipeline_.mean_, fitted_b.pipeline_.mean_)


def test_row_filter_none_uses_all_rows() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1, 1, 0, 0])
    fitted = PipelineWithRowFilter(StandardScaler(), fit_target_value=None).fit(df, y)
    np.testing.assert_allclose(fitted.pipeline_.mean_, [2.5])


def test_row_filter_raises_when_filter_empty() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([0, 0])
    with pytest.raises(ValueError, match="non seleziona"):
        PipelineWithRowFilter(StandardScaler(), fit_target_value=1).fit(df, y)


def test_row_filter_feature_names() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1, 1, 0, 0])
    fitted = PipelineWithRowFilter(StandardScaler(), fit_target_value=1).fit(df, y)
    assert list(fitted.get_feature_names_out()) == ["a"]
