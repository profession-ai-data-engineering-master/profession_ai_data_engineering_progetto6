"""Test delle pipeline e della loro unione (smoke end-to-end + invarianti)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.pipelines import (
    build_full_pipeline,
    build_pipeline_1,
    build_pipeline_2,
    build_pipeline_3,
)


def test_full_pipeline_no_nan_and_unique_names(
    synthetic_df: pd.DataFrame, target: pd.Series
) -> None:
    out = build_full_pipeline(k=3).fit_transform(synthetic_df, target)
    assert isinstance(out, pd.DataFrame)
    assert int(out.isna().sum().sum()) == 0
    assert out.columns.is_unique
    assert len(out) == len(synthetic_df)


def test_fit_transform_equals_fit_then_transform(
    synthetic_df: pd.DataFrame, target: pd.Series
) -> None:
    a = build_full_pipeline(k=3).fit_transform(synthetic_df, target)
    b = build_full_pipeline(k=3).fit(synthetic_df, target).transform(synthetic_df)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_pipeline1_is_row_filtered_on_positives() -> None:
    assert build_pipeline_1().fit_target_value == 1


def test_pipeline3_numeric_only_no_nan(synthetic_df: pd.DataFrame) -> None:
    out = build_pipeline_3().fit_transform(synthetic_df)
    assert isinstance(out, pd.DataFrame)
    assert int(out.isna().sum().sum()) == 0


def test_ordinal_handles_unseen_category(synthetic_df: pd.DataFrame, target: pd.Series) -> None:
    fitted = build_pipeline_2(k=3).fit(synthetic_df, target)
    unseen = synthetic_df.copy()
    unseen["cat"] = "Z"  # categoria mai vista al fit
    out = fitted.transform(unseen)  # non deve sollevare
    assert len(out) == len(unseen)


def test_pipeline2_multiple_categorical_columns() -> None:
    rng = np.random.RandomState(1)
    n = 40
    df = pd.DataFrame(
        {
            "num": rng.normal(0, 1, n),
            "c1": rng.choice(list("AB"), n),
            "c2": rng.choice(list("XYZ"), n),
        }
    )
    y = pd.Series(rng.randint(0, 2, n))
    out = build_pipeline_2(k=3).fit_transform(df, y)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == n


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_constant_and_mostly_nan_columns_do_not_break_pipeline() -> None:
    """Le colonne degeneri (varianza nulla, quasi tutte NaN) non producono NaN/inf.

    sklearn emette warning attesi su feature costanti (binning, ANOVA F): li silenziamo
    perché qui sono il comportamento sotto test, non un problema.
    """
    rng = np.random.RandomState(2)
    n = 60
    df = pd.DataFrame(
        {
            "normal": rng.normal(0, 1, n),
            "skewed": rng.exponential(2, n),
            "constant": np.ones(n),
            "mostly_nan": [np.nan] * (n - 1) + [3.0],
            "cat": rng.choice(list("ABC"), n),
        }
    )
    y = pd.Series(rng.randint(0, 2, n))
    out = build_full_pipeline(k=3).fit_transform(df, y)
    assert int(out.isna().sum().sum()) == 0
    assert np.isfinite(out.to_numpy()).all()
