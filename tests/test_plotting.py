"""Test dell'utility di visualizzazione (backend Agg, nessuna finestra)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.plotting import plot_distribution


def test_plot_distribution_returns_figure() -> None:
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": np.arange(50.0), "b": rng.exponential(2, 50)})
    fig = plot_distribution(df, show=False)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        plot_distribution(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), columns=["zzz"], show=False)


def test_plot_no_numeric_column_raises() -> None:
    with pytest.raises(ValueError, match="numerica"):
        plot_distribution(pd.DataFrame({"c": ["A", "B"]}), show=False)
