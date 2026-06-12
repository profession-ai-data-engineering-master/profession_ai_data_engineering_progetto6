"""Test del caricamento dati e della separazione feature/target."""

from __future__ import annotations

import pandas as pd
import pytest

import preprocessing.data as data
from preprocessing.data import split_features_target


def test_split_features_target() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
    X, y = split_features_target(df)
    assert "target" not in X.columns
    assert list(X.columns) == ["a", "b"]
    assert list(y) == [0, 1]


def test_split_missing_target_raises() -> None:
    with pytest.raises(KeyError):
        split_features_target(pd.DataFrame({"a": [1]}))


def test_load_dataset_delegates_to_read_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(data.pd, "read_csv", lambda url: sentinel)
    assert data.load_dataset("http://fake.local/data.csv") is sentinel
