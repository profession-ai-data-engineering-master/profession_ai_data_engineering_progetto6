"""Test della demo end-to-end (``python -m preprocessing``)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import preprocessing.__main__ as demo


def test_main_runs_and_saves_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_df: pd.DataFrame,
    target: pd.Series,
) -> None:
    df = synthetic_df.copy()
    df["target"] = target.to_numpy()
    output = tmp_path / "pipeline.joblib"

    monkeypatch.setattr(demo, "load_dataset", lambda: df)
    monkeypatch.setattr(demo, "OUTPUT_PATH", output)

    demo.main()

    assert output.exists()
