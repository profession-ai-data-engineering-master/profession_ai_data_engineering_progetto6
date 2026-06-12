"""Fixture condivise e configurazione dei test.

Imposta il backend matplotlib headless (Agg) prima che qualsiasi test importi il
package (e quindi pyplot), così i test grafici non aprono finestre.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402  (import dopo matplotlib.use per forzare il backend)
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

N_ROWS = 100


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """DataFrame realistico: numeriche simmetriche/asimmetriche + 1 categorica, con NaN."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "sym1": rng.normal(0, 1, N_ROWS),
            "sym2": rng.normal(10, 3, N_ROWS),
            "skew1": rng.exponential(2, N_ROWS),
            "skew2": rng.exponential(3, N_ROWS) ** 1.4,
            "cat": rng.choice(list("ABC"), N_ROWS),
        }
    )
    for col in ("sym1", "skew1", "skew2", "cat"):
        idx = rng.choice(N_ROWS, size=15, replace=False)
        df.loc[idx, col] = np.nan
    return df


@pytest.fixture
def target() -> pd.Series:
    """Target binario allineato a ``synthetic_df`` (indice 0..N_ROWS-1)."""
    rng = np.random.RandomState(7)
    return pd.Series(rng.randint(0, 2, N_ROWS), name="target")
