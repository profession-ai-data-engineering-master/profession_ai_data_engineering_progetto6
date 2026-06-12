"""Utility di visualizzazione per l'analisi esplorativa della skewness."""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

# Soglia oltre la quale una distribuzione è considerata asimmetrica.
SKEW_THRESHOLD = 0.5


def plot_distribution(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    wrap: int = 22,
    title_size: int = 9,
    show: bool = True,
) -> Figure:
    """Disegna un istogramma per ogni colonna numerica, colorando il titolo per skewness.

    Il titolo è verde se la distribuzione è circa simmetrica (``|skew| <= 0.5``), rosso
    se asimmetrica.

    :param df: DataFrame sorgente.
    :param columns: Colonne da plottare; se ``None`` usa tutte le numeriche.
    :param wrap: Larghezza massima (caratteri) dei titoli.
    :param title_size: Dimensione del font dei titoli.
    :param show: Se ``True`` chiama ``plt.show()`` (disattivare nei test/headless).
    :returns: La figura matplotlib generata.
    :raises KeyError: Se una colonna richiesta non esiste.
    :raises ValueError: Se nessuna colonna numerica è disponibile.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Colonne mancanti nel DataFrame: {missing}")
        columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

    if not columns:
        raise ValueError("Nessuna colonna numerica da plottare.")

    skew_values = df[columns].skew()
    # bins="auto" è valido a runtime (numpy), ma gli stub di pandas lo tipano stretto.
    axes = np.atleast_2d(df[columns].hist(bins="auto", figsize=(12, 8)))  # type: ignore[arg-type]
    fig = axes.ravel()[0].figure

    for i, column in enumerate(columns):
        row, col = divmod(i, axes.shape[1])
        ax = axes[row, col]
        color = "green" if abs(skew_values[column]) <= SKEW_THRESHOLD else "red"
        ax.set_title("\n".join(textwrap.wrap(column, width=wrap)), color=color, fontsize=title_size)
        ax.tick_params(axis="both", labelsize=8)

    legend_elements = [
        Patch(facecolor="none", edgecolor="green", label="~Simmetrica"),
        Patch(facecolor="none", edgecolor="red", label="Asimmetrica"),
    ]
    fig.legend(
        handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False
    )
    fig.suptitle("Skewness delle feature")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if show:
        plt.show()
    return fig
