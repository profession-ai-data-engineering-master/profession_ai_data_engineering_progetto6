# %% [markdown]
# <a href="https://colab.research.google.com/github/profession-ai-data-engineering-master/profession_ai_data_engineering_progetto6/blob/main/profession_ai_data_engineering_progetto6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Pre-processing di un Dataset di Rilevazione del Tumore al Seno
#
# **Dataset**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/tunguz/breast-cancer-wisconsin-diagnostic-data)
#
# ## Contesto del Progetto
#
# Nel settore sanitario si utilizzano sempre più dati per prendere decisioni informate. Il presente progetto riguarda un dataset di rilevazione del tumore al seno, con l'obiettivo di creare un set di dati pulito e pronto per essere utilizzato nei modelli di machine learning. La pulizia e l'organizzazione dei dati sono fondamentali per migliorare la qualità e le performance dei modelli.
#
# ## Obiettivo del Progetto
#
# L'obiettivo è ottenere un unico oggetto finale che racchiuda tutte le fasi di preprocessing applicabili a tutte le colonne del dataset, eccetto la colonna target. Ciò consente di avere un workflow strutturato e riutilizzabile per la preparazione dei dati prima dell'addestramento del modello. L'utilizzo di un unico oggetto facilita il tracciamento e la riproducibilità del processo di preprocessing.
#
# ## Valore Aggiunto
#
# * Automazione e scalabilità: Le pipeline garantiscono un processo automatizzato e replicabile su nuovi dataset, riducendo i tempi di preparazione del dataset.
# * Ottimizzazione della qualità dei dati: Grazie alla pulizia dei dati e alla gestione delle anomalie, la qualità del dataset sarà migliorata, portando a modelli più robusti.
# * Personalizzazione delle tecniche di pre-processing: Le pipeline offrono flessibilità nell'applicazione di tecniche avanzate come l'analisi della skewness, la PCA (Principal Component Analysis) e la selezione delle variabili più informative.
#
# ## Descrizione delle Pipeline
#
# ### Pipeline 1: Pre-processing per Record con Target = 1
#
# (Questa pipeline si concentra sui soli record in cui il target è pari a 1, ovvero i casi positivi di rilevazione del tumore. La pipeline include:)
#
# 1. Pulizia dei valori mancanti: La pulizia sarà distinta tra variabili asimmetriche e variabili simmetriche; per queste ultime si opterà per metodi di riempimento più standard.
# 2. Simmetrizzazione delle variabili asimmetriche: Per garantire una distribuzione più bilanciata dei dati, si procederà alla simmetrizzazione mediante tecniche appropriate.
# 3. One-Hot Encoding delle variabili categoriche: Tutte le variabili categoriche verranno convertite in formato one-hot encoding, rendendo i dati utilizzabili dal modello di machine learning.
# 4. Riscalatura mediante Standardizzazione: Le variabili numeriche saranno portate a una distribuzione con media zero e deviazione standard pari a uno.
#
# Questa pipeline fornisce un trattamento ottimale dei record positivi, migliorando la coerenza e la qualità del dataset su cui verranno effettuate le analisi.
#
# ### Pipeline 2: Pre-processing per Tutti i Record del Dataset
#
# (Questa pipeline sarà applicata a tutti i record del dataset, con l'obiettivo di trasformare tutte le variabili numeriche e categoriche attraverso i seguenti passaggi:)
#
# 1. Pulizia dei valori mancanti: Verrà adottata una strategia personalizzata per pulire i valori mancanti in modo coerente con la natura delle variabili.
# 2. Discretizzazione a 20 bin delle variabili numeriche: Le variabili numeriche verranno discretizzate in 20 bin per ridurre la complessità del dato e facilitare l'analisi.
# 3. Encoding ordinale delle variabili categoriche: Le variabili categoriche saranno codificate in base a un ordine crescente (A, B, C), mantenendo il valore semantico delle categorie.
# 4. Selezione delle 5 variabili più informative: Al termine della trasformazione verranno selezionate le 5 variabili più significative, migliorando così efficienza e precisione del modello successivo.
#
# ### Pipeline 3: Pre-processing delle Variabili Numeriche
#
# (Questa pipeline è focalizzata sulle variabili numeriche del dataset e prevede i seguenti passaggi:)
#
# 1. Pulizia dei valori mancanti: Come nelle pipeline precedenti, verrà adottato un metodo personalizzato per pulire i valori mancanti delle variabili numeriche.
# 2. Principal Component Analysis (PCA): Verrà applicata la Principal Component Analysis per ridurre il rumore e migliorare le prestazioni del modello.
# 3. Simmetrizzazione: Come nella pipeline 1, si procederà alla simmetrizzazione delle variabili asimmetriche per migliorare la distribuzione dei dati.
# 4. Riscalatura mediante normalizzazione: Infine, le variabili numeriche saranno normalizzate su una scala standard per facilitare il processo di apprendimento del modello.
#
# ## Risultato Finale
#
# Al termine di tutte queste pipeline, verrà creato un oggetto finale che semplificherà la gestione del dataset complesso e l'addestramento di un modello robusto.
#
# ## Conclusione
#
# La pipeline di pre-processing proposta non solo migliora la qualità dei dati, ma ottimizza anche il flusso di lavoro aziendale, dando un contributo significativo all'accuratezza delle previsioni sul tumore al seno.

# %% [markdown]
# # Svolgimento

# %% [markdown]
# All'avvio del progetto mi sono interrogato sui motivi dietro le tre pipeline di pre-processing, apparentemente ridondanti per via delle tecniche diverse applicate agli stessi dati. Il dubbio principale riguardava due punti:
# - La necessità di un unico oggetto finale che concatenasse gli output di tutte le pipeline.
# - L' applicazione della Pipeline 1 esclusivamente ai record con target = 1, con conseguente presenza di valori nulli nelle stesse colonne per i record benigni.
#
# Dopo un attenta analisi sul combinare i tre blocchi in un'unica matrice ho formulato i seguenti possibili benefici di questo approccio:
#
# 1. **Separazione dei domini informativi**: ogni pipeline cattura aspetti statistici differenti consentendo di avere un risultato eterogeneo a livello di contenuto informativo.
#
# 2. **Riduzione del leakage**: l'applicazione condizionata della Pipeline 1 ai soli maligni evita che segnali distintivi contaminino i benigni.
#
#
# 3. **Flessibilità per la data-science**: il dataset max-featured funge da feature-store centralizzato. I data scientist possono selezionare rapidamente sotto-insiemi di colonne coerenti con gli algoritmi prescelti.

# %% [markdown]
# ## Dipendenze e Utility

# %% [markdown]
# Ispirandomi alla risposta ricevuta sul portale di profession ai ho approfondito la soluzione che prevedeva l'utilizzo di oggetti custom, possibilità che offre l'approccio object oriented che offre nativamente sklearn.

# %%
import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    PowerTransformer,
    StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    OrdinalEncoder,
    Normalizer,
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, clone
from joblib import dump, load
from sklearn import set_config
set_config(transform_output="pandas")

# %%
def plotta_distribuzione(
    df: pd.DataFrame,
    columns: list[str] = None,
    wrap: int = 22,
    title_size: int = 9,
) -> None:
    """
    Plotta un istogramma per ogni colonna numerica del DataFrame (o solo quelle specificate),
    colorando il titolo in verde se la distribuzione è circa simmetrica (|skew| < 0.5)
    oppure in rosso se è asimmetrica.

    :param df: DataFrame sorgente.
    :type df: pandas.DataFrame
    :param columns: Elenco delle colonne numeriche da plottare.
                    Se None, vengono usate tutte le colonne numeriche.
    :type columns: list[str], optional
    :param wrap: Numero massimo di caratteri per riga del titolo.
    :type wrap: int
    :param title_size: Dimensione del font dei titoli dei singoli subplot.
    :type title_size: int
    :returns: None
    :rtype: None
    :raises KeyError: Se colonne specificate non esistono nel DataFrame.
    """

    # Auto-seleziona colonne numeriche se non specificate
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing = set(columns) - set(df.columns)
        if missing:
            raise KeyError(f"Colonne mancanti nel DataFrame: {missing}")

        # Filtro per colonne numeriche tra quelle specificate
        columns = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not columns:
            raise ValueError("Nessuna colonna numerica tra quelle specificate.")

    if not columns:
        print("Nessuna colonna numerica da plottare.")
        return

    skew_values = df[columns].skew()
    axes = df[columns].hist(bins="auto", figsize=(12, 8))

    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])

    fig = axes.ravel()[0].figure

    for i, column in enumerate(columns):
        row, col = divmod(i, axes.shape[1])
        ax = axes[row, col]
        skew = skew_values[column]
        color = "green" if abs(skew) <= 0.5 else "red"

        wrapped_title = "\n".join(textwrap.wrap(column, width=wrap))
        ax.set_title(wrapped_title, color=color, fontsize=title_size, pad=8)
        ax.tick_params(axis="both", labelsize=8)

    legend_elements = [
        Patch(facecolor="none", edgecolor="none", label="Distribuzione:"),
        Patch(facecolor="none", edgecolor="green", label="~Simmetrica"),
        Patch(facecolor="none", edgecolor="red", label="Asimmetrica"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    plt.suptitle("Skewness Features")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# %%
class PipelineWithRowFilter(BaseEstimator, TransformerMixin):
    """
    Wrapper che applica una *pipeline* scikit‑learn solo ai record desiderati in
    funzione del target.
    """

    def __init__(
        self,
        pipeline,
        target_value_to_drop: int | float | str = 1,
        fill_value=np.nan,
    ) -> None:
        """
        Costruttore.

        :param pipeline: Oggetto compatibile con l'API di scikit‑learn.
        :type pipeline: sklearn.base.BaseEstimator | sklearn.pipeline.Pipeline
        :param target_value_to_drop: Valore del vettore y che identifica le righe
            da non utilizzare.
        :type target_value_to_drop: int | float | str, *optional*
        :param fill_value: Valore con cui riempire le celle dei record esclusi una
            volta reinseriti nel risultato finale.
        :type fill_value: Any, *optional*
        """

        self.pipeline = pipeline
        self.target_value_to_drop = target_value_to_drop
        self.fill_value = fill_value
        self._mask_to_keep: pd.Series | None = None
        self._output_columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Adatta la pipeline interna esclusivamente sui record scelti.

        :param X: DataFrame di input.
        :type X: pandas.DataFrame
        :param y: Serie di etichette.  Se None nessuna riga viene filtrata.
        :type y: pandas.Series | None
        :returns: L'istanza stessa.
        :rtype: PipelineWithRowFilter
        :raises ValueError: Se la lunghezza di X e y non coincide.
        """

        if y is not None and len(X) != len(y):
            raise ValueError("X e y devono avere la stessa lunghezza")

        if y is not None:
            self._mask_to_keep = y != self.target_value_to_drop
            X_filtered = X.loc[self._mask_to_keep].copy()
            y_filtered = y[self._mask_to_keep]
        else:
            self._mask_to_keep = pd.Series(True, index=X.index)
            X_filtered = X
            y_filtered = None

        transformed = self.pipeline.fit_transform(X_filtered, y_filtered)

        if hasattr(transformed, "columns"):
            self._output_columns = list(transformed.columns)
        else:
            self._output_columns = [f"col_{i}" for i in range(transformed.shape[1])]

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Trasforma i dati e reinserisce i record filtrati.

        :param X: DataFrame di input che deve includere anche le righe da riempire.
        :type X: pandas.DataFrame
        :param y: Ignorato (preservato per compatibilità con l'API).
        :type y: pandas.Series | None
        :returns: DataFrame trasformato che mantiene l'ordine e l'indice di X.
        :rtype: pandas.DataFrame
        :raises RuntimeError: Se fit non è stato chiamato in precedenza.
        """

        if self._mask_to_keep is None:
            raise RuntimeError("fit must be called before transform")

        X_filtered = X.loc[self._mask_to_keep].copy()
        X_transformed = self.pipeline.transform(X_filtered)

        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X_filtered.index,
                columns=self._output_columns,
            )

        empty_rows = pd.DataFrame(
            self.fill_value,
            index=X.index.difference(X_filtered.index),
            columns=self._output_columns,
        )

        X_final = pd.concat([X_transformed, empty_rows]).sort_index()

        return X_final


# %%
class SelectiveTransformerBySkewness(BaseEstimator, TransformerMixin):
    """
    Trasforma soltanto le colonne numeriche scelte in base alla skewness.
    """

    def __init__(
        self,
        transformer,
        skew_threshold: float = 0.5,
        apply_to: str = "skewed",
    ) -> None:
        """
        Costruttore.

        :param transformer: Trasformatore da applicare alle colonne selezionate.
        :type transformer: sklearn.base.BaseEstimator | sklearn.pipeline.Pipeline
        :param skew_threshold: Soglia assoluta di skewness.
        :type skew_threshold: float, *optional*
        :param apply_to: *"skewed"* oppure *"symmetric"*.
        :type apply_to: str, *optional*
        """

        self.transformer = transformer
        self.skew_threshold = skew_threshold
        self.apply_to = apply_to
        self.selected_columns_: list[str] = []
        self.non_selected_columns_: list[str] = []
        self._fitted_transformer = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Seleziona le colonne e fitta il trasformatore.

        :param X: DataFrame con colonne numeriche.
        :type X: pandas.DataFrame
        :param y: Target facoltativo.
        :type y: pandas.Series | None
        :returns: L'istanza stessa.
        :rtype: SelectiveTransformerBySkewness
        :raises ValueError: Se X non è un DataFrame o *apply_to* è invalido.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve essere un DataFrame Pandas")

        skewness = X.skew(numeric_only=True)

        if self.apply_to == "skewed":
            self.selected_columns_ = skewness[skewness.abs() > self.skew_threshold].index.tolist()
            self.non_selected_columns_ = skewness[skewness.abs() <= self.skew_threshold].index.tolist()
        elif self.apply_to == "symmetric":
            self.selected_columns_ = skewness[skewness.abs() <= self.skew_threshold].index.tolist()
            self.non_selected_columns_ = skewness[skewness.abs() > self.skew_threshold].index.tolist()
        else:
            raise ValueError("apply_to deve essere 'skewed' o 'symmetric'")

        if self.selected_columns_:
            self._fitted_transformer = clone(self.transformer)
            self._fitted_transformer.fit(X[self.selected_columns_], y)

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Applica la trasformazione alle colonne selezionate.

        :param X: DataFrame da trasformare.
        :type X: pandas.DataFrame
        :param y: Ignorato.
        :type y: pandas.Series | None
        :returns: DataFrame con colonne trasformate/invariate.
        :rtype: pandas.DataFrame
        :raises ValueError: Se X non è un DataFrame.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X deve essere un DataFrame Pandas")

        X_copy = X.copy()

        if self.selected_columns_ and self._fitted_transformer is not None:
            transformed = self._fitted_transformer.transform(X_copy[self.selected_columns_])
            if not isinstance(transformed, pd.DataFrame):
                transformed = pd.DataFrame(
                    transformed,
                    index=X_copy.index,
                    columns=self.selected_columns_,
                )
            X_copy[self.selected_columns_] = transformed

        return X_copy


# %% [markdown]
# ## Recupero il dataset completo

# %%
df = pd.read_csv("https://proai-datasets.s3.eu-west-3.amazonaws.com/sample_dataset.csv")

# %%
# Separo in features(X) e target (y)
X = df.drop("target",axis=1)
y = df["target"]

# %% [markdown]
# ## Esploro il dataset

# %%
display(X.shape)

# %%
with pd.option_context('display.max_columns', None):
    display(X.describe())

# %% [markdown]
# Considerazioni:
# - Nessun Valore negativo.
# - Nessun valore mancante in target, il resto delle feature li presenta.

# %%
plotta_distribuzione(X)

# %% [markdown]
# La maggior parte delle variabili presenta una distribuzione asimmetrica, tuttavia è necessario valutare questo aspetto dinamicamente durante l'esecuzione della pipeline, poiché fasi precedenti come l'imputazione o altre trasformazioni potrebbero modificarne la simmetria.

# %% [markdown]
# ## Pipeline 1: Pre-processing per Record con Target = 1

# %%
pipeline_1_cat = Pipeline(steps=[
    ("pipeline 1 cat pulizia dei valori mancanti",SimpleImputer(strategy='most_frequent')),
    ("pipeline 1 cat  one hot encoding",OneHotEncoder(sparse_output=False))
])

pipeline_1_num = Pipeline(steps=[
        (
            "pipeline 1 num sym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='mean'),
                apply_to='symmetric'
                )
        ),
        (
            "pipeline 1 num asym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='median'),
                apply_to='skewed'
                )
        ),
        (
            "pipeline 1 num asym simmetrizzazione selettiva",
            SelectiveTransformerBySkewness(
                transformer=PowerTransformer(
                    method='yeo-johnson',
                    standardize=False
                ),
                apply_to='skewed'
                )
        ),
        ("pipeline 1 num riscalatura",StandardScaler())

])

pipeline_1 = PipelineWithRowFilter(
    pipeline= ColumnTransformer(transformers=[
         ("pipeline 1 num",pipeline_1_num,make_column_selector(dtype_include=np.number)),
         ("pipeline 1 cat",pipeline_1_cat,make_column_selector(dtype_exclude=np.number))
         ]),
    target_value_to_drop=0
)

# %% [markdown]
# Pipeline 2: Pre-processing per Tutti i Record del Dataset

# %%
pipeline_2_cat = Pipeline(steps=[
    ("pipeline 2 cat pulizia dei valori mancanti",SimpleImputer(strategy='most_frequent')),
    ("pipeline 2 cat ordinal encoding",OrdinalEncoder(categories=[['A', 'B', 'C']])),
])

pipeline_2_num = Pipeline(steps=[
        (
            "pipeline 2 num sym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='mean'),
                apply_to='symmetric'
                )
        ),
        (
            "pipeline 2 num asym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='median'),
                apply_to='skewed'
                )
        ),
        ("pipeline 2 num binning",KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform'))

])

pipeline_2 = Pipeline(steps=[
    (
        "pipeline 2 preprocessing", ColumnTransformer(transformers=[
            ("pipeline 2 num", pipeline_2_num,make_column_selector(dtype_include=np.number)),
            ("pipeline 2 cat", pipeline_2_cat,make_column_selector(dtype_exclude=np.number)),
            ])
    ),
    ("pipeline 2 selection", SelectKBest(f_classif,k = 5)),
])

# %% [markdown]
# ## Pipeline 3: Pre-processing delle Variabili Numeriche

# %%
pipeline_3_num = Pipeline(steps=[
        (
            "pipeline 3 num sym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='mean'),
                apply_to='symmetric'
                )
        ),
        (
            "pipeline 3 num asym pulizia dei valori mancanti selettiva",
            SelectiveTransformerBySkewness(
                transformer=SimpleImputer(strategy='median'),
                apply_to='skewed'
                )
        ),
        (
            "pipeline 3 num asym simmetrizzazione selettiva",
            SelectiveTransformerBySkewness(
                transformer=PowerTransformer(
                    method='yeo-johnson',
                    standardize=False
                ),
                apply_to='skewed'
                )
        ),
        ("pipeline 3 pca", PCA(n_components=0.85)),
        ("pipeline 3 normalizzazione",Normalizer()),
])

pipeline_3 = ColumnTransformer(transformers=[
            ("pipeline 3 num", pipeline_3_num,make_column_selector(dtype_include=np.number)),
            ])

# %% [markdown]
# ## Pipeline Completa

# %% [markdown]
# Applicazione di FeatureUnion alla fine per crea un dataset max-feature.

# %%
pipeline_completa = FeatureUnion([
    ("pipe1", pipeline_1),
    ("pipe2", pipeline_2),
    ("pipe3", pipeline_3),
])

pipeline_completa.fit(X,y)

# %% [markdown]
# ## Salvataggio della Pipeline

# %%
dump(pipeline_completa, "pipeline_completa.joblib")

# %% [markdown]
# ## Test di utilizzo

# %%
from joblib import load

pipeline_completa_test = load("pipeline_completa.joblib")

X_trasformati = pipeline_completa_test.transform(X)


# %%
with pd.option_context('display.max_columns', None):
    display(X_trasformati)


# %%
with pd.option_context('display.max_columns', None):
    display(X_trasformati.describe(include='all'))

# %%
with pd.option_context('display.max_columns', None):
    plotta_distribuzione(X_trasformati)

# %% [markdown]
# Il dato risulta conforme alle specifiche richeste.
