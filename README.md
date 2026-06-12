# Pre-processing & Feature Engineering — Breast Cancer Wisconsin

![CI](https://github.com/profession-ai-data-engineering-master/profession_ai_data_engineering_progetto6/actions/workflows/ci.yml/badge.svg)

Libreria scikit-learn che costruisce, a partire da un dataset clinico grezzo, un unico
oggetto di preprocessing riutilizzabile: tre pipeline complementari unite in un dataset
*max-feature*. Progetto 6 del Master in Data Engineering di ProfessionAI.

> **Dataset**: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/tunguz/breast-cancer-wisconsin-diagnostic-data) —
> 569 record, 30 feature (29 numeriche + 1 categorica `A/B/C`), target binario benigno/maligno,
> con valori mancanti diffusi.

Il notebook originale di consegna (`profession_ai_data_engineering_progetto6.ipynb`) è
conservato come artefatto del corso; **il prodotto è questo package**, che ne porta i due
transformer custom allo standard di una libreria testata, tipizzata e in CI.

## Il problema

L'obiettivo della consegna è ottenere un **unico oggetto** che racchiuda tutte le fasi di
preprocessing applicabili a tutte le colonne (target escluso), componendo tre pipeline:

| Pipeline | Ambito | Passi |
|----------|--------|-------|
| **1** | addestrata sui soli record positivi (target=1) | imputazione selettiva (media/mediana per simmetria) → simmetrizzazione (Yeo-Johnson) → one-hot encoding → standardizzazione |
| **2** | tutti i record | imputazione selettiva → discretizzazione in 20 bin → encoding ordinale → selezione delle 5 feature più informative (ANOVA F) |
| **3** | solo variabili numeriche | imputazione selettiva → PCA (85% di varianza) → simmetrizzazione → normalizzazione |

Le tre pipeline confluiscono in un `FeatureUnion` (il dataset *max-feature*), pronto come input
per l'addestramento di un modello.

## Architettura

Due transformer custom (estensioni dell'API `BaseEstimator`/`TransformerMixin`) coprono la
logica che le primitive scikit-learn non offrono nativamente:

- **`PipelineWithRowFilter`** — addestra una pipeline interna su un sottoinsieme di record
  selezionato dal target, ma trasforma *tutte* le righe.
- **`SelectiveTransformerBySkewness`** — applica un transformer solo alle colonne numeriche
  selezionate per asimmetria (skewness), lasciando invariate le altre.

```
preprocessing/
├── transformers.py   # i due transformer custom
├── pipelines.py      # build_pipeline_1/2/3 + build_full_pipeline
├── data.py           # load_dataset() + split_features_target()
├── plotting.py       # plot_distribution() (analisi skewness)
└── __main__.py       # demo end-to-end: python -m preprocessing
```

## Scelte di design rilevanti

- **Pipeline 1, semantica "fit sui positivi, transform su tutti".** La pipeline *apprende* i
  parametri esclusivamente dai record positivi (evita che i negativi contaminino media e
  scala), ma li *applica* a tutte le righe. Ne consegue un dataset finale **senza NaN
  strutturali** e l'invariante `fit_transform(X, y) == fit(X, y).transform(X)`.
- **Skewness rivalutata per step** (scelta intenzionale): l'imputazione sceglie media vs
  mediana sulla skewness del dato *grezzo* (la mediana è robusta sulle code), mentre la
  simmetrizzazione agisce sulle colonne ancora asimmetriche *dopo* l'imputazione.
- **Robustezza ai casi degeneri**: colonne costanti o quasi tutte NaN (skewness indefinita)
  vengono comunque imputate; categorie mai viste in inferenza non fanno fallire l'encoding;
  i nomi delle feature sono unici e propagati end-to-end (`get_feature_names_out`).

## Uso

```bash
pip install -r requirements.txt

python -m preprocessing      # scarica il dataset, addestra la full pipeline, salva il .joblib
```

```python
from preprocessing import load_dataset, split_features_target, build_full_pipeline

X, y = split_features_target(load_dataset())
pipeline = build_full_pipeline()
features = pipeline.fit_transform(X, y)   # DataFrame pronto per il modello
```

## Qualità

- **Test**: `pytest` con oracoli espliciti e proprietà ([Hypothesis](https://hypothesis.readthedocs.io/))
  sui transformer — coverage ~97%.
- **Lint & formato**: [ruff](https://docs.astral.sh/ruff/).
- **Type checking**: [mypy](https://mypy-lang.org/) con type hints completi.
- **CI**: GitHub Actions esegue lint, format-check, mypy e test a ogni push/PR.

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && mypy preprocessing && pytest
```
