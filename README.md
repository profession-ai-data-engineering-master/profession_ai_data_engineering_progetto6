# Pre-processing & Feature Engineering — Breast Cancer Wisconsin

![CI](https://github.com/profession-ai-data-engineering-master/profession_ai_data_engineering_progetto6/actions/workflows/ci.yml/badge.svg)

Libreria scikit-learn che costruisce, a partire da un dataset clinico grezzo, un unico
oggetto di preprocessing riutilizzabile: tre pipeline complementari unite in un dataset
*max-feature*. Progetto 6 del Master in Data Engineering di ProfessionAI.

> **Dataset**: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/tunguz/breast-cancer-wisconsin-diagnostic-data) —
> 569 record, 30 feature (29 numeriche + 1 categorica `A/B/C`), target binario benigno/maligno,
> con valori mancanti diffusi.

Lo svolgimento originale era un notebook Colab; **il prodotto è questo package**, che ne
riprende i due transformer scikit-learn custom portandoli allo standard di una libreria
testata, tipizzata e in CI.

## La consegna

Testo della consegna del corso, riportato fedelmente.

### Contesto del progetto

Nel settore sanitario si utilizzano sempre più dati per prendere decisioni informate. Il
presente progetto riguarda un dataset di rilevazione del tumore al seno, con l'obiettivo di
creare un set di dati pulito e pronto per essere utilizzato nei modelli di machine learning.
La pulizia e l'organizzazione dei dati sono fondamentali per migliorare la qualità e le
performance dei modelli.

### Obiettivo del progetto

L'obiettivo è ottenere un unico oggetto finale che racchiuda tutte le fasi di preprocessing
applicabili a tutte le colonne del dataset, eccetto la colonna target. Ciò consente di avere
un workflow strutturato e riutilizzabile per la preparazione dei dati prima dell'addestramento
del modello. L'utilizzo di un unico oggetto facilita il tracciamento e la riproducibilità del
processo di preprocessing.

### Valore aggiunto

* **Automazione e scalabilità**: le pipeline garantiscono un processo automatizzato e
  replicabile su nuovi dataset, riducendo i tempi di preparazione del dataset.
* **Ottimizzazione della qualità dei dati**: grazie alla pulizia dei dati e alla gestione delle
  anomalie, la qualità del dataset sarà migliorata, portando a modelli più robusti.
* **Personalizzazione delle tecniche di pre-processing**: le pipeline offrono flessibilità
  nell'applicazione di tecniche avanzate come l'analisi della skewness, la PCA (Principal
  Component Analysis) e la selezione delle variabili più informative.

### Descrizione delle pipeline

**Pipeline 1 — Pre-processing per record con target = 1.** Si concentra sui soli record in cui
il target è pari a 1, ovvero i casi positivi di rilevazione del tumore. La pipeline include:

1. **Pulizia dei valori mancanti**: distinta tra variabili asimmetriche e simmetriche; per
   queste ultime si opta per metodi di riempimento più standard.
2. **Simmetrizzazione delle variabili asimmetriche**: per garantire una distribuzione più
   bilanciata dei dati, mediante tecniche appropriate.
3. **One-Hot Encoding delle variabili categoriche**: tutte le variabili categoriche vengono
   convertite in formato one-hot encoding, rendendo i dati utilizzabili dal modello.
4. **Riscalatura mediante Standardizzazione**: le variabili numeriche sono portate a una
   distribuzione con media zero e deviazione standard pari a uno.

**Pipeline 2 — Pre-processing per tutti i record del dataset.** Applicata a tutti i record, con
l'obiettivo di trasformare tutte le variabili numeriche e categoriche:

1. **Pulizia dei valori mancanti**: strategia personalizzata, coerente con la natura delle
   variabili.
2. **Discretizzazione a 20 bin delle variabili numeriche**: per ridurre la complessità del dato
   e facilitare l'analisi.
3. **Encoding ordinale delle variabili categoriche**: codifica in base a un ordine crescente
   (A, B, C), mantenendo il valore semantico delle categorie.
4. **Selezione delle 5 variabili più informative**: al termine della trasformazione, per
   migliorare efficienza e precisione del modello successivo.

**Pipeline 3 — Pre-processing delle variabili numeriche.** Focalizzata sulle sole variabili
numeriche:

1. **Pulizia dei valori mancanti**: come nelle pipeline precedenti, con metodo personalizzato.
2. **Principal Component Analysis (PCA)**: per ridurre il rumore e migliorare le prestazioni del
   modello.
3. **Simmetrizzazione**: come nella Pipeline 1, delle variabili asimmetriche.
4. **Riscalatura mediante normalizzazione**: le variabili numeriche sono normalizzate su una
   scala standard per facilitare il processo di apprendimento.

### Risultato finale

Al termine di tutte queste pipeline viene creato un oggetto finale che semplifica la gestione del
dataset complesso e l'addestramento di un modello robusto.

### Conclusione

La pipeline di pre-processing proposta non solo migliora la qualità dei dati, ma ottimizza anche
il flusso di lavoro aziendale, dando un contributo significativo all'accuratezza delle previsioni
sul tumore al seno.

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
