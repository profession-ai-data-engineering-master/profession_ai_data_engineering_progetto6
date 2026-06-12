# Pre-processing di un Dataset di Rilevazione del Tumore al Seno

**Dataset**: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/tunguz/breast-cancer-wisconsin-diagnostic-data)

## Contesto del Progetto

Nel settore sanitario si utilizzano sempre più dati per prendere decisioni informate. Il presente progetto riguarda un dataset di rilevazione del tumore al seno, con l'obiettivo di creare un set di dati pulito e pronto per essere utilizzato nei modelli di machine learning. La pulizia e l'organizzazione dei dati sono fondamentali per migliorare la qualità e le performance dei modelli.

## Obiettivo del Progetto

L'obiettivo è ottenere un unico oggetto finale che racchiuda tutte le fasi di preprocessing applicabili a tutte le colonne del dataset, eccetto la colonna target. Ciò consente di avere un workflow strutturato e riutilizzabile per la preparazione dei dati prima dell'addestramento del modello. L'utilizzo di un unico oggetto facilita il tracciamento e la riproducibilità del processo di preprocessing.

## Valore Aggiunto

* Automazione e scalabilità: Le pipeline garantiscono un processo automatizzato e replicabile su nuovi dataset, riducendo i tempi di preparazione del dataset.
* Ottimizzazione della qualità dei dati: Grazie alla pulizia dei dati e alla gestione delle anomalie, la qualità del dataset sarà migliorata, portando a modelli più robusti.
* Personalizzazione delle tecniche di pre-processing: Le pipeline offrono flessibilità nell'applicazione di tecniche avanzate come l'analisi della skewness, la PCA (Principal Component Analysis) e la selezione delle variabili più informative.

## Descrizione delle Pipeline

### Pipeline 1: Pre-processing per Record con Target = 1

(Questa pipeline si concentra sui soli record in cui il target è pari a 1, ovvero i casi positivi di rilevazione del tumore. La pipeline include:)

1. Pulizia dei valori mancanti: La pulizia sarà distinta tra variabili asimmetriche e variabili simmetriche; per queste ultime si opterà per metodi di riempimento più standard.
2. Simmetrizzazione delle variabili asimmetriche: Per garantire una distribuzione più bilanciata dei dati, si procederà alla simmetrizzazione mediante tecniche appropriate.
3. One-Hot Encoding delle variabili categoriche: Tutte le variabili categoriche verranno convertite in formato one-hot encoding, rendendo i dati utilizzabili dal modello di machine learning.
4. Riscalatura mediante Standardizzazione: Le variabili numeriche saranno portate a una distribuzione con media zero e deviazione standard pari a uno.

Questa pipeline fornisce un trattamento ottimale dei record positivi, migliorando la coerenza e la qualità del dataset su cui verranno effettuate le analisi.

### Pipeline 2: Pre-processing per Tutti i Record del Dataset

(Questa pipeline sarà applicata a tutti i record del dataset, con l'obiettivo di trasformare tutte le variabili numeriche e categoriche attraverso i seguenti passaggi:)

1. Pulizia dei valori mancanti: Verrà adottata una strategia personalizzata per pulire i valori mancanti in modo coerente con la natura delle variabili.
2. Discretizzazione a 20 bin delle variabili numeriche: Le variabili numeriche verranno discretizzate in 20 bin per ridurre la complessità del dato e facilitare l'analisi.
3. Encoding ordinale delle variabili categoriche: Le variabili categoriche saranno codificate in base a un ordine crescente (A, B, C), mantenendo il valore semantico delle categorie.
4. Selezione delle 5 variabili più informative: Al termine della trasformazione verranno selezionate le 5 variabili più significative, migliorando così efficienza e precisione del modello successivo.

### Pipeline 3: Pre-processing delle Variabili Numeriche

(Questa pipeline è focalizzata sulle variabili numeriche del dataset e prevede i seguenti passaggi:)

1. Pulizia dei valori mancanti: Come nelle pipeline precedenti, verrà adottato un metodo personalizzato per pulire i valori mancanti delle variabili numeriche.
2. Principal Component Analysis (PCA): Verrà applicata la Principal Component Analysis per ridurre il rumore e migliorare le prestazioni del modello.
3. Simmetrizzazione: Come nella pipeline 1, si procederà alla simmetrizzazione delle variabili asimmetriche per migliorare la distribuzione dei dati.
4. Riscalatura mediante normalizzazione: Infine, le variabili numeriche saranno normalizzate su una scala standard per facilitare il processo di apprendimento del modello.

## Risultato Finale

Al termine di tutte queste pipeline, verrà creato un oggetto finale che semplificherà la gestione del dataset complesso e l'addestramento di un modello robusto.

## Conclusione

La pipeline di pre-processing proposta non solo migliora la qualità dei dati, ma ottimizza anche il flusso di lavoro aziendale, dando un contributo significativo all'accuratezza delle previsioni sul tumore al seno.
