# Stage ARIA – 3 mois  

## Sujet du stage

**Métriques d’équité et méthodes de réduction des biais algorithmiques appliquées à un classificateur de patchs de mammographie**

### Objectifs
- Analyser les données issues du dataset **CBIS-DDSM**
- Mettre en place un classificateur de patchs de mammographie
- Évaluer l’équité du modèle entraîné à l’aide de métriques de fairness
- Appliquer et comparer différentes méthodes de réduction des biais algorithmiques

---

## Jeu de données

Le projet repose sur le jeu de données **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**

- Source : Kaggle – CBIS-DDSM  
  https://www.kaggle.com/datasets

### Réorganisation des données

Les données ont été **réorganisées** afin de faciliter leur utilisation dans les expérimentations. La version restructurée est accessible via le lien Google Drive suivant :

https://drive.google.com/drive/folders/1aJVvIv6liJcf8Z_EUUGx_wAUQg8apbb1?usp=sharing

L’arborescence se compose de **trois dossiers principaux** :

---

### 1. `Mammographie`

Contient les mammographies complètes, organisées :
- selon le découpage **train / test** original de CBIS-DDSM
- puis selon le **diagnostic** :
  - `BENIGN`
  - `MALIGNANT`
  - `BENIGN_WITHOUT_CALLBACK`

---

### 2. `Segmentation`

Contient les **masques de segmentation** permettant de localiser les anomalies sur les mammographies.

- La structure est identique à celle du dossier `Mammographie`
- Les dossiers `Mammographie` et `Segmentation` sont utilisés conjointement dans le code pour construire la **classe `NORMAL`**

---

### 3. `Patchs_Mammographie_5fold`

Contient les **patchs de mammographie** centrés sur les anomalies annotées par les cliniciens.  
Les données sont organisées selon une **validation croisée en 5 folds**, avec deux sous-dossiers principaux :

#### a. `Entrainement`

- Un sous-dossier par **fold**
- Chaque fold contient :
  - un dossier `Métadonnées` :
    - fichiers CSV indiquant la répartition **train / test**
    - informations sur :
      - la **densité mammaire** (D1 à D4)
      - le **type d’anomalie** (calcification ou masse)
  - `Patch_224x224pixels` :
    - patchs recadrés en **224×224 pixels**
    - organisés par :
      - train / test
      - diagnostic
  - `Patch_original` :
    - patchs avant redimensionnement
    - organisés par :
      - train / test
      - diagnostic

#### b. `Evaluation_Fairness`

- Un sous-dossier par **fold**
- Les patchs "test" y sont répartis selon tous les **sous-groupes sensibles**, définis par :
  - la **densité mammaire** (D1, D2, D3, D4)
  - le **type d’anomalie** :
    - `calc` : calcification
    - `mass` : masse
- Chaque sous-groupe est ensuite séparé selon le **diagnostic**

Cette organisation permet une **évaluation fine de l’équité du modèle** sur différents sous-groupes de population.

---

## Structure du dépôt

*(À compléter : scripts, notebooks, modèles, résultats, etc.)*

