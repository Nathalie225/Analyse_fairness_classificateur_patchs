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

Le projet repose sur le jeu de données **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**.

- **Source** : Kaggle – CBIS-DDSM  
  https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

---

### Réorganisation des données

Les données ont été **réorganisées** afin de faciliter leur exploitation dans les différentes expérimentations.  
La version restructurée est accessible via le lien Google Drive suivant :

https://drive.google.com/drive/folders/1aJVvIv6liJcf8Z_EUUGx_wAUQg8apbb1?usp=sharing

L’arborescence se compose de **trois dossiers principaux** :

---

### 1. `Mammographie`

Ce dossier contient les mammographies complètes, organisées :
- selon le découpage **train / test** original du dataset CBIS-DDSM
- puis selon le **diagnostic** :
  - `BENIGN`
  - `MALIGNANT`
  - `BENIGN_WITHOUT_CALLBACK`

---

### 2. `Segmentation`

Ce dossier contient les **masques de segmentation** permettant de localiser les anomalies sur les mammographies.

- La structure est identique à celle du dossier `Mammographie`
- Les dossiers `Mammographie` et `Segmentation` sont utilisés conjointement dans le code pour construire la **classe `NORMAL`**

---

### 3. `Patchs_Mammographie_5fold`

Ce dossier contient les **patchs de mammographie** centrés sur les anomalies annotées par les cliniciens.  
Les données sont organisées selon une **validation croisée en 5 folds**, et se divisent en deux sous-dossiers principaux :

#### a. `Entrainement`

- Un sous-dossier par **fold**
- Chaque fold contient :
  - un dossier `Métadonnées` :
    - fichiers CSV décrivant la répartition **train / test**
    - informations relatives :
      - à la **densité mammaire** (D1 à D4)
      - au **type d’anomalie** (calcification ou masse)
  - `Patch_224x224pixels` :
    - patchs recadrés en **224 × 224 pixels**
    - organisés par :
      - train / test
      - diagnostic
  - `Patch_original` :
    - patchs avant redimensionnement
    - organisés par :
      - train / test
      - diagnostic

#### b. `Résultats_Entrainement`

Ce dossier contient les **meilleurs modèles obtenus pour chaque fold**, selon différentes stratégies d’apprentissage :
- `Sans_pondération` : modèles entraînés sans pondération des classes
- `Avec_pondération` : modèles entraînés avec pondération des classes dans la fonction de perte
- `In-processing` : modèles entraînés à l’aide d’une méthode d’équité de type *in-processing*


 #### c. `Evaluation_Fairness`

Ce dossier contient les données de test organisées pour l’évaluation de l’équité.

- Un sous-dossier par **fold**
- Les patchs de test sont répartis selon les **sous-groupes sensibles**, définis par :
  - la **densité mammaire** : D1, D2, D3, D4
  - le **type d’anomalie** :
    - `calc` : calcification
    - `mass` : masse
- Chaque sous-groupe est ensuite séparé selon le **diagnostic**

Cette organisation permet une **évaluation fine de l’équité du modèle** sur différents sous-groupes de population.


 #### d. `Résultats_Evaluation`

Ce dossier regroupe les résultats des évaluations d’équité :
- `Sans_pondération` :
  - `csv` : tableaux contenant les métriques d’évaluation (performances globales et métriques d’équité)
  - `images` : visualisations et mises en forme graphiques des résultats
- `Avec_pondération` :
  - résultats obtenus avec pondération des classes
- `In-processing` :
  - résultats obtenus avec la méthode adversariale

---

## Structure du dépôt

Le dépôt est organisé en plusieurs modules de code :
- **`Création_classe_normal.py`**
  Cconstruction de la classe `NORMAL` à partir des mammographies complètes et des masques de segmentation.
  Sauvegarde des métadonnées associée à cette classe.
  
- **`donnees.py`**  
  Analyse exploratoire des données, répartition en **5 folds**, en respectant une distribution homogène des sous-classes.
  
- **`recadrage224`**  
  Recadrage des patchs en **224 × 224 pixels** et comparaison statistique entre patchs recadrés et non recadrés.

- **`entrainement`**  
  Entraînement des modèles selon une validation croisée à 5 folds, avec ou sans pondération des classes dans la fonction de perte.

- **`evaluation`**  
  Évaluation des modèles entraînés à l’aide des métriques :
  - **Disparate Impact**
  - **Equality of Odds**

- **`adversarial`**  
  Implémentation d’une méthode d’équité de type **in-processing**, basée sur une branche adversariale intégrée au modèle.

