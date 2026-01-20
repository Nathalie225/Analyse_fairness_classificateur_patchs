
# Stage ARIA – 3mois- Centre Borelli - 

##  Sujet du stage

**Métriques d’équité et méthodes de réduction des biais algorithmiques appliquées à un classificateur de patchs de mammographie.**  

**Objectifs :**  
- Analyse des données issues du dataset CBIS-DDSM
- Mise en place d'un classificateur de patchs de mammographie
- Évaluer l’équité du modèle entrainé.  
- Appliquer des méthodes de réduction des biais.

---

## Jeu de données

Nous utilisons le jeu de données **CBIS-DDSM (Breast Cancer Image Dataset)**, disponible sur [Kaggle CBIS-DDSM](https://www.kaggle.com/datasets).

###  Réorganisation des données

Les données ont été réorganisées pour faciliter l’utilisation et sont accessibles via un **lien Google Drive** (à insérer ici).  

| Dossier       | Contenu |
|---------------|---------|
| `folds/`      | Patchs de mammographies répartis selon 5 folds pour validation croisée (train/test). Chaque fold contient un CSV avec les métadonnées des patchs. |
| `full/`       | Mammographies entières. |
| `ROI/`        | Segmentation des anomalies (régions d’intérêt annotées par les cliniciens). |

---

## Structure du dépôt
