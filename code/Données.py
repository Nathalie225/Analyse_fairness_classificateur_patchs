# Importation de bibliothèques et de drive

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.image as img
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
%matplotlib inline
from google.colab import drive
drive.mount('/content/drive')

# Lecture des csv du database CBIS-DDSM:  

# 1. Liens vers les fichiers csv 

calctrain_filepath = '/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/calc_case_description_train_set.csv'
calctest_filepath = '/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/calc_case_description_test_set.csv'
dicom_filepath = '/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/dicom_info.csv'
masstest_filepath = '/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/mass_case_description_test_set.csv'
masstrain_filepath = '/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/mass_case_description_train_set.csv'

# 2. Charge les fichiers csv
df_masstrain = pd.read_csv(masstrain_filepath)
df_masstest = pd.read_csv(masstest_filepath)
df_calctrain = pd.read_csv(calctrain_filepath)
df_calctest = pd.read_csv(calctest_filepath)
df = pd.read_csv(dicom_filepath)
patient=pd.read_csv('/content/drive/MyDrive/fairness/patients_paths.csv')

# 3. Raccourcit les noms trop longs
def shorten_paths(df, columns):
    df = df.copy()
    df.rename(columns=lambda x: x.strip(), inplace=True)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].str.split('/').str[-2]
    return df
  
cols_to_shorten = [
    'image file path',
    'ROI mask file path',
    'cropped image file path'
]

df_masstrain = shorten_paths(df_masstrain, cols_to_shorten)
df_masstest  = shorten_paths(df_masstest, cols_to_shorten)
df_calctrain = shorten_paths(df_calctrain, cols_to_shorten)
df_calctest  = shorten_paths(df_calctest, cols_to_shorten)

# 4. Uniformise les annotations
df_calctrain = df_calctrain.rename(columns={'breast density': 'breast_density'})
df_calctest = df_calctest.rename(columns={'breast density': 'breast_density'})

# 5. Dossier avec métadonnées selon: "crop" si patch, "full" si Mammographie et "ROI" si segmentation
df_crop = df[df['SeriesDescription'] == 'cropped images'][['PatientID','image_path', 'SeriesInstanceUID']]
df_full = df[df['SeriesDescription'] == 'full mammogram images'][['PatientID', 'image_path','SeriesInstanceUID']]
df_ROI = df[df['SeriesDescription'] == 'ROI mask images'][['PatientID', 'image_path','SeriesInstanceUID']]

df_crop['PatientID'] = df_crop['PatientID'].str.extract(r'(P_\d{5})')
df_full['PatientID'] = df_full['PatientID'].str.extract(r'(P_\d{5})')
df_ROI['PatientID'] = df_ROI['PatientID'].str.extract(r'(P_\d{5})')

df_crop['image_path'] = df_crop['image_path'].str.split('/').str[-1]
df_full['image_path'] = df_full['image_path'].str.split('/').str[-1]
df_ROI['image_path'] = df_ROI['image_path'].str.split('/').str[-1]

df_masstrain['forme'], df_masstrain['type'] = 'mass', 'train'
df_masstest['forme'], df_masstest['type'] = 'mass', 'test'
df_calctrain['forme'], df_calctrain['type'] = 'calc', 'train'
df_calctest['forme'], df_calctest['type'] = 'calc', 'test'

df_allROI = pd.concat([
    df_masstrain[['ROI mask file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_masstest[['ROI mask file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctrain[['ROI mask file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctest[['ROI mask file path', 'pathology', 'breast_density', 'forme', 'type','image view']]
], ignore_index=True)

df_allcrop = pd.concat([
    df_masstrain[['cropped image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_masstest[['cropped image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctrain[['cropped image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctest[['cropped image file path', 'pathology', 'breast_density', 'forme', 'type','image view']]
], ignore_index=True)

df_allfull = pd.concat([
    df_masstrain[['image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_masstest[['image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctrain[['image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    df_calctest[['image file path', 'pathology', 'breast_density', 'forme', 'type','image view']]
], ignore_index=True)


df_ROI = df_ROI.merge(
    df_allROI [['ROI mask file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    left_on='SeriesInstanceUID',
    right_on='ROI mask file path',
    how='left'
)


df_crop = df_crop.merge(
    df_allcrop[['cropped image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    left_on='SeriesInstanceUID',
    right_on='cropped image file path',
    how='left'
)

df_full = df_full.merge(
    df_allfull[['image file path', 'pathology', 'breast_density', 'forme', 'type','image view']],
    left_on='SeriesInstanceUID',
    right_on='image file path',
    how='left'
)

# Répartition des données selon une validation croisée à 5 fold:

# 1. Compilation des métadonnées pour avoir des folds avec la même proportion de sous-classe par fold 

df_crop_Kfold = df_crop[[
    "image_path",
    "cropped image file path",
    "pathology",
    "breast_density",
    "forme"
]]

df_crop_Kfold = df_crop_Kfold[df_crop_Kfold['breast_density'] != 0]

# 2. Analyse de la répartition de la densité mammaire et du type d'anomalie dans le jeu de données

# Densité mammaire
breast_density_cat = pd.DataFrame({
    "breast_density": [1, 2, 3, 4],
    "Densité de sein":[
        'Densité de type 1',
        'Densité de type 2',
        'Densité de type 3',
        'Densité de type 4']
})

mass_breast_density = df_crop_Kfold['breast_density'].value_counts().reset_index()
mass_breast_density = mass_breast_density.merge(breast_density_cat, on='breast_density', how='left')
mass_breast_density_pathology = df_crop_Kfold.groupby(['pathology', 'breast_density'])['breast_density'].value_counts()
mass_breast_density_pathology = mass_breast_density_pathology.reset_index(name='count')
mass_bd_cat_pathology = mass_breast_density_pathology.merge(mass_breast_density, on='breast_density', how='left')


# Type d'anomalie

form_cat = pd.DataFrame({
    "forme": ['calc', 'mass'],
    "Type anomalie":[
        'De type calcification',
        'De type amas cellulaire']
})

form = df_crop_Kfold['forme'].value_counts().reset_index()
form = form.merge(form_cat, on='forme', how='left')
form_pathology = df_crop_Kfold.groupby(['pathology', 'forme'])['forme'].value_counts()
form_pathology = form_pathology.reset_index(name='count')
form_cat_pathology = form_pathology.merge(form, on='forme', how='left')

# Illustration


plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1) 
sns.barplot(
    data=mass_bd_cat_pathology,
    x='pathology',
    y='count_x',
    hue='Densité de sein'
)
plt.title("Répartition de la densité mammaire selon le diagnostique médical")
plt.axis('on')  

plt.subplot(1, 2, 2)  
sns.barplot(
    data=form_cat_pathology,
    x='pathology',
    y='count_x',
    hue='Type anomalie'
)
plt.title("Répartion du type anomalie selon le diagnostique")
plt.axis('on') 

plt.suptitle("Distribution des caractéristiques selon la pathologie", fontsize=18)
plt.tight_layout()
plt.show()

# 3. Répartition des données au sein des 5 folds:

for i in range(len(df_crop_Kfold)):
# "img" c'est le nom de l'image dans le dossier
    img = df_crop_Kfold.loc[i, "image_path"]
# "fichier" c'est le nom du dossier où se situe l'image
    fichier = df_crop_Kfold.loc[i, "cropped image file path"]
# Extraction du nom de l'image sans l'extension "jpg"
    img_name = os.path.splitext(os.path.basename(img))[0]
# Construction du nouveau nom de l'image avec plus d'information
    new_name = f"{fichier}_{img_name}.jpg"
# Assigne le nouveau nom de l'image dans le csv
    df_crop_Kfold.loc[i, "cropped image file path"] = new_name
# Fusion des images "BENIGN_WITHOUT_CALLBACK" et "BENIGN"
    pathology=df_crop_Kfold.loc[i, "pathology"]
    if pathology=="BENIGN_WITHOUT_CALLBACK":
      df_crop_Kfold.loc[i, "pathology"] = "BENIGN"

# Séparation en deux groupes pour la classification

BENIGN = df_crop_Kfold[df_crop_Kfold["pathology"] == "BENIGN"]
MALIGNANT = df_crop_Kfold[df_crop_Kfold["pathology"] == "MALIGNANT"]
df_all = pd.concat([BENIGN, MALIGNANT], ignore_index=True)

# Supprimer les lignes invalides
df_all = df_all.dropna(subset=["forme", "breast_density", "pathology"])
df_all = df_all[(df_all["forme"].isin(["mass", "calc"])) &
                (df_all["breast_density"].isin([1,2,3,4]))]

# Uniformiser les strings
df_all["pathology"] = df_all["pathology"].str.upper()
df_all["forme"] = df_all["forme"].str.lower()
df_all["breast_density"] = df_all["breast_density"].astype(str)

# Créer la sous-classe pour stratification
df_all["strata"] = df_all["pathology"] + "_" + df_all["forme"] + "_" + df_all["breast_density"]

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = [(train_idx, test_idx) for _, (train_idx, test_idx) in enumerate(skf.split(df_all, df_all["strata"]))]

save_root = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement"

# -----------------------------
# 3. Création des CSV par fold
# -----------------------------
for i, fold in enumerate(folds):
    fold_id = i + 1
    train_idx, test_idx = fold

    fold_dir = os.path.join(save_root, f"crop_{fold_id}fold",Patch_original)
    fold_dir2= os.path.join(fold_dir,Patch_original)
    # création des sous fichiers
    os.makedirs(os.path.join(fold_dir2, "train", "BENIGN"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir2, "train", "MALIGNANT"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir2, "test", "BENIGN"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir2, "test", "MALIGNANT"), exist_ok=True)

    train_df = df_all.iloc[train_idx]
    test_df  = df_all.iloc[test_idx]

    # Sauvegarder CSV
    train_df[train_df["pathology"]=="BENIGN"].to_csv(f"{fold_dir}/Métadonnées/train/benign_train_fold{fold_id}.csv", index=False)
    train_df[train_df["pathology"]=="MALIGNANT"].to_csv(f"{fold_dir}/Métadonnées/train/malignant_train_fold{fold_id}.csv", index=False)
    test_df[test_df["pathology"]=="BENIGN"].to_csv(f"{fold_dir}/Métadonnées/test/benign_test_fold{fold_id}.csv", index=False)
    test_df[test_df["pathology"]=="MALIGNANT"].to_csv(f"{fold_dir}/Métadonnées/test/malignant_test_fold{fold_id}.csv", index=False)

print("\n🎉 Stratified K-Fold terminé sans sous-classe 'unknown' !")




# -----------------------------
# 4. Vérification de la bonne répatition des images
# -----------------------------
# Dataset complet
original_counts = df_all.groupby("strata").size().reset_index(name="count")
original_counts["fold"] = "Original"
original_counts["split"] = "all"
original_counts["percent"] = 100 * original_counts["count"] / original_counts["count"].sum()

# Folds
fold_stats_list = []
for i, fold in enumerate(folds):
    fold_id = i + 1
    train_idx, test_idx = fold
    train_df = df_all.iloc[train_idx]
    test_df  = df_all.iloc[test_idx]

    train_counts = train_df.groupby("strata").size().reset_index(name="count")
    train_counts["fold"] = f"Fold{fold_id}"
    train_counts["split"] = "train"
    train_counts["percent"] = 100 * train_counts["count"] / train_counts["count"].sum()
    fold_stats_list.append(train_counts)

    test_counts = test_df.groupby("strata").size().reset_index(name="count")
    test_counts["fold"] = f"Fold{fold_id}"
    test_counts["split"] = "test"
    test_counts["percent"] = 100 * test_counts["count"] / test_counts["count"].sum()
    fold_stats_list.append(test_counts)

# Combiner toutes les stats
fold_stats_df = pd.concat(fold_stats_list, ignore_index=True)
plot_df = pd.concat([fold_stats_df, original_counts], ignore_index=True)
plot_df["fold_split"] = plot_df["fold"] + "_" + plot_df["split"]

# -----------------------------
# 5. Visualisation bar plot %
# -----------------------------
plt.figure(figsize=(18,6))
sns.barplot(data=plot_df, x="strata", y="count", hue="fold_split", ci=None)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Sous-classe (pathology_forme_density)")
plt.ylabel("Nombre d'images ")
plt.title("Distribution des sous-classes par fold (train/test) et dataset original")
plt.legend(title="Fold + Split", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


plt.figure(figsize=(18,6))
sns.barplot(data=plot_df, x="strata", y="percent", hue="fold_split", ci=None)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Sous-classe (pathology_forme_density)")
plt.ylabel("% d'images ")
plt.title("Distribution des % des sous-classes par fold (train/test) et dataset original")
plt.legend(title="Fold + Split", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
