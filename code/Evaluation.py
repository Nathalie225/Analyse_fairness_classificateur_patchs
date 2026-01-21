"""
Evaluation de l'équité des modèles entrainés
====================================================
Ce script utilise les métriques d'équité : Disparate impact et equality of odds
On à un premier script qui permet d'afficher la matrice de confusion globale ( les 5 folds confondus)
Puis un second script qui permet d'afficher les matrices de confusion par fold et par sous groupes
Ceci permet aussi de récupérer les métriques permettant de calculer le disparate impact et l'equality of odds
"""

# ============================================================
# Importations des  bibliothèques
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError

# ============================================================
#             Ouverture de drive 
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ================================================================
#      I. PARAMÈTRES GÉNÉRAUX POUR UNE ETUDE GENERALE
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lien de sauvegarde des données calculées
base_eval0='/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Evaluation/Avec_pondération/Images'
eval0='/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Evaluation/Avec_pondération/csv'

# Diagnostic
classes = ["BENIGN", "MALIGNANT", "NORMAL"]

# calcul global sur les données test
subgroup1=["global"]
base_eval1='/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement'


results = []      # Liste pour stocker les métriques par fold/sous-groupe
cm_folds = []     # Liste pour stocker les matrices de confusion par fold

# ================================================================
#                   FONCTIONS UTILITAIRES
# ================================================================

# ============================================================
# 1. FONCTION load_model_for_fold() 
#     Prend en entrée le numéro de fold 
#     Renvoie le modèle entrainé
# ============================================================

def load_model_for_fold(fold, device):

    #initialise le modèle, on prend comme exemple le modèle issu de l'entrainement avec pondération des classes
    
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 3)  # 3 classes
    )

    # charge le modèle
    
    model_path = f"/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/{fold}fold/Avec_pondération/best_model.pth"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model
    
# ============================================================
#               Transformation des images test
# ============================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================================================================
#   2. FONCTION matrice() 
#     Prend en entrée le lien vers les données entrainement,
#     le modèle entrainé et la fonction de transformation
#     Renvoie la matrice de confusion, le taux de faux positif, 
#     de faux négatifs et la probabilité d'assigner le label "malin"
#     calculé à partir des labels initiaux ou prédits par le modèle
# ================================================================

def matrice(eval_path, model, val_transform):
    # Chargement dataset par fold
    val_dataset = datasets.ImageFolder(
        os.path.join(eval_path, "test_224"),
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    class_names = val_loader.dataset.classes
    
    # initialisation
    all_preds = []
    all_labels = []
    probs_Y1 = []
    true_Y1 = []
    number = []

    device = next(model.parameters()).device
    model.eval()

    # Évaluation batch par batch
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Évaluation : {os.path.basename(eval_path)}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs_Y1.extend(probs[:, 1].cpu().numpy())
            true_Y1.extend((labels == 1).cpu().numpy())
            number.extend([1]*len(inputs))

    # calcule de métriques
    
    P_global = sum(probs_Y1)/len(number)
    T_global = sum(true_Y1)/len(number)

    # Matrice de confusion
    
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Pourcentage par ligne
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100
    
    # Affichage de la matrice de confusion par fold
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion nombre – {os.path.basename(eval_path)}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(os.path.join(base_eval0, "conf_matrix_nombre.png"))
    plt.close()

    # En pourcentage
    annot = np.array([[f"{cm[i,j]:.0f}\n({cm_percentage[i,j]:.1f}%)" for j in range(n_classes)] for i in range(n_classes)])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_percentage, annot=annot, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion % – {os.path.basename(eval_path)}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(os.path.join(base_eval0, "conf_matrix_percentage.png"))
    plt.close()

    #  Calcul du taux du taux de faux positif et de faux négatif
    
    TN = cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]
    FP = cm[0,1] + cm[2,1]
    FN = cm[1,0] + cm[1,2]
    TP = cm[1,1]
    FPR = FP / (FP+TN) if (FP+TN)>0 else 0
    FNR = FN / (FN+TP) if (FN+TP)>0 else 0

    return P_global, T_global, FPR, FNR, cm

# ================================================================
#              PIPELINE D'ÉVALUATION GLOBAL
# ================================================================

for fold in tqdm(range(1, 6), desc="Évaluation des 5 folds"):
    
    # Charger le modèle
    model = load_model_for_fold(fold, device) 
    
    # Initialise la matrice de confusion
    cm_fold_total = np.zeros((len(classes), len(classes)), dtype=float)

    for subgroup1 in tqdm(subgroups, desc=f"Fold {fold} | Sous-groupes", leave=False):
        eval_path = f"{base_eval1}/{fold}fold/Patch_224x224pixels"
        P_global, T_global, FPR, FNR, cm = matrice(eval_path, model, val_transform)
        cm_fold_total += cm

        results.append({
            "fold": fold,
            "subgroup": subgroup,
            "P_global": P_global,
            "T_global": T_global,
            "FPR": FPR,
            "FNR": FNR
        })

    # Compile les matrices pour la matrice de confusion globale sur 5 fold
    
    cm_folds.append(cm_fold_total)


# ================================================================
#         VISUALISATION MATRICE DE CONFUSION GLOBALE
# ================================================================

# Somme des matrices sur les 5 folds
cm_total = np.sum(cm_folds, axis=0)  

# Pourcentage par ligne 
cm_percentage_total = cm_total / cm_total.sum(axis=1, keepdims=True) * 100

# Moyenne et écart-type du nombre par fold
cm_mean = cm_total / len(cm_folds)
cm_std  = np.std(cm_folds, axis=0)

# Annotation nombre ± écart-type + %
n_classes = len(classes)
annot = np.empty((n_classes, n_classes), dtype=object)
for i in range(n_classes):
    for j in range(n_classes):
        annot[i,j] = f"{cm_mean[i,j]:.1f} ± {cm_std[i,j]:.1f}\n({cm_percentage_total[i,j]:.1f}%)"

# Affichage de la matrice globale sur 5 fold

plt.figure(figsize=(8,6))
sns.heatmap(cm_percentage_total, annot=annot, fmt="", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Matrice de confusion globale – Moyenne ± Écart-type et % par ligne")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig(os.path.join(base_eval0, "conf_matrix_globale_pourcentage.png"))
plt.show()


# ================================================================
#      II. PARAMÈTRES GÉNÉRAUX POUR UNE ETUDE PAR SOUS-GROUPE
# ================================================================

# lien vers les images classées par sous groupe selon le fold
base_eval='/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Evaluation_Fairness'

# Nombre de fold
folds = range(1,6)

# Différents sous-groupes étudiés
subgroups = ["calc", "mass","D1","D2","D3","D4",
             "D1_calc","D1_mass","D2_calc","D2_mass",
             "D3_calc","D3_mass","D4_calc","D4_mass"]

# Initialisation
results = []  
cm_folds = {} 




# ================================================================
#   3. FONCTION matrice() 
#     Prend en entrée le lien vers les données d'évaluation,
#     le modèle entrainé, la fonction de transformation le fold étudié et la sous-classe.
#     Renvoie la matrice de confusion, le taux de faux positif, 
#     de faux négatifs et la probabilité d'assigner le label "malin"
#     calculé à partir des labels initiaux ou prédits par le modèle
# ================================================================

def matrice(eval_path, model, val_transform,fold,subgroup):
    
    # charge les données
    val_dataset = datasets.ImageFolder(eval_path,  transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # initialisation
    all_preds, all_labels = [], []
    probs_Y1, true_Y1 = [], []
    device = next(model.parameters()).device
    model.eval()
    
    # evaluation du modèle
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Évaluation : {os.path.basename(eval_path)}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs_Y1.extend(probs[:, 1].cpu().numpy())
            true_Y1.extend((labels == 1).cpu().numpy())

    n = len(all_labels)
    P_global = sum(probs_Y1) / n
    T_global = sum(true_Y1) / n
    class_names=val_dataset.classes

    # ===== MATRICE =====
    n_classes = len(val_dataset.classes)
    cm = np.zeros((n_classes,n_classes))
    for t,p in zip(all_labels, all_preds):
        cm[t,p] += 1

    # Pourcentage par ligne
    cm_percentage = cm / cm.sum(axis=1, keepdims=True) * 100

    # Heatmap nombre
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion nombre – {os.path.basename(eval_path)}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(os.path.join(base_eval0, "conf_matrix_nombre.png"))
    plt.close()

    # Heatmap pourcentage
    annot = np.array([[f"{cm[i,j]:.0f}\n({cm_percentage[i,j]:.1f}%)" for j in range(n_classes)] for i in range(n_classes)])
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_percentage, annot=annot, fmt="", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice de confusion % – {os.path.basename(eval_path)}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(os.path.join(base_eval0, "conf_matrix_percentage.png"))
    plt.close()


    TN = cm[0,0] + cm[0,2] + cm[2,0] + cm[2,2]
    FP = cm[0,1] + cm[2,1]
    FN = cm[1,0] + cm[1,2]
    TP = cm[1,1]
    FPR = FP / (FP+TN) if (FP+TN)>0 else 0
    FNR = FN / (FN+TP) if (FN+TP)>0 else 0

    return cm, val_dataset.classes, P_global, T_global, FPR, FNR,n

# ================================================================
#                PIPELINE D'ÉVALUATION
# ================================================================

for fold in tqdm(folds, desc="Évaluation folds"):
    model = load_model_for_fold(fold, device)
    cm_folds[fold] = {}
    for subgroup in tqdm(subgroups, desc=f"Fold {fold} sous-groupes", leave=False):
        
        # lien vers les données d'évaluation du fold appartenant à la sous-classe d'intérêt
        eval_path = os.path.join(base_eval, f"crop_{fold}fold", subgroup)
        cm, cls, P_global, T_global, FPR, FNR,n = matrice(eval_path, model, val_transform,fold,subgroup)

        # sauvegarde de la matrice de confusion pour la sous classe
        cm_folds[fold][subgroup] = cm

        # stockage métriques
        results.append({
            "fold": fold,
            "subgroup": subgroup,
            "Nombre":n,
            "P_global": P_global,
            "T_global": T_global,
            "FPR": FPR,
            "FNR": FNR
        })

        # ===== Heatmap par fold/sous-groupe =====
        cm_percentage = cm / cm.sum(axis=1, keepdims=True)*100
        annot = np.array([[f"{int(cm[i,j])}\n({cm_percentage[i,j]:.1f}%)" for j in range(len(cls))] for i in range(len(cls))])
        plt.figure(figsize=(6,5))
        sns.heatmap(cm_percentage, annot=annot, fmt="", cmap="Blues", xticklabels=cls, yticklabels=cls, cbar_kws={'label':'% par ligne'})
        plt.title(f"Fold {fold} – {subgroup}")
        plt.savefig(os.path.join(base_eval0, f"conf_matrix_{subgroup}_fold{fold}.png"))
        plt.close()



# ================================================================
# 5) MATRICES GLOBALES PAR SOUS-GROUPE
# ================================================================


for subgroup in subgroups:
    
    # vérifier présence dans au moins un fold
    mats = [cm_folds[fold][subgroup] for fold in folds if subgroup in cm_folds[fold]]
    if len(mats)==0:
        continue
    mats = np.array(mats)

    cm_total = mats.sum(axis=0)
    cm_mean  = mats.mean(axis=0)
    cm_std   = mats.std(axis=0)
    cm_percentage = cm_total / cm_total.sum(axis=1, keepdims=True)*100

    # annotation : moyenne ± std + %
    n_classes = cm_total.shape[0]
    annot = np.empty((n_classes,n_classes),dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            annot[i,j] = f"{cm_mean[i,j]:.1f} ± {cm_std[i,j]:.1f}\n({cm_percentage[i,j]:.1f}%)"

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percentage, annot=annot, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes,
                cbar_kws={'label':'% par ligne'})
    plt.title(f"Matrice globale – {subgroup} (5 folds)")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(os.path.join(base_eval0, f"conf_matrix_global_{subgroup}.png"))
    plt.close()

# ================================================================
#                   EXPORT CSV MÉTRIQUES
# ================================================================

# Sauvegarde les métriques
df_results = pd.DataFrame(results)
output_path = os.path.join(base_eval0, "metrics_subgroups_folds.csv")
df_results.to_csv(output_path, index=False)


# Colonnes numériques sur lesquelles calculer moyenne et variance
numeric_cols = ["Nombre", "P_global", "T_global", "FPR", "FNR"]

# Agrégation sur les 5 folds → groupby sur "subgroup" uniquement
stats_df = df.groupby("subgroup")[numeric_cols].agg(["mean", "var"])

# Aplatir les colonnes
stats_df.columns = [f"{col}_{stat}" for col, stat in stats_df.columns]

# Remettre l’index normal
stats_df = stats_df.reset_index()

# Sauvegarde les métriques statistiques
output_path = os.path.join(base_eval0, "metrics_subgroups_5fold_stats.csv")
stats_df.to_csv(output_path, index=False)


# ============================================================
#                  VISUALISATION DU DISPARATE IMPACT 
# ============================================================

# Fonction qui renvoie les ratios entre probabilités préalablement calculées

def compute_ratio_and_var(df, mean_col, var_col):
    df_ratio = df[["subgroup", mean_col, var_col]].set_index("subgroup")
    mu = df_ratio[mean_col].to_numpy()
    var = df_ratio[var_col].to_numpy()

    ratio_matrix = mu.reshape(-1, 1) / mu.reshape(1, -1)
    var_matrix = (var.reshape(-1, 1) / (mu.reshape(1, -1) ** 2)
                  + (mu.reshape(-1, 1) ** 2) * var.reshape(1, -1) / (mu.reshape(1, -1) ** 4))

    ratio_df = pd.DataFrame(ratio_matrix, index=df_ratio.index, columns=df_ratio.index).round(2)
    var_df = pd.DataFrame(var_matrix, index=df_ratio.index, columns=df_ratio.index).round(2)

    annot_df = ratio_df.astype(str) + "\n(" + var_df.astype(str) + ")"
    return ratio_df, var_df, annot_df

colors = [(1.0, 0.4, 0.4), (1.0, 1.0, 1.0), (0.4, 0.6, 1.0)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# --- Calcul des matrices ---
ratio_P, var_P, annot_P = compute_ratio_and_var(stats_df, "P_global_mean", "P_global_var")
ratio_T, var_T, annot_T = compute_ratio_and_var(stats_df, "T_global_mean", "T_global_var")

# --- Matrice comparative : T > P et les deux < 0.8 ---
comparison_mask = (ratio_T > ratio_P) & (ratio_T < 0.8) & (ratio_P < 0.8)
comparison_df = comparison_mask.astype(int)  # 1 si condition vraie, 0 sinon
cmap_comp = LinearSegmentedColormap.from_list("comp_cmap", [(1,1,1), (1,0.6,0.2)], N=2)  # blanc = 0, orange = 1

# Affichage

plt.figure(figsize=(12, 18)) 

# 1. Disparate impact calculé avec les labels prédits par le modèle

plt.subplot(3, 1, 1)
sns.heatmap(ratio_P, annot=annot_P, fmt="", cmap=cmap, center=1.0,
            linewidths=0.5, linecolor='gray', cbar_kws={'shrink':0.8})
plt.title("P_global_mean + variance", fontsize=16)
plt.xlabel("Dénominateur", fontsize=14)
plt.ylabel("Numérateur", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 2. Disparate impact calculé avec les labels du dataset

plt.subplot(3, 1, 2)
sns.heatmap(ratio_T, annot=annot_T, fmt="", cmap=cmap, center=1.0,
            linewidths=0.5, linecolor='gray', cbar_kws={'shrink':0.8})
plt.title("T_global_mean + variance", fontsize=16)
plt.xlabel("Dénominateur", fontsize=14)
plt.ylabel("Numérateur", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 3. Mise en évidence des sous-groupes avec un disparate impact vis-à-vis du sous groupe "D1-calc" calculé par le modèle inférieur à celui issu des labels du dataset et inférieur à 0.8

plt.subplot(3, 1, 3)
sns.heatmap(comparison_df, annot=comparison_df, fmt="", cmap=cmap_comp, cbar=False,
            linewidths=0.5, linecolor='gray')
plt.title("T > P et T,P < 0.8", fontsize=16)
plt.xlabel("Dénominateur", fontsize=14)
plt.ylabel("Numérateur", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# Colonnes numériques sur lesquelles calculer moyenne et variance
numeric_cols = ["Nombre", "P_global", "T_global", "FPR", "FNR"]

# Agrégation sur les 5 folds → groupby sur "subgroup" uniquement
stats_df = df.groupby("subgroup")[numeric_cols].agg(["mean", "var"])

# Aplatir les colonnes
stats_df.columns = [f"{col}_{stat}" for col, stat in stats_df.columns]

# Remettre l’index normal
stats_df = stats_df.reset_index()

# Sauvegarde
output_path = os.path.join("/content/drive/MyDrive/fairness/crop_5fold/eval/resultats/global", "metrics_subgroups_5fold_stats.csv")
stats_df.to_csv(output_path, index=False)


# 4. Comparaison du disparate impact avant et après application du modèle

# Récupérer les sous-groupes
subgroups = ratio_P.index.tolist()

# Créer les listes de valeurs et variances
rP = ratio_P.loc['D1_calc'].tolist()
rT = ratio_T.loc['D1_calc'].tolist()
varP = var_P.loc['D1_calc'].tolist()
varT = var_T.loc['D1_calc'].tolist()

# Création du graphique
plt.figure(figsize=(12,6))

for i, subgroup in enumerate(subgroups):

    # T_global à droite (rouge)
    plt.errorbar(i - 0.1, rT[i], yerr=varT[i], fmt='o', color='red', capsize=5, label='T_global' if i==0 else "")
    # P_global à gauche (bleu)
    plt.errorbar(i + 0.1, rP[i], yerr=varP[i], fmt='o', color='blue', capsize=5, label='P_global' if i==0 else "")


plt.xticks(range(len(subgroups)), subgroups, rotation=45)
plt.xlabel("Subgroup")
plt.ylabel("Ratio")
plt.title("Ratios P_global et T_global avec barres de variance pour D1_calc")
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================
#                  VISUALISATION DE EQUALITY OF ODDS 
# ============================================================

# Palette inversée rouge → bleu
cmap = "coolwarm_r"

# Sous-groupes
subgroups = stats_df['subgroup'].values

# Fonction pour différences ± écart-type
def compute_diff_and_std(stats_df, mean_col, var_col, decimals=2):
    mu = stats_df[mean_col].values
    var = stats_df[var_col].values

    mu_i = mu.reshape(-1, 1)
    mu_j = mu.reshape(1, -1)

    # Différences
    diff_matrix = mu_i - mu_j

    # Écart-type de la différence
    std_matrix = np.sqrt(var.reshape(-1, 1) + var.reshape(1, -1))

    # Annotations texte sur deux lignes
    annot = np.empty(diff_matrix.shape, dtype=object)
    for i in range(diff_matrix.shape[0]):
        for j in range(diff_matrix.shape[1]):
            if i == j:
                annot[i, j] = "0"
            else:
                annot[i, j] = f"{diff_matrix[i, j]:.{decimals}f}\n±{std_matrix[i,j]:.{decimals}f}"
    return diff_matrix, std_matrix, annot

# Calcul des matrices
diff_TPR, std_TPR, annot_TPR = compute_diff_and_std(stats_df, "TPR_mean", "TPR_var")
diff_FPR, std_FPR, annot_FPR = compute_diff_and_std(stats_df, "FPR_mean", "FPR_var")

# Dimensions figure
n = diff_TPR.shape[0]
fig_w = max(12, 0.6 * n)
fig_h = max(10, 0.6 * n)
plt.figure(figsize=(fig_w, 2 * fig_h))

# Style uniforme
sns.set_style("white")  # fond blanc
sns.set_context("talk")  # texte pour rapport
plt.rcParams.update({'font.family': 'DejaVu Sans'})

# Heatmap TPR
plt.subplot(2, 1, 1)
sns.heatmap(
    diff_TPR,
    annot=annot_TPR,
    fmt="",
    cmap=cmap,
    center=0,
    linewidths=0.5,
    linecolor="gray",
    xticklabels=subgroups,
    yticklabels=subgroups,
    cbar_kws={'label': 'Différence ± écart-type'},
    annot_kws={"size": 10, "weight": "bold"}
)
plt.title("Equality of Odds — TPR difference ± std", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

# Heatmap FPR
plt.subplot(2, 1, 2)
sns.heatmap(
    diff_FPR,
    annot=annot_FPR,
    fmt="",
    cmap=cmap,
    center=0,
    linewidths=0.5,
    linecolor="gray",
    xticklabels=subgroups,
    yticklabels=subgroups,
    cbar_kws={'label': 'Différence ± écart-type'},
    annot_kws={"size": 10, "weight": "bold"}
)
plt.title("Equality of Odds — FPR difference ± std", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.tight_layout(pad=3)
plt.show()
