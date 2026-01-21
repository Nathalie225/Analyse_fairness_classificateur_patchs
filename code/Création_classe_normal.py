"""
Création_Classe_Normal
====================================================
Ce script décrit le processus de récupération des triplets d’images 
composés d’une mammographie, de son masque de segmentation et du patch associé.
À partir de ce triplet, quatre patchs dits normaux sont extraits,correspondant 
aux régions situées au-dessus, en dessous, à gauche et à droite du patch d’intérêt.
"""

# ============================================================
# Importations des  bibliothèques
# ============================================================
import os
import shutil
from os.path import basename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imwrite
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, models, transforms

# ============================================================
# Ouverture de drive 
# ============================================================
from google.colab import drive
drive.mount("/content/drive")
    


# ============================================================
#                        Données 
#
#     Chargement du fichier CSV "patients_paths.csv" contenant,
#      pour chaque patient, les chemins des triplets d’images
#
# ============================================================
patient = pd.read_csv('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/patients_paths.csv')

# Supprimer lignes avec crop (=patch) = NaN
patient = patient.dropna(subset=['crop']).copy()

# Garder seulement le nom du fichier
patient["crop"] = [basename(x) for x in patient["crop"]]

# ============================================================
#                 Fonctions principales
# ============================================================


# ============================================================
# 1. FONCTION RP() 
#     Prend en entrée le lien vers l'image de segmentation
#     Renvoie les corrodonnées de l'anomalie (ymax, xmax, xmin, ymin) 
# ============================================================

def RP(ROI_path):
    # ouverture de l'image de segmentation 
    roi_img = Image.open(ROI_path).convert('L')
    roi_mask = np.array(roi_img) > 0
    y, x = roi_mask.shape
    # C matrice avec 1 si anomalie et 0 sinon
    C = []
    for i in range(y):
        for p in range(x):
            if roi_mask[i, p]:
                C.append((i, p))

    # Initialisation des coordonnées
    ymin = C[0][0]
    ymax = C[0][0]
    xmin = C[0][1]
    xmax = C[0][1]

    # Assigne les coordonnées de l'anomalie
    for cy, cx in C:
        ymin = min(ymin, cy)
        ymax = max(ymax, cy)
        xmin = min(xmin, cx)
        xmax = max(xmax, cx)

    return (ymax, xmax, xmin, ymin)

# ============================================================
# 2. FONCTION extract_patches() 
#     Prend en entrée les coordonnées de l'anomalie (ROI) et la mammographie (img)
#     Renvoie les patches autour de l'anomalie et leur coordonnées
# ============================================================

def extract_patches(ROI, img):
    # coordonnées de l'anomalie
    ymax, xmax, xmin, ymin = ROI
    # taille de la mammographie et anomalie ( R )
    H, W = img.shape  
    R = img[ymin:ymax, xmin:xmax]
    # Coordonnées de l'anomalie
    y1 = int(ymin)
    y2 = int(xmin)
    y3 = int(xmax)
    y4 = int(ymax)
    # fonction qui vérifie que les limites du patchs sont bien dans la mammographie
    def safe_slice(start, end, max_val):
        start = max(0, start)
        end   = min(max_val, end)
        return slice(start, end)
    # patchs autour de l'anomalie
    P1 = img[safe_slice(y1-256, y1, H), safe_slice(xmin, xmax, W)]
    P2 = img[safe_slice(ymin, ymax, H), safe_slice(y3, y3+257, W)]
    P3 = img[safe_slice(y4, y4+257, H), safe_slice(xmin, xmax, W)]
    P4 = img[safe_slice(ymin, ymax, H), safe_slice(y2-256, y2, W)]
    # sauvegarde des patchs
    patches_dict = {
        "P1_N": P1,
        "P2_N": P2,
        "P3_N": P3,
        "P4_N": P4,
    }
    
    # Filtrer uniquement les patches non vides
    non_empty_patches = {name: p for name, p in patches_dict.items() if p.size > 0}

    # Coordonnées des patches pour affichage
    boxes = {}
    for name, p in non_empty_patches.items():
        if name == "P1_N":
            boxes[name] = (xmin, max(y1-256,0), xmax - xmin, p.shape[0])
        elif name == "P2_N":
            boxes[name] = (y3, ymin, p.shape[1], ymax - ymin)
        elif name == "P3_N":
            boxes[name] = (xmin, y4, xmax - xmin, p.shape[0])
        elif name == "P4_N":
            boxes[name] = (max(y2-256,0), ymin, p.shape[1], ymax - ymin)

    return non_empty_patches, boxes
    
# ============================================================
# 3. FONCTION save_patches() 
#     Prend en entrée les patches, la destination et le nom des images                        
#     Sauvegarde les patches dans le bon dossier
# ============================================================


def save_patches(patches, folder , base_name):
    file_root = os.path.splitext(base_name)[0]
    for name, patch in patches.items():
        save_path = os.path.join(folder, f"{file_root}_{name}.jpg")
        imwrite(save_path, patch)


# ============================================================
# 4. FONCTION show_patches() 
#     Prend en entrée le lien vers la mammographie, les patches et leur coordonnées                      
#     Affiche les patches séparément ainsi que leur position correspondante sur la mammographie
# ============================================================


def show_patches(full_img_path, patches, boxes):
    # lecture de la mammographie
    img = Image.open(full_img_path).convert("L")
    img_array = np.array(img)
    # Attribution des carrées de couleurs à chaque patch
    colors = {"P1_N": "red", "P2_N": "blue", "P3_N": "green", "P4_N": "yellow"}
    # affiche la mammographie
    n_patches = len(patches)
    fig, axes = plt.subplots(1, n_patches + 1, figsize=(5*(n_patches + 1), 5))
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    # affiche les patches sur la mammographie
    for name, (x, y, w, h) in boxes.items():
        rect = plt.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor=colors.get(name, "white"),
                             facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, name, color=colors.get(name, "white"), fontsize=12, weight="bold")
    #affiche le patch séparément
    for i, (name, patch) in enumerate(patches.items()):
        axes[i+1].imshow(patch, cmap="gray")
        axes[i+1].set_title(f"Patch {name}")
        axes[i+1].axis("off")
    plt.show()

# ============================================================
# 5. FONCTION recupere() 
#     Prend le nom du patch                     
#     Renvoie le lien pour la mammographie et l'image de segmentation associée
# ============================================================

def recupere(crop_name):
    # ligne du csv PATIENT contenant le patch d'intérêt
    row = patient[patient["crop"] == crop_name]
    full = row["full"].values[0].strip()
    roi  = row["roi"].values[0].strip()
    return full, roi


# ============================================================
# 6. FONCTION process_patch_list() 
#     Prend la liste des patchs et le lien de sauvegarde                  
#     Sauvegarde les patches de la classe normal extraits de chaque patch
# ===========================================================

def process_patch_list(file_list, dst_folder):
    for crop_name in tqdm(file_list, desc=f"Traitement patches vers {dst_folder}", unit="crop"):
        # recupère les liens vers la mammographie et l'image de segmentation
        full_path, roi_path = recupere(crop_name)
        # recupère les coordonnées de l'anomalie
        ROI = RP(roi_path)
        # lecture de la mammographie
        img = Image.open(full_path).convert("L")
        img_array = np.array(img)
        # récupère les patches et leurs coordonnées
        patches, boxes = extract_patches(ROI, img_array)
        # enregistre les patches
        save_patches(patches, dst_folder, crop_name)
        # affiche les patches
        show_patches(full_path, patches, boxes)
        
# ============================================================
#                        Traitement 5 fold
# ===========================================================
# Répertoire de sauvegarde

save_root = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement"


# ============================================================
#   1. Lecture des métadonnées des autres dossiers
# ===========================================================
malignant_test_folds  = {}
malignant_train_folds = {}
benign_test_folds     = {}
benign_train_folds    = {}

for i in range(5):
    fold_id = i + 1
    fold_dir = os.path.join(save_root, f"crop_{fold_id}fold",Métadonnées)

    # Chargement des CSV et récupère la colonne correpsondant au nom des patchs
    malignant_test_folds[fold_id]  = pd.read_csv(
        f"{fold_dir}/test/MALIGNANT/malignant_test_fold{fold_id}.csv"
    )['cropped image file path']
    malignant_train_folds[fold_id] = pd.read_csv(
        f"{fold_dir}/train/MALIGNANT/malignant_train_fold{fold_id}.csv"
    )['cropped image file path']
    benign_test_folds[fold_id] = pd.read_csv(
        f"{fold_dir}/test/BENIGN/benign_test_fold{fold_id}.csv"
    )['cropped image file path']
    benign_train_folds[fold_id] = pd.read_csv(
        f"{fold_dir}/train/BENIGN/benign_train_fold{fold_id}.csv"
    )['cropped image file path']


# ============================================================
#   2. Création des patchs de la classe normale
# ===========================================================
for i in [5]:
    fold = i + 1
    # liens de sauvegarde
    test_mal_dir  = os.path.join(save_root, f"crop_{fold}fold/test/NORMAL")
    train_mal_dir = os.path.join(save_root, f"crop_{fold}fold/train/NORMAL")
    test_ben_dir  = os.path.join(save_root, f"crop_{fold}fold/test/NORMAL")
    train_ben_dir = os.path.join(save_root, f"crop_{fold}fold/train/NORMAL")
    # Combine toutes les listes pour une barre globale
    all_lists = [
       ("Test Malignant", malignant_test_folds[i], test_mal_dir),
        ("Train Malignant", malignant_train_folds[i], train_mal_dir),
       ("Test Benign", benign_test_folds[i], test_ben_dir),
        ("Train Benign", benign_train_folds[i], train_ben_dir),
    ]
    # parcourt tous les patchs
    for label, crop_list, folder in all_lists:
        print(f"\nTraitement {label} vers {folder} ({len(crop_list)} images)")
        process_patch_list(crop_list, folder)

# ============================================================
#   3. Génération csv pour la classe normale
# ===========================================================
normal_folds = []
normal_folds_train = []

# fonction qui permet de récupèrer le nom des patchs de la classe normale

def clean_filename(filename):
    """
    Nettoie un nom de fichier du type :
    '..._1-114_P1_N.jpg' → '..._1-114.jpg'
    """
    return filename.split("_P")[0] + ".jpg" if "_P" in filename else filename

# liste de liste contenant le nom des patchs
for i in range(5):
    fold_dir = os.path.join(save_root, f"crop_{i+1}fold")
    test_n_dir = os.path.join(fold_dir, "test", "NORMAL")
    train_n_dir = os.path.join(fold_dir, "train", "NORMAL")

    cleaned_test = []
    cleaned_train = []

    for f in os.listdir(test_n_dir):
        name = clean_filename(f)
        if name not in cleaned_test:   
            cleaned_test.append(name)

    for f in os.listdir(train_n_dir):
        name = clean_filename(f)
        if name not in cleaned_train:
            cleaned_train.append(name)

    normal_folds.append(cleaned_test)
    normal_folds_train.append(cleaned_train)

for i in range(5):
    print(f"Fold {i+1} - Val NORMAL: {len(normal_folds[i])}, "
          f"Train NORMAL: {len(normal_folds_train[i])}")


# Compile les métadonnées de tous les patchs (on prend le fold 1)
fold = 1
test_df = compile(
    pd.read_csv(os.path.join(
        save_root,
        f"crop_{fold}fold",
        "Métadonnées",
        "test",
        "MALIGNANT",
        f"malignant_test_fold{fold}.csv"
    )),
    pd.read_csv(os.path.join(
        save_root,
        f"crop_{fold}fold",
        "Métadonnées",
        "train",
        "MALIGNANT",
        f"malignant_train_fold{fold}.csv"
    )),
    pd.read_csv(os.path.join(
        save_root,
        f"crop_{fold}fold",
        "Métadonnées",
        "test",
        "BENIGN",
        f"benign_test_fold{fold}.csv"
    )),
    pd.read_csv(os.path.join(
        save_root,
        f"crop_{fold}fold",
        "Métadonnées",
        "train",
        "BENIGN",
        f"benign_train_fold{fold}.csv"
    ))
)

# récupère les métadonnées de chaque dossier de la classe normal et sauvegarde

for i in range(5):
    fold_id = i + 1
    fold_dir = os.path.join(save_root, f"crop_{fold_id}fold")
    # récupère les métadonnées
    valN   = test_df[test_df["cropped image file path"].isin(normal_folds[i])]
    trainN = test_df[test_df["cropped image file path"].isin(normal_folds_train[i])]

    valN.to_csv(f"{fold_dir}/test/normal_test_fold{fold_id}.csv", index=False)
    trainN.to_csv(f"{fold_dir}/train/normal_train_fold{fold_id}.csv", index=False)


