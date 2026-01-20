# patch normal
# ============================
# IMPORTS
# ============================
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
from PIL import Image
from os.path import basename
import shutil
from imageio import imwrite


# ============================
# LECTURE CSV PATIENTS
# ============================
patient = pd.read_csv('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/CBIS_DDSM_CSV/patients_paths.csv')

# Supprimer lignes avec crop = NaN
patient = patient.dropna(subset=['crop']).copy()

# Garder seulement le nom du fichier
patient["crop"] = [basename(x) for x in patient["crop"]]

print("Aperçu patient CSV :")
print(patient.head())

save_root = "/content/drive/MyDrive/fairness/crop_5fold"



# ============================
# FONCTION RP() : extrait la bounding box de l'image de segmentation
# ============================

def RP(ROI_path):
    # Charger l'image de segmentation
    roi_img = Image.open(ROI_path).convert('L')
    roi_mask = np.array(roi_img) > 0
    y, x = roi_mask.shape

    C = []
    for i in range(y):
        for p in range(x):
            if roi_mask[i, p]:
                C.append((i, p))

    if len(C) == 0:
        raise ValueError(f"ROI vide : {ROI_path}")

    # Initialisation
    ymin = C[0][0]
    ymax = C[0][0]
    xmin = C[0][1]
    xmax = C[0][1]

    # Parcours des points
    for cy, cx in C:
        ymin = min(ymin, cy)
        ymax = max(ymax, cy)
        xmin = min(xmin, cx)
        xmax = max(xmax, cx)

    return (ymax, xmax, xmin, ymin)

# ============================
# EXTRACTION DES PATCHES
# ============================

def extract_patches(ROI, img):
    ymax, xmax, xmin, ymin = ROI
    H, W = img.shape  # dimensions de l'image

  
    R = img[ymin:ymax, xmin:xmax]

    # Coordonnées pour les patches
    y1 = int(0*(ymax-ymin)+ymin)
    y2 = int(0*(xmax-xmin)+xmin)
    y3 = int((xmax-xmin)+xmin)
    y4 = int((ymax-ymin)+ymin)

    def safe_slice(start, end, max_val):
        start = max(0, start)
        end   = min(max_val, end)
        if start >= end:
            return slice(0,0)
        return slice(start, end)

    # Patches sécurisés
    P1 = img[safe_slice(y1-256, y1, H), safe_slice(xmin, xmax, W)]
    P2 = img[safe_slice(ymin, ymax, H), safe_slice(y3, y3+257, W)]
    P3 = img[safe_slice(y4, y4+257, H), safe_slice(xmin, xmax, W)]
    P4 = img[safe_slice(ymin, ymax, H), safe_slice(y2-256, y2, W)]

    # Construire dictionnaire de patches
    patches_dict = {
        "P1_N": P1,
        "P2_N": P2,
        "P3_N": P3,
        "P4_N": P4,
    }

    # Filtrer uniquement les patches non vides
    non_empty_patches = {name: p for name, p in patches_dict.items() if p.size > 0}

    # Coordonnées des patches (pour affichage)
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
# ============================
# Sauvegarde patchs
# ============================
def save_patches(patches, folder, base_name):


    file_root = os.path.splitext(base_name)[0]
    for name, patch in patches.items():
        save_path = os.path.join(folder, f"{file_root}_{name}.jpg")
        imwrite(save_path, patch)
        print(f"Sauvegardé : {save_path}")

def show_patches(full_img_path, patches, boxes):
    img = Image.open(full_img_path).convert("L")
    img_array = np.array(img)
    colors = {"P1_N": "red", "P2_N": "blue", "P3_N": "green", "P4_N": "yellow"}

    n_patches = len(patches)
    fig, axes = plt.subplots(1, n_patches + 1, figsize=(5*(n_patches + 1), 5))
    axes[0].imshow(img_array, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    # Dessiner rectangles
    for name, (x, y, w, h) in boxes.items():
        rect = plt.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor=colors.get(name, "white"),
                             facecolor="none")
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, name, color=colors.get(name, "white"), fontsize=12, weight="bold")

    # Afficher les patches
    for i, (name, patch) in enumerate(patches.items()):
        axes[i+1].imshow(patch, cmap="gray")
        axes[i+1].set_title(f"Patch {name}")
        axes[i+1].axis("off")
    plt.show()



def recupere(crop_name):
    row = patient[patient["crop"] == crop_name]
    if len(row) == 0:
        raise ValueError(f"Image '{crop_name}' introuvable dans le CSV patient_paths.csv")
    full = row["full"].values[0].strip()
    roi  = row["roi"].values[0].strip()
    return full, roi

for crop_name in patient["crop"].tolist():
    full_path, roi_path = recupere(crop_name)


    if os.path.exists(roi_path):

        continue

    filename = os.path.basename(roi_path)
    parent_dir = os.path.dirname(os.path.dirname(roi_path))

    # Liste des dossiers alternatifs
    search_dirs = [
        os.path.join(parent_dir, "BENIGN"),
        os.path.join(parent_dir, "MALIGNANT"),
        os.path.join(parent_dir, "BENIGN_WITHOUT_CALLBACK")
    ]

    found = False
    for alt_dir in search_dirs:
        alt_path = os.path.join(alt_dir, filename)
        # Ne pas déplacer si c'est déjà la destination finale
        if alt_path == roi_path:
            found = True
            break

        if os.path.exists(alt_path):
            os.makedirs(os.path.dirname(roi_path), exist_ok=True)
            shutil.move(alt_path, roi_path)  # Déplacement direct
            print(f"Déplacé : {alt_path} -> {roi_path}")
            found = True
            break

    if not found:
        print(f"Image introuvable dans aucun dossier alternatif : {filename}")




from tqdm import tqdm
def process_patch_list(file_list, dst_folder):
    for crop_name in tqdm(file_list, desc=f"Traitement patches vers {dst_folder}", unit="crop"):
        full_path, roi_path = recupere(crop_name)
        ROI = RP(roi_path)
        img = Image.open(full_path).convert("L")
        img_array = np.array(img)
        patches, boxes = extract_patches(ROI, img_array)
        save_patches(patches, dst_folder, crop_name)
        show_patches(full_path, patches, boxes)
# ============================
# TRAITEMENT DES 5 FOLDS AVEC BARRE GLOBALE
# ============================
base_dst = "/content/drive/MyDrive/fairness/crop_5fold"

for i in [2]:
    fold = i + 1
    print(f"\n=== Traitement du fold {fold}/5 ===")

    #test_mal_dir  = os.path.join(base_dst, f"crop_{fold}fold/test/NORMAL")
    train_mal_dir = os.path.join(base_dst, f"crop_{fold}fold/train/NORMAL")
    #test_ben_dir  = os.path.join(base_dst, f"crop_{fold}fold/test/NORMAL")
    train_ben_dir = os.path.join(base_dst, f"crop_{fold}fold/train/NORMAL")

    # Combine toutes les listes pour une barre globale
    all_lists = [
    #    ("Test Malignant", malignant_folds[i], test_mal_dir),
        ("Train Malignant", malignant_folds_train[i], train_mal_dir),
     #   ("Test Benign", benign_folds[i], test_ben_dir),
        ("Train Benign", benign_folds_train[i], train_ben_dir),
    ]

    for label, crop_list, folder in all_lists:
        print(f"\nTraitement {label} vers {folder} ({len(crop_list)} images)")
        process_patch_list(crop_list, folder)
        drive.flush_and_unmount()
        drive.mount('/content/drive', force_remount=True)
    # Flush et remount du Drive pour éviter les erreurs
    drive.flush_and_unmount()
    drive.mount('/content/drive', force_remount=True)

print("\n=== FIN DU SCRIPT ===")
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)


# csv normal
import os

base_dst = "/content/drive/MyDrive/fairness/crop_5fold"

normal_folds = []
normal_folds_train = []

def clean_filename(filename):
    """
    Nettoie un nom de fichier du type :
    '..._1-114_P1_N.jpg' → '..._1-114.jpg'
    """
    return filename.split("_P")[0] + ".jpg" if "_P" in filename else filename


for i in range(5):
    fold_dir = os.path.join(base_dst, f"crop_{i+1}fold")

    test_n_dir = os.path.join(fold_dir, "test", "NORMAL")
    train_n_dir = os.path.join(fold_dir, "train", "NORMAL")

    cleaned_test = []
    cleaned_train = []

    # Test
    for f in os.listdir(test_n_dir):
        name = clean_filename(f)
        if name not in cleaned_test:   # ici ça marche car cleaned_test existe
            cleaned_test.append(name)

    # Train
    for f in os.listdir(train_n_dir):
        name = clean_filename(f)
        if name not in cleaned_train:
            cleaned_train.append(name)

    normal_folds.append(cleaned_test)
    normal_folds_train.append(cleaned_train)


# Affichage
for i in range(5):
    print(f"Fold {i+1} - Val NORMAL: {len(normal_folds[i])}, "
          f"Train NORMAL: {len(normal_folds_train[i])}")


save_root = "/content/drive/MyDrive/fairness/crop_5fold"


for i in range(5):
    fold_id = i + 1
    print(f"\n=== Sauvegarde Fold {fold_id} ===")

    # Dossiers
    fold_dir = os.path.join(save_root, f"crop_{fold_id}fold")


    # DataFrame filtrés
    valN   = test_df[test_df["cropped image file path"].isin(normal_folds[i])]
    trainN = test_df[test_df["cropped image file path"].isin(normal_folds_train[i])]


    # Sauvegarde CSV
    valN.to_csv(f"{fold_dir}/test/normal_test_fold{fold_id}.csv", index=False)
    trainN.to_csv(f"{fold_dir}/train/normal_train_fold{fold_id}.csv", index=False)


    print(" → CSV enregistrés.")

print("\n🎉 FIN — Tous les CSV K-Fold ont été sauvegardés avec succès !")

drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)
