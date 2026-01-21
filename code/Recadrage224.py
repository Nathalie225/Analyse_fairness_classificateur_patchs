"""
Recadrage des patchs en 224x224 pixels
====================================================
Ce script permet d'homogénéiser la taille des patchs des classes "malin","bénin" et "normal".
Pour cela on réalise un patch de taille 224 x 224 pixels au centre du patch initial si 
la hauteur ou la largeur du patch est supérieur à 224 pixels. 
On illustre ensuite l'homogénéisation de la taille des images.
"""


# ============================================================
# Importations des  bibliothèques
# ============================================================

import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# Ouverture de drive 
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
#                 Fonctions principales
# ============================================================

# ============================================================
# 1. FONCTION crop_images() 
#     Prend en entrée le lien vers les patchs, le lien de sauvegarde des patchs recadrés et la taille du crop final souhaité.
#     Crée un nouveau dossier avec les images recadrées
# ============================================================


def crop_images(image_folder, output_folder, crop_size=224):
    
    # liste des patchs étudiés
    image_files = [f for f in os.listdir(image_folder)]
    
    # traitement image par image
    for img_file in tqdm(image_files, desc=f"Cropping {os.path.basename(image_folder)}"):
        # ouverture du patch
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)
        # récupère la taille du patch
        img_width, img_height = img.size
        
        # si la hauteur ou la largeur du patch est inférieure à 224 pixels alors on pas recadre
        if img_width < crop_size or img_height < crop_size:
            print(f"{img_file} est trop petite pour des crops ({img_width}x{img_height}), copie seule.")
            # on sauvegarde l'image non recadrée dans le nouveau fichier
            img.save(os.path.join(output_folder, img_file))
            continue
            
        # Coordonnées du centre du patch
        
        left = (img_width - crop_size) // 2
        top = (img_height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # récupère le patch central recadré
        
        crop = img.crop((left, top, right, bottom))

        # sauvegarde l'image
        
        crop_name = f"{os.path.splitext(img_file)[0]}_center224.jpg"
        crop.save(os.path.join(output_folder, crop_name))


# ---------------------------------------------------------
#   Application automatique du recadrage sur différents dossiers
# ---------------------------------------------------------

base = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement"

folds = ["1fold", "2fold", "3fold", "4fold", "5fold"]

splits = ["train", "test"]

classes = ["BENIGN","MALIGNANT","NORMAL"]

for fold in folds:
    for split in splits:
        for cls in classes:

            image_folder = f"{base}/{fold}/Patch_original/{split}/{cls}"
            output_folder = f"{base}/{fold}/Patch_224x224pixels/{split}_224/{cls}"

            crop_images(image_folder, output_folder, crop_size=224)




# ============================================================
# 2. FONCTION show_center_crop() 
#     Prend en entrée le lien vers les patchs.
#     Affiche le patch central sur le patch initial.
# ============================================================

def show_center_crop(image_folder, crop_size=224, n_cols=5):
    image_files = [f for f in os.listdir(image_folder)]

    for img_file in tqdm(image_files, desc=f"Visualisation crops {os.path.basename(image_folder)}"):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Si l'image est trop petite pour un crop, on affiche l'image originale
        if img_width < crop_size or img_height < crop_size:
            print(f"{img_file} trop petite ({img_width}x{img_height}), affichage originale.")
            plt.figure(figsize=(6,6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{img_file} (originale)")
            plt.show()
            continue

        # Coordonnées du crop central
        left = (img_width - crop_size) // 2
        top = (img_height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        crop = img.crop((left, top, right, bottom))

        # Affichage côte à côte : image originale + crop
        plt.figure(figsize=(10,5))

        # Image originale avec rectangle du crop
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Original avec crop central")
        # Rectangle du crop
        ax = plt.gca()
        rect = patches.Rectangle((left, top), crop_size, crop_size, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Affichage du crop seul
        plt.subplot(1,2,2)
        plt.imshow(crop)
        plt.axis('off')
        plt.title("Crop central 224x224")

        plt.suptitle(img_file)
        plt.show()


# -----------------------------
#  Exemple d'utilisation
# -----------------------------

image_folder = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_original/1fold/train/MALIGNANT"
show_center_crop(image_folder, crop_size=224)


# ============================================================
# 3. FONCTION taille_moyenne_pixels() 
#     Prend en entrée le lien du dossier.
#     Renvoie les distributions statistiques de la taille des images du dossier.
# ============================================================

def taille_moyenne_pixels(dossier):
    # initialise
    largeurs = []
    hauteurs = []
    # liste des noms des patchs
    fichiers = os.listdir(dossier)

    # Récupère les données de taille sur tous les patchs du dossier
    for f in tqdm(fichiers, desc=f"Traitement {os.path.basename(dossier)}"):
        chemin = os.path.join(dossier, f)
        img = Image.open(chemin)
        largeur, hauteur = img.size
        largeurs.append(largeur)
        hauteurs.append(hauteur)

    # Calcul statistique de la répartition de taille des images du dossier
    
    m_largeur = np.mean(largeurs)
    m_hauteur = np.mean(hauteurs)
    V_largeur = np.var(largeurs)
    V_hauteur = np.var(hauteurs)
    m_pixels = m_largeur * m_hauteur

    return m_largeur, m_hauteur, m_pixels, V_largeur, V_hauteur, largeurs, hauteurs


# ---------------------------------------------------------
#   Comparaison des distributions de taille selon les dossiers (on réalise la comparaison sur les données train du fold1)
# ---------------------------------------------------------

# Initialisation 
U = []   # pour stocker les stats
all_widths = {}   # largeurs par classe
all_heights = {}  # hauteurs par classe

# Dossiers à recadrer et dossier recadré

dossiers = {
    "normal": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_original/1fold/train/NORMAL",
    "benign": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_original/1fold/train/BENIGN",
    "malignant": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_original/1fold/train/MALIGNANT",
    "normal_224": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_224x224pixels/1fold/train/NORMAL",
    "benign_224": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_224x224pixels/1fold/train/BENIGN",
    "malignant_224": "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/Patch_224x224pixels/1fold/train/MALIGNANT",
}

# calcul statistique de taille par dossier et sauvegarde des données
for type_dossier, chemin in dossiers.items():
    larg, haut, pix, vlar, vhaut, Lw, Lh = taille_moyenne_pixels(chemin)
    U.append((larg, haut, pix, vlar, vhaut))

    all_widths[type_dossier] = Lw
    all_heights[type_dossier] = Lh

    print(f"{type_dossier} : {larg:.1f}px × {haut:.1f}px (≈ {pix:.0f} pixels)")
    print(f"    Variances : largeur={vlar:.1f}, hauteur={vhaut:.1f}")
    print("")


# Largeurs 
df_width = pd.DataFrame([
    {"classe": k.replace("_224",""),
     "version": "224" if "_224" in k else "original",
     "largeur": w}
    for k, Lw in all_widths.items()
    for w in Lw
])

# Hauteurs
df_height = pd.DataFrame([
    {"classe": k.replace("_224",""),
     "version": "224" if "_224" in k else "original",
     "hauteur": h}
    for k, Lh in all_heights.items()
    for h in Lh
])

# Visualisation des résultats

plt.figure(figsize=(14,5))

# Largeurs
plt.subplot(1,2,1)
sns.violinplot(x="classe", y="largeur", hue="version", data=df_width,
               palette={"original":"skyblue", "224":"lightcoral"}, dodge=True)
plt.title("Comparaison des largeurs : original vs 224")
plt.xlabel("Classe")
plt.ylabel("Largeur (pixels)")

# Hauteurs
plt.subplot(1,2,2)
sns.violinplot(x="classe", y="hauteur", hue="version", data=df_height,
               palette={"original":"skyblue", "224":"lightcoral"}, dodge=True)
plt.title("Comparaison des hauteurs : original vs 224")
plt.xlabel("Classe")
plt.ylabel("Hauteur (pixels)")

plt.tight_layout()
plt.show()
