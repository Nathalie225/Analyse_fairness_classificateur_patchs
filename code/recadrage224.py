# ---------------------------------------------------------
# PIPELINE COMPLET DE CROPPING ET ANALYSE D'IMAGES
# ---------------------------------------------------------

import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

# -----------------------------
# MONTAGE GOOGLE DRIVE
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')
# -----------------------------
# FONCTION : CROP DES IMAGES
# -----------------------------
def crop_images(image_folder, output_folder, crop_size=224):
    image_files = [f for f in os.listdir(image_folder)]

    # BARRE DE PROGRESSION
    for img_file in tqdm(image_files, desc=f"Cropping {os.path.basename(image_folder)}"):

        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size

        if img_width < crop_size or img_height < crop_size:
            print(f"{img_file} est trop petite pour des crops ({img_width}x{img_height}), copie seule.")
            img.save(os.path.join(output_folder, img_file))
            continue

        n_crops_x = (img_width + crop_size - 1) // crop_size
        n_crops_y = (img_height + crop_size - 1) // crop_size

        step_x = (img_width - crop_size) // (n_crops_x - 1) if n_crops_x > 1 else crop_size
        step_y = (img_height - crop_size) // (n_crops_y - 1) if n_crops_y > 1 else crop_size

        crop_idx = 0
        for i in range(n_crops_y):
            for j in range(n_crops_x):
                left = min(j * step_x, img_width - crop_size)
                top = min(i * step_y, img_height - crop_size)
                crop = img.crop((left, top, left + crop_size, top + crop_size))

                crop_name = f"{os.path.splitext(img_file)[0]}_crop{crop_idx+1}.jpg"
                crop.save(os.path.join(output_folder, crop_name))
                crop_idx += 1

    print(f"Terminé : {len(image_files)} images traitées dans {image_folder}")


# ---------------------------------------------------------
# AUTOMATISATION DES 5 FOLDS, 2 SPLITS, ET 2 CLASSES
# ---------------------------------------------------------

base = "/content/drive/MyDrive/fairness/crop_5fold"

#folds = ["crop_1fold", "crop_2fold", "crop_3fold", "crop_4fold", "crop_5fold"]
folds = ["crop_4fold", "crop_5fold"]
#splits = ["train", "test"]
splits = ["train"]
classes = ["NORMAL"]

for fold in folds:
    for split in splits:
        for cls in classes:

            image_folder = f"{base}/{fold}/{split}/{cls}"
            output_folder = f"{base}/{fold}/{split}_224/{cls}"

            print("\n=== TRAITEMENT ===")
            print("Images : ", image_folder)
            print("Sauvegarde :", output_folder)

            crop_images(image_folder, output_folder, crop_size=224)



drive.flush_and_unmount()
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# visualiser le crop central

from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# -----------------------------
# FONCTION : VISUALISER LE CROP CENTRAL
# -----------------------------
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
        import matplotlib.patches as patches
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
# Exemple d'utilisation
# -----------------------------
image_folder = "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train/BENIGN"
show_center_crop(image_folder, crop_size=224)

# 224 mesure

import os
import numpy as np
from PIL import Image
from scipy.stats import f_oneway  # test ANOVA

U = []   # pour stocker les stats
all_widths = {}   # largeurs par classe
all_heights = {}  # hauteurs par classe

from tqdm import tqdm

# -----------------------------
# FONCTION : CALCUL STATISTIQUES TAILLE IMAGES
# -----------------------------
def taille_moyenne_pixels(dossier):
    largeurs = []
    hauteurs = []

    fichiers = os.listdir(dossier)

    # Boucle avec barre de progression
    for f in tqdm(fichiers, desc=f"Traitement {os.path.basename(dossier)}"):
        chemin = os.path.join(dossier, f)
        img = Image.open(chemin)
        largeur, hauteur = img.size
        largeurs.append(largeur)
        hauteurs.append(hauteur)

    m_largeur = np.mean(largeurs)
    m_hauteur = np.mean(hauteurs)
    V_largeur = np.var(largeurs)
    V_hauteur = np.var(hauteurs)
    m_pixels = m_largeur * m_hauteur

    return m_largeur, m_hauteur, m_pixels, V_largeur, V_hauteur, largeurs, hauteurs


# -----------------------------
# EXEMPLE D'UTILISATION
# -----------------------------
dossiers = {
    "normal": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train/NORMAL",
    "benign": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train/BENIGN",
    "malignant": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train/MALIGNANT",
    "normal_224": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train_224/NORMAL",
    "benign_224": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train_224/BENIGN",
    "malignant_224": "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train_224/MALIGNANT",
}

# --- Calcul stats locales + stockage largeurs/hauteurs ---
for type_dossier, chemin in dossiers.items():
    larg, haut, pix, vlar, vhaut, Lw, Lh = taille_moyenne_pixels(chemin)
    U.append((larg, haut, pix, vlar, vhaut))

    all_widths[type_dossier] = Lw
    all_heights[type_dossier] = Lh

    print(f"{type_dossier} : {larg:.1f}px × {haut:.1f}px (≈ {pix:.0f} pixels)")
    print(f"    Variances : largeur={vlar:.1f}, hauteur={vhaut:.1f}")
    print("")

import pandas as pd

# Largeurs avec colonne version
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

import seaborn as sns
import matplotlib.pyplot as plt

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
