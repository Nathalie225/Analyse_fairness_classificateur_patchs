"""
Entrainement avec ou sans pondération des classes
====================================================
Ce script décrit l'entrainement du classificateur de patchs.
Il propose l'utilisation ou non de la pondération des classes.
"""

# ============================================================
# Importations des  bibliothèques
# ============================================================

import os
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# Ouverture de drive 
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
#         TRANSFORMATIONS
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ============================================================
#                        Données 
#
#     Chargement du fichier CSV "patients_paths.csv" contenant,
#      pour chaque patient, les chemins des triplets d’images
#
# ============================================================

# Entrainement fold par fold ( ici on prend fold 1)
train_dataset = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Patch_224x224pixels/train_224', transform=train_transform)
val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Patch_224x224pixels/test_224',  transform=val_transform)

#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/2fold/Patch_224x224pixels/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/2fold/Patch_224x224pixels/test_224',  transform=val_transform)

#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/3fold/Patch_224x224pixels/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/3fold/Patch_224x224pixels/test_224',  transform=val_transform)
#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/4fold/Patch_224x224pixels/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/4fold/Patch_224x224pixels/test_224',  transform=val_transform)
#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/5fold/Patch_224x224pixels/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/5fold/Patch_224x224pixels/test_224',  transform=val_transform)

# ============================================================
#                  CHARGEMENT DES DATASETS
# ============================================================

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#                   INITIALISATION DU MODÈLE
# ============================================================

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

# On FREEZE totalement les features au début
for param in model.features.parameters():
    param.requires_grad = False

# Nouveau classifier
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)

model = model.to(device)

# ============================================================
#               Option: Pondération des classes
# ============================================================

targets = train_dataset.targets
class_counts = Counter(targets)

print("Images par classe :", class_counts)
num_classes = len(train_dataset.classes)
num_samples = len(targets)

class_weights = torch.zeros(num_classes)

# Assigne un poids à chaque classe qui correspond à l'inverse de sa fréquence 
for c in range(num_classes):
    class_weights[c] = num_samples / (num_classes * class_counts[c])

class_weights = class_weights.to(device)

print("Poids des classes :", class_weights)


for idx, name in enumerate(train_dataset.classes):
    print(f"Classe {idx} ({name}) → poids {class_weights[idx]:.4f}")

# ============================================================
#        Fonction de perte, optimizer et scheduler
# ============================================================

# Option sans pondération des classes
# criterion = nn.CrossEntropyLoss()

# Option avec pondération des classes: 
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ============================================================
#                      FONCTIONS D'ENTRAÎNEMENT
# ============================================================

# 1. Entrainement avec les données train

def train_one_epoch():
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc="Training"):
        # récupère les images et labels par batch
        images, labels = images.to(device), labels.to(device)
        # remise à zéro des gradients
        optimizer.zero_grad()
        # labels prédits par le modèle
        outputs = model(images)
        # calcul de la fonction de perte entre les labels prédits et les labels originaux
        loss = criterion(outputs, labels)
        # descente de gradient
        loss.backward()
        # change les poids du modèle selon les gradients
        optimizer.step()
        # sauvegarde la perte et la précision de cette époque
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# 2. Evaluation avec les données test
def evaluate():
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total
    
# ============================================================
#                 Structure de l'entrainement
# ============================================================

# sauvegarde la fonction et perte et l'évolution de la précision selon les époques
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# 1. Première phase d'entrainement sur 20 époques

epochs = 20
early_stop_patience = 5
no_improve = 0
best_val_loss = float("inf")

# lieu de sauvegarde de meilleur modèle et des courbes d'évolution (exemple fold1)

output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold1/Avec_pondération"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold2/Avec_pondération"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold3/Avec_pondération"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold4/Avec_pondération"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold5/Avec_pondération"



for epoch in range(epochs):
    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    # Phase 1 : entraînement classifier
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Affiche les résultats par époque
    print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f}")
    print(f"Val:   loss={val_loss:.4f} acc={val_acc:.4f}")

    # Sauvegarde le meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    else:
        no_improve += 1

    # Arrêt si pas d'amélioration
    if no_improve >= early_stop_patience:
        print("le modèle ne s'améliore plus.")
        break

# 2. Seconde phase d'entrainement sur 5 époques


# déblocage du denseblock 4
for param in model.features.denseblock4.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(5):  # courte phase de fine-tuning
    print(f"\n===== Fine-tuning Epoch {epoch+1}/5 =====")
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))



# ============================================================
#                  VISUALISATION DES RESULTATS
# ============================================================
plt.figure(figsize=(6,5))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss.png"))
plt.show()

plt.figure(figsize=(6,5))
plt.plot(history["train_acc"], label="train")
plt.plot(history["val_acc"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "acc.png"))
plt.show()


print(os.path.join(output_dir, "best_model.pth"))

