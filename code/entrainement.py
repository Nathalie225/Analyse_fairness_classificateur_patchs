# ============================================================
#         IMPORTS
# ============================================================
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



# Entrainement fold par fold

# 1. Télécharge les images


#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/test_224',  transform=val_transform)

#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_2fold/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_2fold/test_224',  transform=val_transform)

#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_3fold/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_3fold/test_224',  transform=val_transform)
#train_dataset = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_4fold/train_224', transform=train_transform)
#val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_4fold/test_224',  transform=val_transform)

train_dataset = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/train_224', transform=train_transform)
val_dataset   = datasets.ImageFolder('/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/test_224',  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialise le modèle pré-entrainé

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
# 1) On FREEZE totalement les features au début
for param in model.features.parameters():
    param.requires_grad = False

# Nouveau classifier
num_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, num_classes)
)

model = model.to(device)

# 3. Prend en compte les différences de quantité selon images

from collections import Counter
import torch
import torch.nn as nn
targets = train_dataset.targets
class_counts = Counter(targets)

print("Images par classe :", class_counts)
num_classes = len(train_dataset.classes)
num_samples = len(targets)

class_weights = torch.zeros(num_classes)

for c in range(num_classes):
    class_weights[c] = num_samples / (num_classes * class_counts[c])

class_weights = class_weights.to(device)

print("Poids des classes :", class_weights)


for idx, name in enumerate(train_dataset.classes):
    print(f"Classe {idx} ({name}) → poids {class_weights[idx]:.4f}")



model = model.to(device)
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def train_one_epoch():
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


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

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
epochs = 20
early_stop_patience = 5
no_improve = 0
best_val_loss = float("inf")

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/resultats/résultats_avec_poids"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_2fold/resultats/résultats_avec_poids"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_3fold/resultats/résultats_avec_poids"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_4fold/resultats/résultats_avec_poids"

output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/resultats/résultats_avec_poids"

os.makedirs(output_dir, exist_ok=True)

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

    print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f}")
    print(f"Val:   loss={val_loss:.4f} acc={val_acc:.4f}")

    # Sauvegarde best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print("🔥 Nouveau meilleur modèle sauvegardé !")
    else:
        no_improve += 1

    # EARLY STOPPING
    if no_improve >= early_stop_patience:
        print("\n⛔ Early stopping : le modèle ne s'améliore plus.")
        break

# =======================
#   DEFREEZE PARTIEL APRÈS PHASE 1
# =======================

print("\n🔓 Déblocage des dernières couches (fine-tuning progressif)…")

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
        print("🔥 Nouveau meilleur modèle sauvegardé !")
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

print("\n🎉 Script terminé ! Le meilleur modèle est enregistré dans :")
print(os.path.join(output_dir, "best_model.pth"))
drive.flush_and_unmount()
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
