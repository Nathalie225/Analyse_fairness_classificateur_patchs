# ============================================================
#                         IMPORTS
# ============================================================

import os
import re
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Function
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# ============================================================
#                     GOOGLE DRIVE
# ============================================================
from google.colab import drive
drive.mount('/content/drive')


# ============================================================
#                    DATASET FAIRNESS
# ============================================================
class FairnessImageFolder(Dataset):
    def __init__(self, root, csv_df, transform=None):
        """
        root      : dossier ImageFolder (train_224 ou test_224)
        csv_df    : DataFrame contenant "cropped image file path", "breast_density", "forme"
        La variable sensible sera 1 si breast_density==1 et forme=='calc', sinon 0
        """
        self.base = datasets.ImageFolder(root=root, transform=transform)
        self.classes = self.base.classes
        df = csv_df

        # Créer un mapping image de base (nom CSV) -> sensitive
        self.csv_map = {}
        for _, row in df.iterrows():
            fname_csv = os.path.basename(row["cropped image file path"])  # ex: 1.3.6.1.4.1...jpg
            s = int(row["breast_density"] == 1 and row["forme"].lower() == "calc")
            self.csv_map[fname_csv] = s

        # Maintenant associer chaque fichier du dossier à la variable sensible
        self.sensitive_map = {}
        for path, _ in self.base.samples:
            fname = os.path.basename(path)  # ex: 1.3.6.1.4.1..._P2_N_center224.jpg

            # Cherche dans le CSV un nom qui est préfixe de fname
            matched = None
            for csv_name in self.csv_map.keys():
                if fname.startswith(csv_name.replace(".jpg", "")):
                    matched = csv_name
                    break

            if matched:
                self.sensitive_map[fname] = self.csv_map[matched]
            else:
                self.sensitive_map[fname] = None

        # Vérification
        missing = [fname for fname, s in self.sensitive_map.items() if s is None]
        if len(missing) > 0:
            raise ValueError(f"{len(missing)} images sans variable sensible (ex: {missing[:5]})")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        path, _ = self.base.samples[idx]
        fname = os.path.basename(path)
        s = self.sensitive_map[fname]
        return x, y, torch.tensor(s, dtype=torch.long)


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
#                       CHARGEMENT DES CSV
# ============================================================

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

# ============================================================
#                     DATASETS & DATALOADERS
# ============================================================

# Lis les CSV
df_normal = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/train/normal_train_fold5.csv")
df_benign = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/train/benign_train_fold5.csv")
df_malignant = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/train/malignant_train_fold5.csv")

# Concaténation
df_all = pd.concat([df_normal, df_benign, df_malignant], ignore_index=True)

train_dataset = FairnessImageFolder(
    root="/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/train_224",
    csv_df=df_all,
    transform=train_transform
)


# Lis les CSV
df_normal = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/test/normal_test_fold5.csv")
df_benign = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/test/benign_test_fold5.csv")
df_malignant = pd.read_csv("/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/test/malignant_test_fold5.csv")

# Concaténation
df_all = pd.concat([df_normal, df_benign, df_malignant], ignore_index=True)


val_dataset = FairnessImageFolder(
    root="/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/test_224",
    csv_df=df_all,
    transform=val_transform
)



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)



# ============================================================
#                         MODELE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(train_dataset.classes)
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


# ============================================================
#                     POIDS DES CLASSES
# ============================================================
# Récupération des labels directement depuis ImageFolder
targets = train_dataset.base.targets  # plus rapide que d'itérer sur tout le dataset
class_counts = Counter(targets)

# Accès aux classes
classes = train_dataset.base.classes
num_classes = len(classes)
num_samples = len(targets)

# Calcul des poids de classes
class_weights = torch.tensor([num_samples / (num_classes * class_counts[c]) for c in range(num_classes)],
                             dtype=torch.float)

# Déplacer sur le bon device
class_weights = class_weights.to(device)

print("Images par classe :", class_counts)
print("Poids des classes :", class_weights)


for idx, name in enumerate(train_dataset.classes):
    print(f"Classe {idx} ({name}) → poids {class_weights[idx]:.4f}")





# ============================================================
#                         LOSS & OPTIM
# ============================================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

import torch
from torch.autograd import Function
lambda_w2 = 0.1

class W2Regularization(Function):
    """
    W2-like regularization for fairness (Equality of Opportunity)

    Aligns the mean of logits between sensitive groups s=0 and s=1
    conditioned on y=1.

    Arguments:
        logits : Tensor (N, D)
        s      : Tensor (N,) with values in {0,1}
        y      : Tensor (N,) with values in {0,1}
        lambda_w2 : float
    """

    @staticmethod
    def forward(ctx, logits, s, y, lambda_w2):
        # Safety checks (optional but recommended)
        assert logits.dim() == 2, "logits must be of shape (N, D)"
        assert s.dim() == 1 and y.dim() == 1, "s and y must be 1D tensors"

        ctx.save_for_backward(logits, s, y)
        ctx.lambda_w2 = lambda_w2

        # Regularization does not contribute to the loss value
        return logits.new_tensor(0.0)

    @staticmethod
    def backward(ctx, grad_output):
        logits, s, y = ctx.saved_tensors
        λ = ctx.lambda_w2

        grad_logits = torch.zeros_like(logits)

        # Equality of Opportunity: condition on y = 1
        mask0 = (s == 0) & (y == 1)
        mask1 = (s == 1) & (y == 1)

        n0 = mask0.sum()
        n1 = mask1.sum()

        # Apply only if both groups are present
        if n0 > 0 and n1 > 0:
            μ0 = logits[mask0].mean(dim=0, keepdim=True)
            μ1 = logits[mask1].mean(dim=0, keepdim=True)

            grad_logits[mask0] = λ * (μ0 - μ1) / n0
            grad_logits[mask1] = λ * (μ1 - μ0) / n1

        # grad_output is usually 1.0 but we propagate it correctly
        grad_logits = grad_logits * grad_output

        # No gradients for s, y, lambda
        return grad_logits, None, None, None

  # ============================================================
#                 TRAINING & EVALUATION
# ============================================================
def train_one_epoch():
    model.train()
    lambda_w2 = 0.1
    running_loss, correct, total = 0, 0, 0
    for images, labels, sensible in tqdm(train_loader, desc="Training"):
        images, labels, sensible = images.to(device), labels.to(device), sensible.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss_ce = criterion(outputs, labels)
        loss_w2 = W2Regularization.apply(outputs, sensible, labels, lambda_w2)
        loss = loss_ce + loss_w2

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
        for images, labels, _ in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
lambda_w2 = 0.1
epochs = 20
early_stop_patience = 5
no_improve = 0
best_val_loss = float("inf")

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_1fold/resultats"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_2fold/resultats"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_3fold/resultats"

#output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_4fold/resultats"

output_dir = "/content/drive/MyDrive/fairness/crop_5fold/crop_5fold/resultats"

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
