"""
Entrainement avec une méthode de In-processing de type adversarial debiaising
====================================================
Ce script décrit l'entrainement du classificateur de patchs.
Il propose l'utilisation d'une méthode de In-processing .
"""


# ============================================================
# Importations des  bibliothèques
# ============================================================

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import Counter

# ============================================================
# Ouverture de drive 
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
#                        Données 
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
#                     DATASETS & DATALOADERS
# ============================================================

# Lis les CSV ( à modifier selon le fold choisit )
df_normal = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/train/normal_train_fold1.csv")
df_benign = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/train/benign_train_fold1.csv")
df_malignant = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/train/malignant_train_fold1.csv")

# Concaténation
df_all = pd.concat([df_normal, df_benign, df_malignant], ignore_index=True)

train_dataset = FairnessImageFolder(
    root="/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Patch_224x224pixels/train_224",
    csv_df=df_all,
    transform=train_transform
)


# Lis les CSV
df_normal = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/test/normal_test_fold1.csv")
df_benign = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/test/benign_test_fold1.csv")
df_malignant = pd.read_csv("/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Métadonnées/malignant_test_fold1.csv")

# Concaténation
df_all = pd.concat([df_normal, df_benign, df_malignant], ignore_index=True)


val_dataset = FairnessImageFolder(
    root="/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Entrainement/1fold/Patch_224x224pixels/test_224",
    csv_df=df_all,
    transform=val_transform
)


# ============================================================
#                  CHARGEMENT DES DATASETS
# ============================================================

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)


# ============================================================
#                   INITIALISATION DU MODÈLE
# ============================================================

# Gradient Reversal Layer
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# Adversary
class Adversary(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # 2 classes sensibles
        )

    def forward(self, x):
        x = self.grl(x)
        return self.net(x)

# DenseNet pré-entrainé
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

adversary = Adversary(num_features).to(device)


# ============================================================
#        Fonction de perte, optimizer et scheduler
# ============================================================

criterion_adv = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(adversary.parameters()), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ============================================================
#         FORWARD AVEC FEATURES
# ============================================================
def forward_with_features(model, x):
    features = model.features(x)
    features = nn.functional.relu(features, inplace=True)
    features = nn.functional.adaptive_avg_pool2d(features, (1,1)).view(features.size(0), -1)
    out = model.classifier(features)
    return out, features
    
# ============================================================
#                      FONCTIONS D'ENTRAÎNEMENT
# ============================================================

# 1. Entrainement avec les données train

def train_one_epoch():
    model.train()
    adversary.train()
    running_loss, running_adv_loss = 0, 0
    correct, total = 0, 0

    for images, labels, sens in tqdm(train_loader, desc="Training"):
        images, labels, sens = images.to(device), labels.to(device), sens.to(device)
        optimizer.zero_grad()

        outputs, features = forward_with_features(model, images)
        loss_cls = criterion(outputs, labels)

        outputs_sens = adversary(features)
        loss_adv = criterion_adv(outputs_sens, sens)

        loss = loss_cls + loss_adv
        loss.backward()
        optimizer.step()

        running_loss += loss_cls.item() * images.size(0)
        running_adv_loss += loss_adv.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_adv_loss / total, correct / total

# 2. Evaluation avec les données test

def evaluate():
    model.eval()
    adversary.eval()
    running_loss, running_adv_loss = 0, 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, sens in tqdm(val_loader, desc="Validation"):
            images, labels, sens = images.to(device), labels.to(device), sens.to(device)
            outputs, features = forward_with_features(model, images)
            loss_cls = criterion(outputs, labels)

            outputs_sens = adversary(features)
            loss_adv = criterion_adv(outputs_sens, sens)

            running_loss += loss_cls.item() * images.size(0)
            running_adv_loss += loss_adv.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, running_adv_loss / total, correct / total
    
# ============================================================
#                 Structure de l'entrainement
# ============================================================

# sauvegarde la fonction et perte et l'évolution de la précision selon les époques
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
lambda_adv=1.0

# 1. Première phase d'entrainement sur 20 époques
epochs = 20
early_stop_patience = 5
no_improve = 0
best_val_loss = float("inf")

# lieu de sauvegarde de meilleur modèle et des courbes d'évolution (exemple fold1)

output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold1/In-processing"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold2/In-processing"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold3/In-processing"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold4/In-processing"

#output_dir = "/content/drive/MyDrive/Stage_ARIA_3mois_Données_Réorganisées/Patchs_Mammographie_5fold/Résultats_Entrainement/fold5/In-processing"



for epoch in range(epochs):
    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    # Phase 1 : entraînement classifier
    train_loss, train_adv_loss, train_acc = train_one_epoch()
    val_loss, val_adv_loss, val_acc = evaluate()

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

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
    train_loss, train_adv_loss, train_acc = train_one_epoch()
    val_loss, val_adv_loss, val_acc = evaluate()

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


