import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from NIH_fin import FairChestXrayModel
from NIH_model import ChestXrayModel
from NIH_dataset import ChestXrayDataset, transform
from tqdm import tqdm
import time
import numpy as np
import random


from sklearn.metrics import roc_auc_score

@torch.no_grad()
def evaluate_auc(model, val_loader, use_fin, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels, groups in val_loader:
        images, labels, groups = images.to(device), labels.to(device), groups.to(device)
        outputs = model(images, groups) if use_fin else model(images)
        probs = torch.sigmoid(outputs)
        all_preds.append(probs.cpu())
        all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_preds, average='macro')
    except ValueError:
        auc = float('nan')

    return auc



class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probas = torch.sigmoid(inputs)

        # Clamp values for stability
        probas = torch.clamp(probas, min=1e-6, max=1. - 1e-6)

        focal_term = (1 - probas) ** self.gamma
        loss = self.alpha * focal_term * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", default="bce", choices=["bce", "focal"],
                    help="Loss function to use: 'bce' or 'focal'")
parser.add_argument("--use_fin", action="store_true", help="Use FIN model instead of standard ResNet model")  # ✅ Correct
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()



set_seed(args.seed)
# Define paths
#CSV_FILE = "/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv"
#no hpc
#IMAGE_FOLDER = "/Users/gloriastucchi/Desktop/NIH/images/"
#hpc
#IMAGE_FOLDER = "/work3/s232437/images_full/"
#TRAIN_LIST = "/Users/gloriastucchi/Desktop/NIH/train_val_list.txt"
#TEST_LIST = "/Users/gloriastucchi/Desktop/NIH/test_list.txt"  # Corrected test file
CSV_FILE = "/zhome/4b/b/202548/NIH/Data_Entry_2017_v2020_.csv"
IMAGE_FOLDER = "/work3/s232437/images_full/images/"
TRAIN_LIST = "/zhome/4b/b/202548/NIH/train_val_list.txt"
TEST_LIST = "/zhome/4b/b/202548/NIH/test_list.txt"

TRAIN_LIST = "/work3/s232437/fair-medical-AI-fin/NIH/train_list.txt"
VAL_LIST = "/work3/s232437/fair-medical-AI-fin/NIH/val_list.txt"

train_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TRAIN_LIST, transform)
val_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, VAL_LIST, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"✅ Training on {len(train_loader.dataset)} samples.")  # inside ChestXrayDataset
 # ! does not consider subset dimension
full_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TRAIN_LIST, transform)
subset_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TRAIN_LIST, transform, subset_size=20000, stratify=True)
full_prevalence = full_dataset.labels.mean(axis=0)
subset_prevalence = subset_dataset.labels.mean(axis=0)

for i, (fp, sp) in enumerate(zip(full_prevalence, subset_prevalence)):
    print(f"Class {i}: Full = {fp:.4f}, Subset = {sp:.4f}, Δ = {abs(fp - sp):.4f}")

# Create test dataset (FULL dataset, NO sampling)
test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ✅ Choose Model Based on Argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_dim = 1024  # DENSENET18 feature dimension
num_groups = 2  # Example: Male/Female

if args.use_fin:
    print("✅ Using FIN model for training.")
    base_model = ChestXrayModel(num_classes=feature_dim)
    model = FairChestXrayModel(base_model, feature_dim, num_groups).to(device)
else:
    print("✅ Using standard ResNet model for training.")
    model = ChestXrayModel(num_classes=14).to(device)

# Define loss and optimizer
if args.loss_type == "focal":
    print("✅ Using Focal Loss.")
    criterion = FocalLoss(alpha=1.0, gamma=0.5)
else:
    print("✅ Using BCEWithLogitsLoss.")
    criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 15
best_val_auc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels, groups in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels, groups = images.to(device), labels.to(device), groups.to(device)
        optimizer.zero_grad()
        outputs = model(images, groups) if args.use_fin else model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")

    # Evaluate on validation set
    val_auc = evaluate_auc(model, val_loader, use_fin=args.use_fin, device=device)
    print(f"Validation AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), f"FINALFIN_best_model_epoch{epoch+1}_auc{val_auc:.4f}.pth")
        print("Saved new best model.")


# Salva il modello con nome che riflette FIN, tipo di loss e loss finale
fin_tag = "fin" if args.use_fin else "nofin"
loss_tag = args.loss_type
final_loss_tag = f"{total_loss/len(train_loader):.4f}"
model_filename = f"FINALFIN_NIH_model_{fin_tag}_{loss_tag}_loss{final_loss_tag}_seed{args.seed}.pth"

torch.save(model.state_dict(), model_filename)
print(f"✅ Model saved as '{model_filename}' 🎉")
