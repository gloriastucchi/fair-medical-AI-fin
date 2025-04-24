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



parser = argparse.ArgumentParser()
parser.add_argument("--loss_type", default="bce", choices=["bce", "focal"],
                    help="Loss function to use: 'bce' or 'focal'")
parser.add_argument("--use_fin", action="store_true", help="Use FIN model instead of standard ResNet model")  # ✅ Correct
args = parser.parse_args()

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

# Create training dataset (LIMITED to 20.000 samples)
train_dataset = ChestXrayDataset(
    CSV_FILE, IMAGE_FOLDER, TRAIN_LIST, transform,
    subset_size=None, stratify=True
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
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
feature_dim = 512  # ResNet18 feature dimension
num_groups = 2  # Example: Male/Female

if args.use_fin:
    print("✅ Using FIN model for training.")
    base_model = ChestXrayModel(num_classes=feature_dim)
    model = FairChestXrayModel(base_model, feature_dim, num_groups).to(device)
else:
    print("✅ Using standard ResNet model for training.")
    model = ChestXrayModel(num_classes=15).to(device)

# Define loss and optimizer
if args.loss_type == "focal":
    print("✅ Using Focal Loss.")
    criterion = FocalLoss(alpha=1.0, gamma=0.5)
else:
    print("✅ Using BCEWithLogitsLoss.")
    criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels, identity_group in progress_bar:  # ✅ Now expecting 3 values
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, identity_group) if args.use_fin else model(images)  # ✅ Pass identity_group when using FIN
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Salva il modello con nome che riflette FIN, tipo di loss e loss finale
fin_tag = "fin" if args.use_fin else "nofin"
loss_tag = args.loss_type
final_loss_tag = f"{total_loss/len(train_loader):.4f}"
model_filename = f"NIH_model_{fin_tag}_{loss_tag}_loss{final_loss_tag}.pth"

torch.save(model.state_dict(), model_filename)
print(f"✅ Model saved as '{model_filename}' 🎉")
