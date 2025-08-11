import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_passion_baseline import PassionClassifierNoFIN, ImpetigoBinaryClassifierNoFIN
from PASSION_dataset import PassionDataset
import argparse
from tqdm import tqdm
import os
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import roc_auc_score


# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--task', choices=['condition', 'impetigo'], default='condition')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--train_subjects', required=True)
parser.add_argument('--val_subjects', required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data augmentation ---
def get_default_transforms():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Load datasets ---
transform = get_default_transforms()
train_dataset = PassionDataset(args.csv_path, args.img_dir, task=args.task, subject_list=args.train_subjects, transform=transform)
val_dataset = PassionDataset(args.csv_path, args.img_dir, task=args.task, subject_list=args.val_subjects, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# --- Model and loss ---
if args.task == 'condition':
    model = PassionClassifierNoFIN(num_classes=4)
    criterion = nn.CrossEntropyLoss()
else:
    model = ImpetigoBinaryClassifierNoFIN()
    criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# --- Training loop with early stopping ---
best_val_loss = float('inf')
epochs_no_improve = 0
best_val_auc = 0.0  # or float('-inf')


for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float() if args.task == 'impetigo' else labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # --- Validation ---
    val_loss = 0
    all_labels = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            if args.task == 'condition':
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                probs = torch.sigmoid(outputs).cpu().numpy()
                probs = np.squeeze(probs)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            loss = criterion(outputs, labels.float() if args.task == 'impetigo' else labels.to(device))
            val_loss += loss.item()

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    val_loss /= len(val_loader)

    if args.task == 'condition':
        val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    else:
        val_auc = roc_auc_score(all_labels, all_probs)

    print(f"Epoch {epoch+1} | Validation loss: {val_loss:.4f} | Validation AUC: {val_auc:.4f}")

    # --- Check for improvement ---
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"best_model_{args.task}_thirdseed.pth")
        print(f"✅ Saved new best model at epoch {epoch+1} (AUC: {val_auc:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print("⏹️ Early stopping triggered")
            break
    print(f"Best validation AUC so far: {best_val_auc:.4f}")