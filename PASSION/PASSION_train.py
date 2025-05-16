import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PASSION_model import PassionClassifier, ImpetigoBinaryClassifier
from PASSION_dataset import PassionDataset
import argparse
from tqdm import tqdm
import os

# --- Argument parser ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--task', choices=['condition', 'impetigo'], default='condition')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--train_subjects', required=True)
parser.add_argument('--val_subjects', required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load datasets ---
train_dataset = PassionDataset(args.csv_path, args.img_dir, task=args.task, subject_list=args.train_subjects)
val_dataset = PassionDataset(args.csv_path, args.img_dir, task=args.task, subject_list=args.val_subjects)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# --- Model and loss ---
if args.task == 'condition':
    model = PassionClassifier(num_classes=4)
    criterion = nn.CrossEntropyLoss()
else:
    model = ImpetigoBinaryClassifier()
    criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# --- Training loop with early stopping ---
best_val_loss = float('inf')
epochs_no_improve = 0

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
    model.eval()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float() if args.task == 'impetigo' else labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} | Validation loss: {val_loss:.4f}")

    # --- Check for improvement ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), f"best_model_{args.task}.pth")
        print(f"✅ Saved new best model at epoch {epoch+1}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= args.patience:
            print("⏹️ Early stopping triggered")
            break
