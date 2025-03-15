import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from NIH_fin import FairChestXrayModel
from NIH_model import ChestXrayModel
from NIH_dataset import ChestXrayDataset, transform

parser = argparse.ArgumentParser()
parser.add_argument("--use_fin", action="store_true", help="Use FIN model instead of standard ResNet model")  # ✅ Correct
args = parser.parse_args()

# Define paths
CSV_FILE = "/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv"
IMAGE_FOLDER = "/Users/gloriastucchi/Desktop/NIH/images/"
TRAIN_LIST = "/Users/gloriastucchi/Desktop/NIH/train_val_list.txt"
TEST_LIST = "/Users/gloriastucchi/Desktop/NIH/test_list.txt"  # Corrected test file

# Create training dataset (LIMITED to 500 samples)
train_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TRAIN_LIST, transform, subset_size=500)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create test dataset (FULL dataset, NO sampling)
test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform, subset_size=None)
print(f"✅ Training on {len(train_dataset)} samples.")
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
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels, identity_group in train_loader:  # ✅ Now expecting 3 values
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, identity_group) if args.use_fin else model(images)  # ✅ Pass identity_group when using FIN
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
model_filename = "chestxray_model_fin.pth" if args.use_fin else "chestxray_model.pth"
torch.save(model.state_dict(), model_filename)
print(f"✅ Model saved as '{model_filename}' 🎉")