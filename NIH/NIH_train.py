import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel

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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestXrayModel(num_classes=15).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "chestxray_model.pth")
print("✅ Model saved as 'chestxray_model.pth' 🎉")
