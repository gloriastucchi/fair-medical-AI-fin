import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel
from NIH_fin import FairChestXrayModel  # Import FIN model

# ✅ Add argument parsing for FIN
parser = argparse.ArgumentParser()
parser.add_argument("--use_fin", action="store_true", help="Use FIN model for evaluation")
args = parser.parse_args()

# Define paths
CSV_FILE = "/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv"
IMAGE_FOLDER = "/Users/gloriastucchi/Desktop/NIH/images/"
TEST_LIST = "/Users/gloriastucchi/Desktop/NIH/test_list.txt"

# Load test dataset
test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform, subset_size=200)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Choose Model Based on Argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = test_dataset.labels.shape[1]
feature_dim = 512  # ResNet feature dimension
num_groups = 2  # Gender groups (M/F)

if args.use_fin:
    print("✅ Using FIN model for testing.")
    base_model = ChestXrayModel(num_classes=feature_dim)  # Feature extractor
    model = FairChestXrayModel(base_model, feature_dim, num_groups).to(device)
    model.load_state_dict(torch.load("chestxray_model_fin.pth"))
else:
    print("✅ Using standard ResNet model for testing.")
    model = ChestXrayModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("chestxray_model.pth"))

model.eval()  # Set model to evaluation mode

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Run testing
total_loss = 0
all_predictions = []
all_labels = []

with torch.no_grad():  # No gradient updates needed during testing
    for batch in test_loader:
        # ✅ Handle both FIN and non-FIN cases
        if args.use_fin:
            images, labels, identity_group = batch  # ✅ Extract identity_group
            identity_group = identity_group.to(device)
        else:
            images, labels = batch[:2]  # ✅ Extract only the first two values for standard model

        images, labels = images.to(device), labels.to(device)

        # ✅ Pass identity_group if using FIN
        outputs = model(images, identity_group) if args.use_fin else model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Convert outputs to probabilities using sigmoid activation
        predictions = torch.sigmoid(outputs).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

# Convert to NumPy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# ✅ Compute AUC-ROC Score per class
auc_per_class = []
for i in range(num_classes):
    try:
        auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
        auc_per_class.append(auc)
    except ValueError:
        auc_per_class.append(float("nan"))  # Handle cases where a class is missing

# ✅ Compute Mean AUC
mean_auc = np.nanmean(auc_per_class)  # Ignore NaNs if any class is missing

# ✅ Compute Accuracy with Threshold
threshold = 0.5
binary_predictions = (all_predictions >= threshold).astype(int)  # Convert probabilities to 0/1
accuracy = accuracy_score(all_labels.flatten(), binary_predictions.flatten())

# ✅ Print Results
avg_loss = total_loss / len(test_loader)
print(f"✅ Test Loss: {avg_loss:.4f}")
print(f"✅ Mean AUC-ROC: {mean_auc:.4f}")
print(f"✅ Accuracy: {accuracy:.4f}")

# Save predictions for further analysis
np.save("test_predictions.npy", all_predictions)
np.save("test_labels.npy", all_labels)

print("✅ Testing complete! Predictions saved as 'test_predictions.npy' and 'test_labels.npy'.")
