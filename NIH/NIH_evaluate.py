import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel
from NIH_fin import FairChestXrayModel  # Import FIN model

def inspect_group_data(labels, predictions, group_mask, group_name):
    # Check for NaN values in predictions
    nan_in_predictions = np.isnan(predictions[group_mask]).any()
    print(f"NaN in {group_name} predictions: {nan_in_predictions}")

    # Count positive samples in true labels
    positive_samples = np.sum(labels[group_mask] == 1)
    print(f"Number of positive samples in {group_name} group: {positive_samples}")

    return nan_in_predictions, positive_samples

# Argument parsing for FIN
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

# Choose Model Based on Argument
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
all_groups = []

with torch.no_grad():  # No gradient updates needed during testing
    for batch in test_loader:
        images, labels, identity_group = batch  # Extract identity_group
        images, labels, identity_group = images.to(device), labels.to(device), identity_group.to(device)

        # Pass identity_group if using FIN
        outputs = model(images, identity_group) if args.use_fin else model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Convert outputs to probabilities using sigmoid activation
        predictions = torch.sigmoid(outputs).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        all_groups.extend(identity_group.cpu().numpy())  # Store group info from batch

# Convert to NumPy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups).flatten()  # Ensure groups are always stored as a flat array

# Compute AUC-ROC Score per class
auc_per_class = []
for i in range(num_classes):
    try:
        auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
        auc_per_class.append(auc)
    except ValueError:
        auc_per_class.append(float("0"))  # Handle cases where a class is missing

# Compute Mean AUC
mean_auc = np.nanmean(auc_per_class)  # Ignore NaNs if any class is missing

# Compute Accuracy with Threshold
threshold = 0.5
binary_predictions = (all_predictions >= threshold).astype(int)
overall_accuracy = accuracy_score(all_labels.flatten(), binary_predictions.flatten())

# Compute AUC-ROC and Accuracy for Male and Female groups
male_mask = all_groups == 0
female_mask = all_groups == 1

# Inspect male group data
nan_in_male_predictions, positive_male_samples = inspect_group_data(
    all_labels, all_predictions, male_mask, "male"
)

# Inspect female group data
nan_in_female_predictions, positive_female_samples = inspect_group_data(
    all_labels, all_predictions, female_mask, "female"
)
# Initialize metrics
auc_male = auc_female = accuracy_male = accuracy_female = float("nan")

# Compute metrics for male group
if positive_male_samples > 0 and not nan_in_male_predictions:
    try:
        auc_male = roc_auc_score(all_labels[male_mask], all_predictions[male_mask])
    except ValueError as e:
        print(f"Error computing AUC-ROC for male group: {e}")
    accuracy_male = accuracy_score(
        all_labels[male_mask].flatten(),
        (all_predictions[male_mask] >= 0.5).astype(int).flatten()
    )
else:
    print("Insufficient data to compute AUC-ROC for male group.")

# Compute metrics for female group
if positive_female_samples > 0 and not nan_in_female_predictions:
    try:
        auc_female = roc_auc_score(all_labels[female_mask], all_predictions[female_mask])
    except ValueError as e:
        print(f"Error computing AUC-ROC for female group: {e}")
    accuracy_female = accuracy_score(
        all_labels[female_mask].flatten(),
        (all_predictions[female_mask] >= 0.5).astype(int).flatten()
    )
else:
    print("Insufficient data to compute AUC-ROC for female group.")

# Print the computed metrics
print(f"AUC-ROC (Male): {auc_male}")
print(f"Accuracy (Male): {accuracy_male}")
print(f"AUC-ROC (Female): {auc_female}")
print(f"Accuracy (Female): {accuracy_female}")

print("✅ Testing complete! Predictions saved as 'test_predictions.npy' and 'test_labels.npy'.")
