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
all_groups = []

with torch.no_grad():  # No gradient updates needed during testing
    for batch in test_loader:
        # ✅ Handle both FIN and non-FIN cases
        if args.use_fin:
            images, labels, identity_group = batch  # ✅ Extract identity_group
            identity_group = identity_group.to(device)
        else:
            images, labels = batch[:2]  # ✅ Extract only the first two values for standard model
            identity_group = None

        images, labels = images.to(device), labels.to(device)

        # ✅ Pass identity_group if using FIN
        outputs = model(images, identity_group) if args.use_fin else model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Convert outputs to probabilities using sigmoid activation
        predictions = torch.sigmoid(outputs).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

        if args.use_fin:
            all_groups.extend(identity_group.cpu().numpy())  # ✅ Store group info from batch
        else:
            # ✅ Infer identity_group directly from test_dataset
            all_groups = np.array([test_dataset[i][2].item() for i in range(len(test_dataset))])


# Convert to NumPy arrays
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups)  # Ensure groups are always stored


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
binary_predictions = (all_predictions >= threshold).astype(int)
overall_accuracy = accuracy_score(all_labels.flatten(), binary_predictions.flatten())

male_mask = np.array(all_groups) == 0
female_mask = np.array(all_groups) == 1

# ✅ Ensure all_masks match `all_labels` in shape
male_mask = male_mask[:len(all_labels)]  
female_mask = female_mask[:len(all_labels)]  


accuracy_male = accuracy_score(all_labels[male_mask].flatten(), binary_predictions[male_mask].flatten()) if male_mask.any() else float("nan")
accuracy_female = accuracy_score(all_labels[female_mask].flatten(), binary_predictions[female_mask].flatten()) if female_mask.any() else float("nan")

# ✅ Print Results for FIN & Non-FIN
model_type = "FIN Model" if args.use_fin else "Standard Model"
print(f"\n✅ {model_type} Evaluation Results:")
print(f"   🔹 Accuracy (Male):   {accuracy_male:.4f}")
print(f"   🔹 Accuracy (Female): {accuracy_female:.4f}")
print(f"   🔹 Overall Accuracy:  {overall_accuracy:.4f}")

# Save predictions for further analysis
np.save("test_predictions.npy", all_predictions)
np.save("test_labels.npy", all_labels)

print("✅ Testing complete! Predictions saved as 'test_predictions.npy' and 'test_labels.npy'.")
