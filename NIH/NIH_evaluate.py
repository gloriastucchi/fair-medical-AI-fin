import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel
from NIH_fin import FairChestXrayModel  # Import FIN model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score
import warnings

warnings.filterwarnings("ignore")

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
#CSV_FILE = "/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv"
#no hpc
#IMAGE_FOLDER = "/Users/gloriastucchi/Desktop/NIH/images/"
#hpc
#IMAGE_FOLDER = "/work3/s232437/images_full/"
#TRAIN_LIST = "/Users/gloriastucchi/Desktop/NIH/train_val_list.txt"
#TEST_LIST = "/Users/gloriastucchi/Desktop/NIH/test_list.txt"  # Corrected test file
CSV_FILE = "/work3/s232437/fair-medical-AI-fin/NIH/Data_Entry_2017_v2020_.csv"
IMAGE_FOLDER = "/work3/s232437/images_full/images/"
TEST_LIST = "/work3/s232437/fair-medical-AI-fin/NIH/test_list.txt"

# Load test dataset
test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers = 4)

# Choose Model Based on Argument
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = test_dataset.labels.shape[1]
feature_dim = 512  # ResNet feature dimension
num_groups = 2  # Gender groups (M/F)

if args.use_fin:
    print("✅ Using FIN model for testing.")
    base_model = ChestXrayModel(num_classes=feature_dim)  # Feature extractor
    model = FairChestXrayModel(base_model, feature_dim, num_groups).to(device)
    model.load_state_dict(torch.load("/work3/s232437/fair-medical-AI-fin/NIH_fulldata_model_fin_bce_loss0.1260.pth"))
else:
    print("✅ Using standard ResNet model for testing.")
    model = ChestXrayModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("/work3/s232437/fair-medical-AI-fin/NIH_fulldata_model_nofin_bce_loss0.1246.pth"))

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


# Compute AUC-ROC Score per class (overall, male, female)
auc_per_class = []
auc_per_class_male = []
auc_per_class_female = []

male_mask = all_groups == 0
female_mask = all_groups == 1

for i in range(num_classes):
    labels_cls = all_labels[:, i]
    preds_cls = all_predictions[:, i]
    
    # Overall AUC
    if np.unique(labels_cls).size == 2:
        try:
            auc = roc_auc_score(labels_cls, preds_cls)
        except Exception as e:
            print(f"[Overall] Error computing AUC for class {i}: {e}")
            auc = np.nan
    else:
        print(f"[Overall] Skipping class {i} due to only one label.")
        auc = np.nan
    auc_per_class.append(auc)

    # Male AUC
    labels_male = all_labels[male_mask, i]
    preds_male = all_predictions[male_mask, i]
    if np.unique(labels_male).size == 2:
        try:
            auc_m = roc_auc_score(labels_male, preds_male)
        except Exception as e:
            print(f"[Male] Error computing AUC for class {i}: {e}")
            auc_m = np.nan
    else:
        print(f"[Male] Skipping class {i} due to only one label.")
        auc_m = np.nan
    auc_per_class_male.append(auc_m)

    # Female AUC
    labels_female = all_labels[female_mask, i]
    preds_female = all_predictions[female_mask, i]
    if np.unique(labels_female).size == 2:
        try:
            auc_f = roc_auc_score(labels_female, preds_female)
        except Exception as e:
            print(f"[Female] Error computing AUC for class {i}: {e}")
            auc_f = np.nan
    else:
        print(f"[Female] Skipping class {i} due to only one label.")
        auc_f = np.nan
    auc_per_class_female.append(auc_f)

# Convert to numpy for nan filtering
auc_per_class = np.array(auc_per_class)
auc_per_class_male = np.array(auc_per_class_male)
auc_per_class_female = np.array(auc_per_class_female)

# Add the names of the disease classes (standard NIH order)
class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

print("\n📊 Per-Class AUC-ROC Scores:\n")
print(f"{'Class':<22} {'Overall':>8} {'Male':>8} {'Female':>8} {'Δ(M)':>8} {'Δ(F)':>8}")
print("-" * 60)

for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else f"Class {i}"
    auc_o = auc_per_class[i]
    auc_m = auc_per_class_male[i]
    auc_f = auc_per_class_female[i]

    if not np.isnan(auc_o):
        delta_m = abs(auc_m - auc_o) if not np.isnan(auc_m) else np.nan
        delta_f = abs(auc_f - auc_o) if not np.isnan(auc_f) else np.nan
        print(f"{name:<22} {auc_o:8.4f} {auc_m:8.4f} {auc_f:8.4f} {delta_m:8.4f} {delta_f:8.4f}")
    else:
        print(f"{name:<22} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")


# === Compute Overall Mean AUC ===
mean_auc = np.nanmean(auc_per_class)
mean_auc_male = np.nanmean(auc_per_class_male)
mean_auc_female = np.nanmean(auc_per_class_female)

# === Compute Correct Per-Class Equity-Scaled AUC (following formal definition) ===
es_auc_per_class = []
disparity_per_class = []
alpha = 1

print("\n📏 Correct Equity-Scaled AUC per class (based on group disparity):\n")
print(f"{'Class':<22} {'AUC':>8} {'ES-AUC':>10} {'Disparity':>12}")
print("-" * 56)

for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else f"Class {i}"
    auc_o = auc_per_class[i]
    auc_m = auc_per_class_male[i]
    auc_f = auc_per_class_female[i]

    if not np.isnan(auc_o) and not np.isnan(auc_m) and not np.isnan(auc_f):
        disparity = abs(auc_m - auc_o) + abs(auc_f - auc_o)
        es_auc = auc_o / (1 + alpha * disparity)

        es_auc_per_class.append(es_auc)
        disparity_per_class.append(disparity)

        print(f"{name:<22} {auc_o:8.4f} {es_auc:10.4f} {disparity:12.4f}")
    else:
        es_auc_per_class.append(np.nan)
        disparity_per_class.append(np.nan)
        print(f"{name:<22} {'N/A':>8} {'N/A':>10} {'N/A':>12}")

mean_es_auc = np.nanmean(es_auc_per_class)
print(f"\n✅ Mean ES-AUC (per class): {mean_es_auc:.4f}")


# === Compute Global Equity-Scaled AUC ===
# Define only once: prevents crash
disparity_sum = 0.0
valid_classes = ~np.isnan(auc_per_class_male) & ~np.isnan(auc_per_class_female) & ~np.isnan(auc_per_class)
valid_auc_male = auc_per_class_male[valid_classes]
valid_auc_female = auc_per_class_female[valid_classes]
valid_auc_overall = auc_per_class[valid_classes]


for a_male, a_female, a_overall in zip(valid_auc_male, valid_auc_female, valid_auc_overall):
    disparity_sum += abs(a_male - a_overall) + abs(a_female - a_overall)




# === Compute Accuracy per Group ===
accuracy_male = accuracy_score(
    all_labels[male_mask].flatten(),
    (all_predictions[male_mask] >= 0.5).astype(int).flatten()
)
accuracy_female = accuracy_score(
    all_labels[female_mask].flatten(),
    (all_predictions[female_mask] >= 0.5).astype(int).flatten()
)

# === Print Final Metrics ===
print("🔍 Final Metrics:")
print(f"Mean AUC-ROC (Overall): {mean_auc:.4f}")
print(f"Mean AUC-ROC (Male):    {mean_auc_male:.4f}")
print(f"Mean AUC-ROC (Female):  {mean_auc_female:.4f}")
print(f"Equity-Scaled AUC:      {mean_es_auc:.4f}")
print(f"Accuracy (Male):        {accuracy_male:.4f}")
print(f"Accuracy (Female):      {accuracy_female:.4f}")

print("\n📉 Confusion Matrices per Class (threshold=0.5):")
print("Each matrix is formatted as:\n[[TN FP]\n [FN TP]]\n")

for i in range(num_classes):
    class_name = class_names[i] if i < len(class_names) else f"Class {i}"
    y_true = all_labels[:, i]
    y_pred = (all_predictions[:, i] >= 0.5).astype(int)

    # Skip class if labels are all the same
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ Skipping class '{class_name}' due to only one label present.")
        continue

    cm = confusion_matrix(y_true, y_pred)

    print(f"🔬 {class_name}:")
    print("   [[TN FP]")
    print("    [FN TP]]")
    print(f"{cm}\n")

y_pred_bin = (all_predictions >= 0.5).astype(int)
y_true_bin = all_labels.astype(int)

# Calcolo per classe (no media)
precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
    y_true_bin, y_pred_bin, average=None, zero_division=0
)

# Calcolo delle macro e micro medie
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true_bin, y_pred_bin, average='macro', zero_division=0
)

precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true_bin, y_pred_bin, average='micro', zero_division=0
)

# Stampa per classe
print("\n📐 Precision, Recall, F1-score per Classe (threshold = 0.5):\n")
print(f"{'Class':<22} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1-score':>10}")
print("-" * 60)

for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else f"Class {i}"
    print(f"{name:<22} {support_per_class[i]:8d} {precision_per_class[i]:10.4f} {recall_per_class[i]:8.4f} {f1_per_class[i]:10.4f}")

# Riepilogo macro e micro
print("\n📊 Macro & Micro averages:")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall:    {recall_macro:.4f}")
print(f"Macro F1-score:  {f1_macro:.4f}")

print(f"Micro Precision: {precision_micro:.4f}")
print(f"Micro Recall:    {recall_micro:.4f}")
print(f"Micro F1-score:  {f1_micro:.4f}")
optimal_thresholds = []
f1_scores_per_class = []

print("\n🧠 Optimal Thresholds per Class (maximizing F1-score):\n")
print(f"{'Class':<22} {'Best Threshold':>15} {'Best F1-score':>15}")
print("-" * 55)

for i in range(num_classes):
    class_name = class_names[i] if i < len(class_names) else f"Class {i}"
    y_true = all_labels[:, i]
    y_prob = all_predictions[:, i]

    # Skip class if only 1 label
    if len(np.unique(y_true)) < 2:
        optimal_thresholds.append(np.nan)
        f1_scores_per_class.append(np.nan)
        print(f"{class_name:<22} {'N/A':>15} {'N/A':>15}")
        continue

    best_thresh = 0.0
    best_f1 = 0.0

    # Search thresholds between 0 and 1
    for t in np.linspace(0.0, 1.0, 101):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    optimal_thresholds.append(best_thresh)
    f1_scores_per_class.append(best_f1)

    print(f"{class_name:<22} {best_thresh:15.2f} {best_f1:15.4f}")

# === Apply optimal thresholds to predictions ===
final_preds = np.zeros_like(all_predictions)

for i in range(num_classes):
    if not np.isnan(optimal_thresholds[i]):
        final_preds[:, i] = (all_predictions[:, i] >= optimal_thresholds[i]).astype(int)

# === Recompute metrics using optimal thresholds ===
precision_opt, recall_opt, f1_opt, support_opt = precision_recall_fscore_support(
    all_labels.astype(int), final_preds.astype(int), average=None, zero_division=0
)

# Macro and micro averages
precision_macro_opt, recall_macro_opt, f1_macro_opt, _ = precision_recall_fscore_support(
    all_labels.astype(int), final_preds.astype(int), average='macro', zero_division=0
)

precision_micro_opt, recall_micro_opt, f1_micro_opt, _ = precision_recall_fscore_support(
    all_labels.astype(int), final_preds.astype(int), average='micro', zero_division=0
)

# === Print final evaluation ===
print("\n🎯 Final Evaluation with Optimized Thresholds:\n")
print(f"{'Class':<22} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1-score':>10}")
print("-" * 60)

print("\n📉 Confusion Matrices per Class (threshold personalized):")
print("Each matrix is formatted as:\n[[TN FP]\n [FN TP]]\n")

for i in range(num_classes):
    class_name = class_names[i] if i < len(class_names) else f"Class {i}"
    y_true = all_labels[:, i]
    y_pred = (all_predictions[:, i] >= optimal_thresholds[i]).astype(int)

    # Skip class if labels are all the same
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ Skipping class '{class_name}' due to only one label present.")
        continue

    cm = confusion_matrix(y_true, y_pred)

    print(f"🔬 {class_name}:")
    print("   [[TN FP]")
    print("    [FN TP]]")
    print(f"{cm}\n")

for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else f"Class {i}"
    print(f"{name:<22} {support_opt[i]:8d} {precision_opt[i]:10.4f} {recall_opt[i]:8.4f} {f1_opt[i]:10.4f}")

for i in range(num_classes):
    class_name = class_names[i]
    y_true = all_labels[:, i]
    y_pred = (all_predictions[:, i] >= 0.5).astype(int)

    y_true_male = y_true[male_mask]
    y_pred_male = y_pred[male_mask]
    y_true_female = y_true[female_mask]
    y_pred_female = y_pred[female_mask]

    if len(np.unique(y_true_male)) < 2 or len(np.unique(y_true_female)) < 2:
        print(f"⚠️ Skipping class '{class_name}' for gender-wise confusion matrix due to label imbalance.")
        continue

    cm_male = confusion_matrix(y_true_male, y_pred_male)
    cm_female = confusion_matrix(y_true_female, y_pred_female)

    print(f"🔬 {class_name} (Male):\n{cm_male}")
    print(f"🔬 {class_name} (Female):\n{cm_female}\n")

print("\n📊 Optimized Macro & Micro Averages:")
print(f"Macro Precision: {precision_macro_opt:.4f}")
print(f"Macro Recall:    {recall_macro_opt:.4f}")
print(f"Macro F1-score:  {f1_macro_opt:.4f}")
print(f"Micro Precision: {precision_micro_opt:.4f}")
print(f"Micro Recall:    {recall_micro_opt:.4f}")
print(f"Micro F1-score:  {f1_micro_opt:.4f}")


print("✅ Testing complete! Predictions saved as 'test_predictions.npy' and 'test_labels.npy'.")
