import torch
from torch.utils.data import DataLoader
from PASSION_model_fin import PassionClassifierFIN
from passion_dataset_fin import PassionDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
import argparse
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--task', choices=['condition', 'impetigo'], default='condition')
parser.add_argument('--model_path', required=True)
parser.add_argument('--test_subjects', required=True)
parser.add_argument('--identity_column', required=True)
parser.add_argument('--num_identities', type=int, required=True)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# --- Dataset ---
dataset = PassionDataset(
    csv_path=args.csv_path,
    img_dir=args.img_dir,
    task=args.task,
    subject_list=args.test_subjects,
    identity_column=args.identity_column
)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

# --- Model ---
model = PassionClassifierFIN(num_classes=4, num_identities=args.num_identities)
model.load_state_dict(torch.load(args.model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# --- Group map ---
df = dataset.df.drop_duplicates("subject_id")
subject_to_group = dict(zip(df["subject_id"], df[args.identity_column]))

def print_test_dataset_overview(dataset, identity_column, task):
    print("\n📊 Test Dataset Overview:")

    # Identity group distribution
    group_counts = Counter(dataset.df[identity_column])
    group_df = pd.DataFrame(sorted(group_counts.items()), columns=["Identity Group", "Count"])
    total_samples = sum(group_counts.values())
    group_df["Percentage"] = group_df["Count"] / total_samples * 100

    print("\n🔹 Samples per Identity Group:")
    print(group_df.to_string(index=False))

    # Class label distribution
    if task == "condition":
        label_column = "conditions_PASSION"
    elif task == "impetigo":
        label_column = "impetig"
    else:
        raise ValueError("Unsupported task")

    class_counts = Counter(dataset.df[label_column])
    class_df = pd.DataFrame(sorted(class_counts.items()), columns=["Class Label", "Count"])
    class_df["Percentage"] = class_df["Count"] / total_samples * 100

    print("\n🔹 Samples per Class Label:")
    print(class_df.to_string(index=False))
    print(f"\n🔢 Total Test Samples: {total_samples}")

# Call this after loading the dataset
print_test_dataset_overview(dataset, args.identity_column, args.task)

# 👇 Create contingency table: class label vs skin type
df = dataset.df.drop_duplicates("subject_id")
disease_skin_table = pd.crosstab(df['conditions_PASSION'], df[args.identity_column], margins=True, normalize='index') * 100

print("\n📊 Disease Class Distribution Across Skin Types (percentage within each class):")
print(disease_skin_table.round(2))
# Display the contingency table as a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(disease_skin_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
plt.title("Disease Class Distribution Across Skin Types")
plt.xlabel("Skin Type")
plt.ylabel("Disease Class")
plt.show()

# Count how many image entries exist per subject_id
img_per_subject = dataset.df.groupby("subject_id").size()

# Number of subjects with more than 1 image
n_multi_img_subjects = (img_per_subject > 1).sum()
print(f"📸 Subjects with >1 image: {n_multi_img_subjects} out of {len(img_per_subject)}")

# Number of unique disease labels per subject
labels_per_subject = dataset.df.groupby("subject_id")["conditions_PASSION"].nunique()

# Count how many subjects have >1 disease label
n_multi_label_subjects = (labels_per_subject > 1).sum()
print(f"🏷️ Subjects with >1 disease label: {n_multi_label_subjects} out of {len(labels_per_subject)}")

# Sample of such cases
print("\nSample subjects with multiple labels:")
print(labels_per_subject[labels_per_subject > 1].head(10))

# --- Evaluation loop ---
all_logits = []
all_labels = []
all_subjects = []

with torch.no_grad():
    for images, labels, identities in data_loader:
        images = images.to(device)
        identities = identities.to(device)
        outputs = model(images, identities)
        all_logits.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_subjects.extend(dataset.df["subject_id"].tolist()[len(all_subjects):len(all_subjects) + len(labels)])

logits = np.array(all_logits)
labels = np.array(all_labels)
preds = np.argmax(logits, axis=1)

# --- Majority vote per subject ---
subject_predictions = defaultdict(list)
subject_truth = {}
subject_logits = defaultdict(list)

for logit, label, pred, subj in zip(logits, labels, preds, all_subjects):
    subject_predictions[subj].append(pred)
    subject_truth[subj] = label
    subject_logits[subj].append(logit)

y_true = []
y_pred = []
y_score = []
group_ids = []
prediction_percentages = []

for subj in subject_predictions:
    vote = Counter(subject_predictions[subj]).most_common(1)[0][0]
    avg_logits = np.mean(subject_logits[subj], axis=0)
    probs = softmax(avg_logits)

    y_pred.append(vote)
    y_true.append(subject_truth[subj])
    y_score.append(probs)
    group_ids.append(subject_to_group[subj])

    prediction_percentages.append({
        "subject_id": subj,
        "true_label": subject_truth[subj],
        "predicted_label": vote,
        "class_0_pct": round(probs[0] * 100, 2),
        "class_1_pct": round(probs[1] * 100, 2),
        "class_2_pct": round(probs[2] * 100, 2),
        "class_3_pct": round(probs[3] * 100, 2)
    })

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

# --- Global metrics ---
print("\n📊 Evaluation on Test Set (Majority Vote per Subject):")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

# 🔥 Prediction distribution
print("\n📈 Prediction Distribution:")
total = len(y_pred)
for cls in range(4):
    pct = np.sum(y_pred == cls) / total * 100
    print(f"Class {cls}: {pct:.2f}%")

# 🔥 Confusion matrix
print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 🔥 Prediction percentages per subject
print("\n📊 Prediction Percentages per Subject:")
df_pred_pct = pd.DataFrame(prediction_percentages)
print(df_pred_pct.head(10).to_string(index=False))  # Show top 10
# Optionally save all:
# df_pred_pct.to_csv("prediction_percentages.csv", index=False)

# 🔥 Calibration curve (ECE)
print("\n🎯 Calibration:")
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
ece_total = 0

plt.figure(figsize=(8, 6))
for i in range(4):
    prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_score[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f"Class {i}")

    # ECE computation
    bin_conf = np.linspace(0.0, 1.0, 11)
    bins = np.digitize(y_score[:, i], bin_conf) - 1
    ece = 0
    for b in range(10):
        bin_mask = bins == b
        if np.any(bin_mask):
            acc = np.mean(y_true_bin[:, i][bin_mask])
            conf = np.mean(y_score[:, i][bin_mask])
            ece += np.abs(acc - conf) * np.sum(bin_mask) / len(y_score)
    print(f"Class {i} ECE: {ece:.4f}")
    ece_total += ece

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Calibration Curve")
plt.legend()
plt.grid()
plt.show()

print(f"\n🔢 Average ECE: {ece_total / 4:.4f}")

# --- Metrics by group ---
print(f"\n📊 Metrics by identity group ({args.identity_column}):")
unique_groups = sorted(set(group_ids))
for group in unique_groups:
    mask = [g == group for g in group_ids]
    acc = accuracy_score(y_true[mask], y_pred[mask])
    print(f"Group {group}: Accuracy = {acc:.4f} (n={sum(mask)})")

# --- AUC and Recall per class ---
print("\n📈 Per-Class Metrics:")
print("{:<10} {:>10} {:>10}".format("Class", "AUC", "Recall"))

for cls in range(4):
    y_true_bin_cls = (y_true == cls).astype(int)
    try:
        auc = roc_auc_score(y_true_bin_cls, y_score[:, cls])
    except ValueError:
        auc = float('nan')

    recall = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
    print("{:<10} {:>10.4f} {:>10.4f}".format(f"Class {cls}", auc, recall))

# --- Equity Gaps (max-min across identity groups) ---
print("\n📊 Equity Gaps (max-min across identity groups):")
print("{:<12} {:>15} {:>10}".format("Class", "Recall Gap", "AUC Gap"))

for cls in range(4):
    recall_per_group = []
    auc_per_group = []
    for g in unique_groups:
        mask = [gg == g for gg in group_ids]
        y_true_g = (np.array(y_true)[mask] == cls).astype(int)
        y_pred_g = (np.array(y_pred)[mask] == cls).astype(int)

        if np.sum(y_true_g) == 0:
            recall = np.nan
            auc = np.nan
        else:
            recall = recall_score(y_true_g, y_pred_g)
            try:
                auc = roc_auc_score(y_true_g, y_score[mask][:, cls])
            except ValueError:
                auc = np.nan

        recall_per_group.append(recall)
        auc_per_group.append(auc)

    recall_gap = np.nanmax(recall_per_group) - np.nanmin(recall_per_group)
    auc_gap = np.nanmax(auc_per_group) - np.nanmin(auc_per_group)
    print("{:<12} {:>15.4f} {:>10.4f}".format(f"Class {cls}", recall_gap, auc_gap))

print("\n📊 Per-Class × Identity Group Metrics (Recall & AUC):")

identity_groups = sorted(set(group_ids))
num_classes = y_score.shape[1]

recall_matrix = pd.DataFrame(index=[f"Class {c}" for c in range(num_classes)], columns=identity_groups)
auc_matrix = pd.DataFrame(index=[f"Class {c}" for c in range(num_classes)], columns=identity_groups)

for group in identity_groups:
    group_mask = np.array(group_ids) == group
    y_true_group = y_true[group_mask]
    y_pred_group = y_pred[group_mask]
    y_score_group = y_score[group_mask]

    for cls in range(num_classes):
        # Recall per gruppo identitario e classe
        recall = recall_score((y_true_group == cls).astype(int), (y_pred_group == cls).astype(int), zero_division=0)
        recall_matrix.loc[f"Class {cls}", group] = round(recall, 4)

        # AUC per gruppo identitario e classe
        try:
            auc = roc_auc_score((y_true_group == cls).astype(int), y_score_group[:, cls])
        except:
            auc = np.nan
        auc_matrix.loc[f"Class {cls}", group] = round(auc, 4)

print("\n🔬 Recall per classe × gruppo identitario:")
print(recall_matrix)

print("\n📐 AUC per classe × gruppo identitario:")
print(auc_matrix)
