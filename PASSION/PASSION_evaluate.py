import torch
from torch.utils.data import DataLoader
from PASSION_model import PassionClassifier, ImpetigoBinaryClassifier
from PASSION_dataset import PassionDataset
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
import torch.nn.functional as F

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--task', choices=['condition', 'impetigo'], default='condition')
parser.add_argument('--model_path', required=True)
parser.add_argument('--test_subjects', required=True)
parser.add_argument('--identity_column', default=None)
args = parser.parse_args()

# --- Load dataset ---
dataset = PassionDataset(
    csv_path=args.csv_path,
    img_dir=args.img_dir,
    task=args.task,
    subject_list=args.test_subjects
)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print("🧾 Available columns in dataset.df:", dataset.df.columns.tolist())



# --- Load subject-to-group map ---
#df = dataset.df.drop_duplicates("subject_id")
'''
if args.identity_column:
    subject_to_group = dict(zip(df["subject_id"], df[args.identity_column]))
else:
    subject_to_group = None
'''

# --- Load subject-to-group map ---
if args.identity_column:
    id_col = args.identity_column
    group_conflicts = dataset.df.groupby("subject_id")[id_col].nunique()
    conflicts = group_conflicts[group_conflicts > 1]
    if not conflicts.empty:
        print("\n⚠️ Warning: The following subjects have multiple identity group annotations:")
        print(conflicts)
        raise ValueError("Inconsistent identity group assignment per subject.")

    # Create subject_to_group map (safe, since each subject has a unique group)
    subject_to_group = dict(dataset.df.groupby("subject_id")[id_col].first())
else:
    subject_to_group = None


def print_test_dataset_overview(test_dataset):
    print("\n📊 Test Dataset Overview:")

    # Count identity group distribution
    group_counts = Counter(test_dataset.df['fitzpatrick'])
    group_df = pd.DataFrame(sorted(group_counts.items()), columns=["Fitzpatrick Group", "Count"])
    total_samples = sum(group_counts.values())
    group_df["Percentage"] = group_df["Count"] / total_samples * 100

    print("\n🔹 Samples per Fitzpatrick Skin Type:")
    print(group_df.to_string(index=False))

    # Count class distribution
    if args.task == "condition":
        label_column = "conditions_PASSION"
    elif args.task == "impetigo":
        label_column = "impetigo"
    else:
        raise ValueError("Unsupported task")

    class_counts = Counter(test_dataset.df[label_column])

    class_df = pd.DataFrame(sorted(class_counts.items()), columns=["True Class", "Count"])
    class_df["Percentage"] = class_df["Count"] / total_samples * 100

    print("\n🔹 Samples per True Class Label:")
    print(class_df.to_string(index=False))

    print(f"\n🔢 Total Test Samples: {total_samples}")

print_test_dataset_overview(dataset)
 
# --- Load model ---
if args.task == 'condition':
    model = PassionClassifier(num_classes=4)
    num_classes = 4
else:
    model = ImpetigoBinaryClassifier()
    num_classes = 1

model.load_state_dict(torch.load(args.model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# --- Evaluation loop ---
all_logits = []
all_labels = []
all_subjects = []

with torch.no_grad():
    for images, labels, subject_ids in data_loader:
        images = images.to(device)
        outputs = model(images)
        all_logits.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_subjects.extend(subject_ids)

logits = np.array(all_logits)
labels = np.array(all_labels)
preds = np.argmax(logits, axis=1)

# --- Majority vote ---
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
    probs = F.softmax(torch.tensor(avg_logits), dim=0).numpy()

    y_pred.append(vote)
    y_true.append(subject_truth[subj])
    y_score.append(probs)
    if subject_to_group:
        group_ids.append(subject_to_group[subj])

    prediction_percentages.append({
        "subject_id": subj,
        "true_label": subject_truth[subj],
        "predicted_label": vote,
        **{f"class_{i}_pct": round(p * 100, 2) for i, p in enumerate(probs)}
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

# --- Prediction Distribution ---
print("\n📈 Prediction Distribution:")
total = len(y_pred)
for cls in range(num_classes):
    pct = np.sum(y_pred == cls) / total * 100
    print(f"Class {cls}: {pct:.2f}%")

# --- Confusion Matrix ---
print("\n🧩 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- Prediction Percentages ---
print("\n📊 Prediction Percentages per Subject:")
df_pred_pct = pd.DataFrame(prediction_percentages)
print(df_pred_pct.head(10).to_string(index=False))
# df_pred_pct.to_csv("prediction_percentages.csv", index=False)

# --- Calibration ---
print("\n🎯 Calibration:")
y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
ece_total = 0
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    prob_true, prob_pred = calibration_curve(y_true_bin[:, i], y_score[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f"Class {i}")

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

print(f"\n🔢 Average ECE: {ece_total / num_classes:.4f}")

# --- Group metrics ---
if subject_to_group:
    print(f"\n📊 Metrics by identity group ({args.identity_column}):")
    unique_groups = sorted(set(group_ids))
    for group in unique_groups:
        mask = [g == group for g in group_ids]
        acc = accuracy_score(y_true[mask], y_pred[mask])
        print(f"Group {group}: Accuracy = {acc:.4f} (n={sum(mask)})")

# --- AUC and Recall per class ---
print("\n📈 Per-Class Metrics:")
print("{:<10} {:>10} {:>10}".format("Class", "AUC", "Recall"))

for cls in range(num_classes):
    y_true_bin_cls = (y_true == cls).astype(int)
    try:
        auc = roc_auc_score(y_true_bin_cls, y_score[:, cls])
    except ValueError:
        auc = float('nan')
    recall = recall_score(y_true, y_pred, labels=[cls], average=None)[0]
    print("{:<10} {:>10.4f} {:>10.4f}".format(f"Class {cls}", auc, recall))

# --- Equity gap (Recall / AUC gap per classe tra gruppi identitari) ---
if subject_to_group:
    print("\n📊 Equity Gaps (max-min across identity groups):")
    group_metrics = defaultdict(lambda: defaultdict(list))  # class -> metric -> list of values

    for group in unique_groups:
        mask = [g == group for g in group_ids]
        if sum(mask) == 0:
            continue
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        y_score_group = y_score[mask]

        for cls in range(num_classes):
            # Recall
            recall = recall_score(y_true_group, y_pred_group, labels=[cls], average=None)
            group_metrics[cls]['recall'].append(recall[0] if len(recall) else np.nan)

            # AUC
            y_true_bin_cls = (y_true_group == cls).astype(int)
            try:
                auc = roc_auc_score(y_true_bin_cls, y_score_group[:, cls])
            except ValueError:
                auc = float('nan')
            group_metrics[cls]['auc'].append(auc)

    print("{:<10} {:>12} {:>12}".format("Class", "Recall Gap", "AUC Gap"))
    for cls in range(num_classes):
        recalls = [v for v in group_metrics[cls]['recall'] if not np.isnan(v)]
        aucs = [v for v in group_metrics[cls]['auc'] if not np.isnan(v)]
        recall_gap = max(recalls) - min(recalls) if recalls else float('nan')
        auc_gap = max(aucs) - min(aucs) if aucs else float('nan')
        print("{:<10} {:>12.4f} {:>12.4f}".format(f"Class {cls}", recall_gap, auc_gap))

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
