import torch
from torch.utils.data import DataLoader
from model_passion_baseline import PassionClassifierNoFIN
from PASSION_dataset import PassionDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


# --- Argument parsing ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--test_subjects', required=True)
parser.add_argument('--identity_column', required=True)   # e.g., 'fitzpatrick'
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# --- Load dataset ---
dataset = PassionDataset(
    csv_path=args.csv_path,
    img_dir=args.img_dir,
    task="condition",
    subject_list=args.test_subjects
)
if len(dataset) == 0:
    raise ValueError("Test set is empty! Check your subject list file.")

data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

# --- Build subject-to-group map ---
id_col = args.identity_column
group_conflicts = dataset.df.groupby("subject_id")[id_col].nunique()
conflicts = group_conflicts[group_conflicts > 1]
if not conflicts.empty:
    print("\n⚠️ Warning: The following subjects have multiple identity group annotations:")
    print(conflicts)
    raise ValueError("Inconsistent identity group assignment per subject.")

subject_to_group = dict(dataset.df.groupby("subject_id")[id_col].first())

print("\n✅ Loaded test set with {} subjects and {} samples.".format(
    dataset.df['subject_id'].nunique(), len(dataset)
))
fitz_groups = sorted(set(subject_to_group.values()))
print("Available Fitzpatrick groups:", fitz_groups)

# --- Load model ---
model = PassionClassifierNoFIN(num_classes=4)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# --- Inference loop (collect predictions for all patches/images) ---
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

# --- Majority voting and feature aggregation by subject ---
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

for subj in subject_predictions:
    # Majority vote for predicted class per subject
    vote = Counter(subject_predictions[subj]).most_common(1)[0][0]
    # Mean logits across all subject's patches/images
    avg_logits = np.mean(subject_logits[subj], axis=0)
    probs = F.softmax(torch.tensor(avg_logits), dim=0).numpy()
    y_pred.append(vote)
    y_true.append(subject_truth[subj])
    y_score.append(probs)
    group_ids.append(subject_to_group[subj])

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)
group_ids = np.array(group_ids)
num_classes = y_score.shape[1]

class_names = ['Eczema', 'Fungal', 'Scabies', 'Other']

# --- Per-class, per-group metrics & ES-AUC ---
table_rows = []

for cls_idx, disease in enumerate(class_names):
    y_true_cls = (y_true == cls_idx).astype(int)
    y_pred_cls = (y_pred == cls_idx).astype(int)
    auc_group = {}
    acc_group = {}
    # For all fitz groups present
    for group in fitz_groups:
        mask = group_ids == group
        if np.sum(mask) == 0:
            auc, acc = np.nan, np.nan
        else:
            try:
                auc = roc_auc_score(y_true_cls[mask], y_score[mask, cls_idx])
            except ValueError:
                auc = np.nan
            acc = accuracy_score(y_true_cls[mask], y_pred_cls[mask])
        auc_group[group] = auc
        acc_group[group] = acc
    # Overall
    try:
        auc_overall = roc_auc_score(y_true_cls, y_score[:, cls_idx])
    except ValueError:
        auc_overall = np.nan
    acc_overall = accuracy_score(y_true_cls, y_pred_cls)
    # ES-AUC: as in the Harvard paper
    delta = sum([abs(auc_overall - auc_group[g]) for g in fitz_groups if not np.isnan(auc_group[g])])
    es_auc = auc_overall / (1 + delta) if not np.isnan(auc_overall) else np.nan

    table_rows.append([
        disease,
        *[auc_group[g] for g in fitz_groups],
        auc_overall,
        *[acc_group[g] for g in fitz_groups],
        acc_overall,
        es_auc
    ])

# Dynamic column headers
cols = (
    ["Disease"] +
    [f"AUC ({g})" for g in fitz_groups] +
    ["Overall AUC"] +
    [f"Acc ({g})" for g in fitz_groups] +
    ["Overall Acc", "ES-AUC"]
)
df = pd.DataFrame(table_rows, columns=cols)

# --- Format for print and LaTeX ---
for col in df.columns[1:]:
    df[col] = df[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "--")

print("\n==== Per-Class Table (AUC/Acc/ES-AUC by Fitzpatrick Group) ====")
print(df.to_string(index=False))
print("\nLaTeX table (copy-paste into thesis):\n")
print(df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns)-1)))
# --- Save results ---
output_path = args.model_path.replace('.pth', '_results.csv')
df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")
from sklearn.metrics import confusion_matrix

print("\n==== Confusion Matrix by Fitzpatrick Group ====")
for group in fitz_groups:
    mask = group_ids == group
    if np.sum(mask) == 0:
        print(f"\nFitzpatrick {group}: (no subjects)")
        continue
    y_true_g = y_true[mask]
    y_pred_g = y_pred[mask]
    cm = confusion_matrix(y_true_g, y_pred_g, labels=list(range(num_classes)))
    print(f"\nFitzpatrick {group}:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
