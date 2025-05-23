import torch
from torch.utils.data import DataLoader
from PASSION_model import PassionClassifier, ImpetigoBinaryClassifier
from PASSION_dataset import PassionDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import argparse
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

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

# --- Load subject-to-group map ---
df = dataset.df.drop_duplicates("subject_id")
if args.identity_column:
    subject_to_group = dict(zip(df["subject_id"], df[args.identity_column]))
else:
    subject_to_group = None

# --- Load model ---
if args.task == 'condition':
    model = PassionClassifier(num_classes=4)
else:
    model = ImpetigoBinaryClassifier()

model.load_state_dict(torch.load(args.model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# --- Evaluation loop ---
all_preds = []
all_labels = []
all_subjects = []

with torch.no_grad():
    for images, labels, subject_ids in data_loader:
        images = images.to(device)
        outputs = model(images)
        if args.task == 'impetigo':
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_subjects.extend(subject_ids)

# --- Majority vote per subject ---
subject_predictions = defaultdict(list)
subject_truth = {}
for pred, label, subj in zip(all_preds, all_labels, all_subjects):
    subject_predictions[subj].append(pred)
    subject_truth[subj] = label

y_true = []
y_pred = []
group_ids = []

for subj in subject_predictions:
    pred_votes = Counter(subject_predictions[subj]).most_common(1)[0][0]
    y_pred.append(pred_votes)
    y_true.append(subject_truth[subj])
    if subject_to_group:
        group_ids.append(subject_to_group[subj])

# --- Global metrics ---
print("\n📊 Evaluation on Test Set (Majority Vote per Subject):")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))

# --- Metrics per group ---
if subject_to_group:
    print("\n📊 Metrics by identity group ({}):".format(args.identity_column))
    unique_groups = sorted(set(group_ids))
    for group in unique_groups:
        mask = [g == group for g in group_ids]
        acc = accuracy_score(np.array(y_true)[mask], np.array(y_pred)[mask])
        print(f"Group {group}: Accuracy = {acc:.4f} (n={sum(mask)})")
