import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PASSION_model_fin import PassionClassifierFIN, ImpetigoBinaryClassifierFIN
from passion_dataset_fin import PassionDataset
import argparse
import numpy as np

# --- Argparser ---
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--task', choices=['condition', 'impetigo'], default='condition')
parser.add_argument('--test_subjects', required=True)
parser.add_argument('--identity_column', default='gender')
parser.add_argument('--num_identities', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model_path', required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load test set ---
test_dataset = PassionDataset(
    args.csv_path, args.img_dir,
    task=args.task,
    subject_list=args.test_subjects,
    identity_column=args.identity_column
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# --- Load model ---
if args.task == 'condition':
    model = PassionClassifierFIN(num_classes=4, num_identities=args.num_identities)
else:
    model = ImpetigoBinaryClassifierFIN(num_identities=args.num_identities)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Evaluation loop ---
all_preds, all_labels, all_identities = [], [], []

with torch.no_grad():
    for images, labels, identities in test_loader:
        images = images.to(device)
        identities = identities.to(device)
        outputs = model(images, identities)

        if args.task == 'impetigo':
            preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
            preds = (preds > 0.5).astype(int)
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_identities.extend(identities.cpu().numpy())

# --- Metrics (overall) ---
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
print("✅ Overall Metrics")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

# --- Metrics per group (optional fairness insight) ---
print("\n📊 Metrics by identity group:")
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_identities = np.array(all_identities)

for group_id in range(args.num_identities):
    mask = all_identities == group_id
    group_acc = accuracy_score(all_labels[mask], all_preds[mask])
    print(f"Group {group_id}: Accuracy = {group_acc:.4f} (n={mask.sum()})")

