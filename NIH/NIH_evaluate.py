# Codice completo per valutazione con soglie ottimizzate, includendo metriche per classe, confusion matrices, e valutazione per genere
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel
from NIH_fin import FairChestXrayModel

# === Setup ===
parser = argparse.ArgumentParser()
parser.add_argument("--use_fin", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FILE = "/work3/s232437/fair-medical-AI-fin/NIH/Data_Entry_2017_v2020_.csv"
IMAGE_FOLDER = "/work3/s232437/images_full/images/"
TEST_LIST = "/work3/s232437/fair-medical-AI-fin/NIH/test_list.txt"

test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

num_classes = 14
num_groups = 2
feature_dim = 512

if args.use_fin:
    base_model = ChestXrayModel(num_classes=feature_dim)
    model = FairChestXrayModel(base_model, feature_dim, num_groups).to(device)
    model.load_state_dict(torch.load("/work3/s232437/fair-medical-AI-fin/NIH/0.4_NIH_model_fin_bce_loss0.0408.pth"))
else:
    model = ChestXrayModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("/work3/s232437/fair-medical-AI-fin/NIH/NIH_model_nofin_bce_loss0.0398.pth"))

model.eval()

all_predictions, all_labels, all_groups = [], [], []
with torch.no_grad():
    for batch in test_loader:
        images, labels, groups = [b.to(device) for b in batch]
        outputs = model(images, groups) if args.use_fin else model(images)
        all_predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_groups.extend(groups.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_groups = np.array(all_groups).flatten()

# === Optimal threshold search ===
optimal_thresholds = []
final_preds = np.zeros_like(all_predictions)
for i in range(num_classes):
    y_true, y_prob = all_labels[:, i], all_predictions[:, i]
    if len(np.unique(y_true)) < 2:
        optimal_thresholds.append(np.nan)
        continue
    best_thresh, best_j = 0.0, -1.0
    for t in np.linspace(0, 1, 101):
        y_pred_thresh = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh, labels=[0,1]).ravel()
        tpr = tp / (tp + fn + 1e-6)
        fpr = fp / (fp + tn + 1e-6)
        youden_j = tpr - fpr
        if youden_j > best_j:
            best_j = youden_j
            best_thresh = t
    optimal_thresholds.append(best_thresh)
    final_preds[:, i] = (y_prob >= best_thresh).astype(int)


class_names = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

male_mask = all_groups == 0
female_mask = all_groups == 1

rows = []

# === Inspect Thresholds ===
thresholds_df = pd.DataFrame({"Class": class_names, "OptimalThreshold": optimal_thresholds})
print("\n🔍 Optimal Thresholds per Class:")
print(thresholds_df.to_string(index=False))

# Optional: Save to CSV
thresholds_df.to_csv(f"NIH_thresholds_{'fin' if args.use_fin else 'nofin'}.csv", index=False)

# === Compute Per-Class + Gender Metrics Table ===
for i in range(num_classes):
    name = class_names[i] if i < len(class_names) else f"Class {i}"
    y_true = all_labels[:, i]
    y_pred = final_preds[:, i]
    y_prob = all_predictions[:, i]

    for gender, mask in zip(["Male", "Female"], [male_mask, female_mask]):
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]

        if len(np.unique(y_true_g)) < 2:
            continue

        cm = confusion_matrix(y_true_g, y_pred_g).ravel()
        tn, fp, fn, tp = cm if len(cm) == 4 else (0, 0, 0, 0)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true_g, y_pred_g, average='binary', zero_division=0)
        auc = roc_auc_score(y_true_g, y_prob_g) if len(np.unique(y_true_g)) == 2 else np.nan
        prevalence = np.mean(y_true_g)

        rows.append({
            "Class": f"{name} ({gender})",
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1-score": f1,
            "AUC": auc, "Prevalence": prevalence
        })

# === Print Summary Table ===
df = pd.DataFrame(rows)
print("\n📋 Summary Table (Class x Gender):")
print(df.to_string(index=False))

# === Macro and Micro ===
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, final_preds, average='macro', zero_division=0)
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, final_preds, average='micro', zero_division=0)

print("\n📊 Macro & Micro Averages:")
print(f"Macro Precision: {macro_p:.4f}")
print(f"Macro Recall:    {macro_r:.4f}")
print(f"Macro F1-score:  {macro_f1:.4f}")
print(f"Micro Precision: {micro_p:.4f}")
print(f"Micro Recall:    {micro_r:.4f}")
print(f"Micro F1-score:  {micro_f1:.4f}")

# === Per-gender Accuracy ===
acc_male = accuracy_score(all_labels[male_mask].flatten(), final_preds[male_mask].flatten())
acc_female = accuracy_score(all_labels[female_mask].flatten(), final_preds[female_mask].flatten())

print("\n🎯 Accuracy per Gender (with optimized thresholds):")
print(f"Male Accuracy:   {acc_male:.4f}")
print(f"Female Accuracy: {acc_female:.4f}")

df.to_csv(f"NIH_eval_{'fin' if args.use_fin else 'nofin'}.csv", index=False)
thresholds_df = pd.DataFrame({"Class": class_names, "OptimalThreshold": optimal_thresholds})
thresholds_df.to_csv(f"NIH_thresholds_{'fin' if args.use_fin else 'nofin'}.csv", index=False)
auc_macro = roc_auc_score(all_labels, all_predictions, average='macro')
auc_micro = roc_auc_score(all_labels, all_predictions, average='micro')
print(f"\nGlobal AUC (Macro): {auc_macro:.4f}")
print(f"Global AUC (Micro): {auc_micro:.4f}")
gender_gap_auc = df[df["Class"].str.contains("Male")]["AUC"].values - df[df["Class"].str.contains("Female")]["AUC"].values
print(f"\n📉 Avg AUC gap (Male - Female): {np.nanmean(gender_gap_auc):.4f}")
# === ES-AUC Computation per class ===
esauc_values = []
for cls in class_names:
    try:
        overall_auc = df[df["Class"].str.contains(cls)]["AUC"].mean()
        male_auc = df[df["Class"] == f"{cls} (Male)"]["AUC"].values[0]
        female_auc = df[df["Class"] == f"{cls} (Female)"]["AUC"].values[0]
        group_gaps = abs(male_auc - overall_auc) + abs(female_auc - overall_auc)
        es_auc = overall_auc / (1 + group_gaps)
    except Exception:
        es_auc = np.nan
    esauc_values.append(es_auc)

# Add to the dataframe (only once per class, Male row)
for i, es_auc in enumerate(esauc_values):
    for j in df.index:
        if df.loc[j, "Class"] == f"{class_names[i]} (Male)":
            df.loc[j, "ES-AUC"] = es_auc
        elif df.loc[j, "Class"] == f"{class_names[i]} (Female)":
            df.loc[j, "ES-AUC"] = ""  # Leave it blank for Female row

# Save updated CSV with ES-AUC
eval_csv_path = f"m0,4_NIH_eval_{'fin' if args.use_fin else 'nofin'}.csv"
df.to_csv(eval_csv_path, index=False)
print(f"\n📄 Saved updated evaluation CSV with ES-AUC: {eval_csv_path}")


print("\n✅ Evaluation complete!")
