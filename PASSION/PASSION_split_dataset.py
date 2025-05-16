import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Stratification: ensures class proportions (eczema, scabies, etc) are preserved across the splits
# Grouping: ensures that all samples (images) from the same subject_id stay in the same split (prevents data leakage)

# Configuration
csv_path = "passion_label.csv"
train_list_path = "train_subjects.txt"
val_list_path = "val_subjects.txt"
test_list_path = "test_subjects.txt"
split_ratio = 0.8  # 80% for development (train+val), 20% for final test

# Load CSV
df = pd.read_csv(csv_path)
# estrae tutti id unici dei pazienti
subjects = df["subject_id"].unique()
labels = df.drop_duplicates("subject_id")["conditions_PASSION"]

# First I build a dictionary that maps each unique diagnosis label 
# to a unique integer index

# Encode labels
label_map = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
# make sure to consider only one row per subject_id. This avoids counting a subject multiple times (since each subject may have multiple images).
df_labels = df.drop_duplicates("subject_id")
# convert the string-based diagnosis labels into their corresponding numeric codes
# using the label_map we just created. This gives us a vector `y` that holds one label per subject.
# `y` will be used as the target for stratified splitting, to preserve class proportions
# across the training and validation sets !!
y = df_labels["conditions_PASSION"].map(label_map)

# First split: dev (80%) vs test (20%)
group_kfold_outer = StratifiedGroupKFold(n_splits=int(1 / (1 - split_ratio)), shuffle=True, random_state=42)
dev_idx, test_idx = next(group_kfold_outer.split(subjects, y, groups=subjects))

dev_subjects = subjects[dev_idx]
test_subjects = subjects[test_idx]

# Prepare dev split labels for further stratified split
dev_labels = df_labels[df_labels["subject_id"].isin(dev_subjects)]["conditions_PASSION"].map(label_map)

# Second split: train (80% of dev) vs val (20% of dev)
group_kfold_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(group_kfold_inner.split(dev_subjects, dev_labels, groups=dev_subjects))

train_subjects = dev_subjects[train_idx]
val_subjects = dev_subjects[val_idx]

# Save lists
with open(train_list_path, "w") as f:
    f.writelines([s + "\n" for s in train_subjects])
with open(val_list_path, "w") as f:
    f.writelines([s + "\n" for s in val_subjects])
with open(test_list_path, "w") as f:
    f.writelines([s + "\n" for s in test_subjects])

print(f"✅ Saved {len(train_subjects)} train subjects, {len(val_subjects)} val subjects, {len(test_subjects)} test subjects")
