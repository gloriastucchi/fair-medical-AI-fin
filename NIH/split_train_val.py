import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

# Load CSV
csv_file = "/work3/s232437/fair-medical-AI-fin/NIH/Data_Entry_2017_v2020_.csv"
all_images_file = "/work3/s232437/fair-medical-AI-fin/NIH/train_val_list.txt"

df = pd.read_csv(csv_file)
with open(all_images_file) as f:
    all_images = set(f.read().splitlines())

# Filter to only images in train_val_list
df = df[df["Image Index"].isin(all_images)]

# One-hot encode disease labels (excluding "No Finding")
labels = df["Finding Labels"].str.get_dummies(sep="|")
labels = labels[[col for col in labels.columns if col != "No Finding"]]

# Drop samples with no disease
df = df[labels.sum(axis=1) > 0].reset_index(drop=True)
labels = labels[labels.sum(axis=1) > 0].reset_index(drop=True)

# Stratified split
splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

train_imgs = df.iloc[train_idx]["Image Index"].tolist()
val_imgs = df.iloc[val_idx]["Image Index"].tolist()

# Save
with open("/work3/s232437/fair-medical-AI-fin/NIH/train_list.txt", "w") as f:
    f.write("\n".join(train_imgs))

with open("/work3/s232437/fair-medical-AI-fin/NIH/val_list.txt", "w") as f:
    f.write("\n".join(val_imgs))

print(f"✅ Wrote {len(train_imgs)} training and {len(val_imgs)} validation samples.")
