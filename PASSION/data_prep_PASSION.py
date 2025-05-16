import numpy as np
import matplotlib.pyplot as plp
import pandas as pd

# Set paths
METADATA_CSV = "/work3/s232437/fair-medical-AI-fin/PASSION/passion_label.csv"  # Replace with the real path
IMAGES_DIR = "/work3/s232437/fair-medical-AI-fin/PASSION/images"          # Replace with the real path

df = pd.read_csv(METADATA_CSV)
# Basic dataset overview
print("✅ Dataset Overview:")
print(f"Total images: {len(df)}")
print("Columns available:", list(df.columns))
print(df.head())