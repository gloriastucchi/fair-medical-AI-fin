import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# === CONFIG ===
EVAL_FILES = glob.glob("/work3/s232437/fair-medical-AI-fin/NIH/m*_NIH_eval_fin.csv")
OUTPUT_DIR = "/work3/s232437/fair-medical-AI-fin/NIH/experiments_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Found eval files:", EVAL_FILES)

# === PROCESS EACH FILE TO BUILD ES_AUC CSVs ===
for path in sorted(EVAL_FILES):
    df = pd.read_csv(path)
    
    # Extract m value from filename, e.g., "m0,1_NIH_eval_fin.csv" -> "0,1"
    filename = os.path.basename(path)
    m_str = filename.split("_")[0][1:]  # strip 'm'
    
    class_names = sorted(list(set(c.split(" (")[0] for c in df["Class"])))
    
    rows = []
    for cls in class_names:
        try:
            auc_male = df[df["Class"] == f"{cls} (Male)"]["AUC"].values[0]
            auc_female = df[df["Class"] == f"{cls} (Female)"]["AUC"].values[0]
            
            auc_overall = (auc_male + auc_female) / 2
            group_gap = abs(auc_male - auc_overall) + abs(auc_female - auc_overall)
            esauc = auc_overall / (1 + group_gap)
            
            rows.append({
                "Class": cls,
                "AUC_Male": auc_male,
                "AUC_Female": auc_female,
                "AUC_Overall": auc_overall,
                "GroupGap": group_gap,
                "ES-AUC": esauc,
                "m": m_str
            })
        except Exception:
            continue
    
    result_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"ES_AUC_m{m_str}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

# === PLOT: AUC vs ES-AUC PER CLASS & M ===
es_auc_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "ES_AUC_m*.csv")))
df_all = pd.concat([pd.read_csv(f).assign(m=os.path.basename(f).split("_")[2][1:-4]) for f in es_auc_files])

# Convert m to float for sorting
df_all["m"] = df_all["m"].str.replace(",", ".").astype(float)

# Plot per class
classes = df_all["Class"].unique()
for cls in classes:
    data = df_all[df_all["Class"] == cls].sort_values("m")
    plt.figure(figsize=(8, 4))
    plt.plot(data["m"], data["AUC_Overall"], marker="o", label="AUC")
    plt.plot(data["m"], data["ES-AUC"], marker="s", label="ES-AUC")
    plt.title(f"{cls} - AUC and ES-AUC vs m")
    plt.xlabel("m")
    plt.ylabel("Score")
    plt.ylim(0.4, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"AUC_ES-AUC_{cls.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Saved plot for {cls} at {plot_path}")
