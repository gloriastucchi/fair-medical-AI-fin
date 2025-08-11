import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
EVAL_FILES = glob.glob("/work3/s232437/fair-medical-AI-fin/NIH/experiments_graphs/ES_AUC_m*.csv")
OUTPUT_DIR = "/work3/s232437/fair-medical-AI-fin/NIH/experiments_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Found eval files:", EVAL_FILES)
# === READ AND COLLECT ===
all_data = []
for path in sorted(EVAL_FILES):
    df = pd.read_csv(path)
    df["m"] = df["m"].astype(str)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# === OPTION 1: Macro average per m ===
macro_df = combined_df.groupby("m").agg({
    "AUC_Overall": "mean",
    "ES-AUC": "mean"
}).reset_index()

# === OPTION 2: Prevalence-weighted average (Micro) — only if we had prevalence ===
# For now, we simulate equal prevalence since prevalence column isn't available here.
# (To do proper micro, we would merge original CSVs and re-calculate.)

# === OPTION 3: Mean ± Std ===
std_df = combined_df.groupby("m").agg({
    "AUC_Overall": "std",
    "ES-AUC": "std"
}).reset_index()

macro_df["AUC_std"] = std_df["AUC_Overall"]
macro_df["ESAUC_std"] = std_df["ES-AUC"]

# === OPTION 4: Boxplot Data ===
# Stored in combined_df directly.

# === PLOT OPTION 1 & 3: Lineplot with std ===
plt.figure(figsize=(11, 6))
x_vals = macro_df["m"]
x_ticks = np.arange(len(x_vals))

plt.plot(x_ticks, macro_df["AUC_Overall"], label="Macro AUC", marker="o")
plt.fill_between(x_ticks,
                 macro_df["AUC_Overall"] - macro_df["AUC_std"],
                 macro_df["AUC_Overall"] + macro_df["AUC_std"],
                 alpha=0.2)

plt.plot(x_ticks, macro_df["ES-AUC"], label="Macro ES-AUC", marker="s")
plt.fill_between(x_ticks,
                 macro_df["ES-AUC"] - macro_df["ESAUC_std"],
                 macro_df["ES-AUC"] + macro_df["ESAUC_std"],
                 alpha=0.2)

plt.xticks(x_ticks, x_vals, rotation=45)
plt.ylabel("Score")
plt.xlabel("m")
plt.title("Macro AUC and ES-AUC with ±1 STD")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "macro_auc_esauc_lineplot.png"))

# === PLOT OPTION 4: Boxplot ===
plt.figure(figsize=(12, 6))
combined_df.boxplot(column="ES-AUC", by="m")
plt.title("ES-AUC per Class (Boxplot)")
plt.suptitle("")
plt.xlabel("m")
plt.ylabel("ES-AUC")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "esauc_boxplot.png"))

plt.figure(figsize=(12, 6))
combined_df.boxplot(column="AUC_Overall", by="m")
plt.title("AUC per Class (Boxplot)")
plt.suptitle("")
plt.xlabel("m")
plt.ylabel("AUC")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "auc_boxplot.png"))

print("✅ All plots saved in:", OUTPUT_DIR)

# === PLOT OPTION 1 & 3: Lineplot with std ===
plt.figure(figsize=(11, 6))
x_vals = macro_df["m"]
x_ticks = np.arange(len(x_vals))

# Custom colors
color_auc = "#6A5ACD"      # Slate Blue
fill_auc = "#C0C8F0"       # Light Slate fill
color_esauc = "#556B2F"    # Dark Olive Green
fill_esauc = "#C6D8AF"     # Olive fill

# Plot AUC
plt.plot(x_ticks, macro_df["AUC_Overall"], label="Macro AUC", marker="o", color=color_auc)
plt.fill_between(x_ticks,
                 macro_df["AUC_Overall"] - macro_df["AUC_std"],
                 macro_df["AUC_Overall"] + macro_df["AUC_std"],
                 color=fill_auc, alpha=0.4)

# Plot ES-AUC
plt.plot(x_ticks, macro_df["ES-AUC"], label="Macro ES-AUC", marker="s", color=color_esauc)
plt.fill_between(x_ticks,
                 macro_df["ES-AUC"] - macro_df["ESAUC_std"],
                 macro_df["ES-AUC"] + macro_df["ESAUC_std"],
                 color=fill_esauc, alpha=0.5)

plt.xticks(x_ticks, x_vals, rotation=45)
plt.ylabel("Score")
plt.xlabel("m")
plt.title("Macro AUC and ES-AUC with ±1 STD")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "macro_auc_esauc_lineplot_colored.png"))


# === RICARICA E UNISCI TUTTI I CSV ===
esauc_files = glob.glob(os.path.join(OUTPUT_DIR, "ES_AUC_m*.csv"))
df_all = pd.concat([
    pd.read_csv(f).assign(m=os.path.basename(f).split("m")[1].replace(".csv", "")) 
    for f in esauc_files
])
df_all["m"] = df_all["m"].str.replace(",", ".").astype(float)
 
# === GAP ANALYSIS ===
gap_summary = []

for m_val, group in df_all.groupby("m"):
    gaps = []
    for cls in group["Class"].unique():
        male_auc = group[(group["Class"] == cls)]["AUC_Male"].values
        female_auc = group[(group["Class"] == cls)]["AUC_Female"].values
        if len(male_auc) == 1 and len(female_auc) == 1:
            gap = abs(male_auc[0] - female_auc[0])
            gaps.append(gap)
    
    if gaps:
        gap_summary.append({
            "m": m_val,
            "Mean_AUC_Gap": np.mean(gaps),
            "Max_AUC_Gap": np.max(gaps)
        })

gap_df = pd.DataFrame(gap_summary).sort_values("m")
gap_df.to_csv(os.path.join(OUTPUT_DIR, "gap_metrics_summary.csv"), index=False)

# === PLOT GAP METRICS ===
color_auc = "#6A5ACD"      # Slate Blue
fill_auc = "#C0C8F0"       # Light Slate fill
color_esauc = "#C6D8AF"    # Dark Olive Green
fill_esauc = "#C6D8AF"     # Olive fill
plt.figure(figsize=(10, 5))
plt.plot(gap_df["m"], gap_df["Mean_AUC_Gap"], label="Mean AUC Gap", marker="o", color=color_auc, linestyle=	'dashdot')
plt.plot(gap_df["m"], gap_df["Max_AUC_Gap"], label="Max AUC Gap", marker="s", color=color_esauc, linestyle='solid')
plt.xlabel("m")
plt.ylabel("AUC Gap")
plt.title("Group AUC Gaps (Mean and Max) across m values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "auc_gap_plot.png"))
plt.close()

print("📊 Saved AUC Gap plot and summary.")
