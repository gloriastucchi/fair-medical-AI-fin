import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Paths and constants
CSV_PATH = "/work3/s232437/fair-medical-AI-fin/NIH/Data_Entry_2017_v2020_.csv"
OUTPUT_DIR = "/work3/s232437/fair-medical-AI-fin/NIH/dataset_stats"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load NIH metadata
metadata = pd.read_csv(CSV_PATH)

# Gender distribution
gender_counts = metadata['Patient Gender'].value_counts()
print("\nGender Distribution:")
print(gender_counts)
with open(f"{OUTPUT_DIR}/gender_distribution.txt", "w") as f:
    f.write(gender_counts.to_string())

# Disease prevalence distribution (bar plot per gender)
all_labels = [label for sublist in metadata['Finding Labels'].str.split('|') for label in sublist]
all_unique_labels = sorted(list(set(all_labels)))

# Initialize counts per gender
gender_disease_counts = {g: {label: 0 for label in all_unique_labels} for g in ['M', 'F']}

for _, row in metadata.iterrows():
    gender = row['Patient Gender']
    labels = row['Finding Labels'].split('|')
    for label in labels:
        gender_disease_counts[gender][label] += 1

# Convert to DataFrame
plot_df = pd.DataFrame(gender_disease_counts).fillna(0).astype(int)
plot_df = plot_df.T  # Gender as rows

# Plot disease prevalence per gender
plot_df.T.plot(kind='bar', figsize=(16, 8))
plt.title("Disease Prevalence per Gender")
plt.ylabel("Number of Cases")
plt.xlabel("Disease")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/disease_distribution_by_gender.png")
plt.close()

print("\nSaved bar plot of disease prevalence per gender.")

# Label imbalance statistics
imbalance_df = plot_df.T.copy()
imbalance_df['Total'] = imbalance_df.sum(axis=1)
imbalance_df['Ratio_M_to_F'] = imbalance_df['M'] / (imbalance_df['F'] + 1e-6)  # avoid zero division
imbalance_df.sort_values('Total', ascending=False).to_csv(f"{OUTPUT_DIR}/label_imbalance_stats.csv")

print("Saved label imbalance statistics to CSV.")

# Plot label imbalance (total cases per label)
total_cases_per_label = imbalance_df['Total'].sort_values(ascending=False)
plt.figure(figsize=(14, 6))
total_cases_per_label.plot(kind='bar', color='salmon')
plt.title("Label Imbalance: Total Cases per Disease")
plt.ylabel("Number of Cases")
plt.xlabel("Disease")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/label_imbalance_barplot.png")
plt.close()

print("Saved bar plot of label imbalance.")

# Bar plot from label_imbalance_stats.csv
imbalance_stats = pd.read_csv(f"{OUTPUT_DIR}/label_imbalance_stats.csv", index_col=0)
plt.figure(figsize=(14, 6))
imbalance_stats['Ratio_M_to_F'].sort_values(ascending=False).plot(kind='bar', color='orchid')
plt.axhline(1.0, color='black', linestyle='--', linewidth=1)
plt.title("Male to Female Ratio per Disease")
plt.ylabel("Ratio M/F")
plt.xlabel("Disease")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/label_m_to_f_ratio_barplot.png")
plt.close()

print("Saved bar plot of male to female label ratio.")

# Co-occurrence matrix
co_matrix = pd.DataFrame(0, index=all_unique_labels, columns=all_unique_labels)

for findings in metadata['Finding Labels']:
    labels = findings.split('|')
    for label1 in labels:
        for label2 in labels:
            co_matrix.loc[label1, label2] += 1

# Heatmap of co-occurrence
plt.figure(figsize=(12, 10))
sns.heatmap(co_matrix, annot=False, cmap="viridis")
plt.title("Disease Co-occurrence Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/co_occurrence_matrix.png")
plt.close()

print("\nSaved heatmap of disease co-occurrence matrix.")

# Bar plot of co-occurrence (sum of co-occurrences per disease)
co_sum = co_matrix.sum(axis=1).sort_values(ascending=False)
plt.figure(figsize=(14, 6))
co_sum.plot(kind='bar', color='skyblue')
plt.title("Total Disease Co-occurrence Counts")
plt.ylabel("Number of Co-occurrences")
plt.xlabel("Disease")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/co_occurrence_barplot.png")
plt.close()

print("Saved bar plot of disease co-occurrence counts.")
