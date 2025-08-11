import pandas as pd
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

csv_files = sorted(glob.glob('best_model_fin_condition_m*_results.csv'))

fitz_auc_cols = ['AUC (3)', 'AUC (4)', 'AUC (5)', 'AUC (6)']

m_values = []
mean_group_aucs = []
std_group_aucs = []
mean_class_esaucs = []
std_class_esaucs = []

for file in csv_files:
    match = re.search(r'_m([0-9]+[,.]?[0-9]*)_', file)
    if not match:
        print(f"Skipping file (cannot extract m): {file}")
        continue
    m_value = float(match.group(1).replace(',', '.'))
    m_values.append(m_value)
    df = pd.read_csv(file)
    
    # AUCs across groups (all classes, all groups)
    aucs = []
    for idx, row in df.iterrows():
        aucs.extend(row[fitz_auc_cols].dropna().values.astype(float))
    mean_group_aucs.append(np.mean(aucs))
    std_group_aucs.append(np.std(aucs))
    
    # ES-AUC across classes
    esaucs = df['ES-AUC'].dropna().values.astype(float)
    mean_class_esaucs.append(np.mean(esaucs))
    std_class_esaucs.append(np.std(esaucs))

# Sort for plotting
sorted_indices = np.argsort(m_values)
m_values = np.array(m_values)[sorted_indices]
mean_group_aucs = np.array(mean_group_aucs)[sorted_indices]
std_group_aucs = np.array(std_group_aucs)[sorted_indices]
mean_class_esaucs = np.array(mean_class_esaucs)[sorted_indices]
std_class_esaucs = np.array(std_class_esaucs)[sorted_indices]

plt.figure(figsize=(8,5))

# Black solid line: mean group AUC
plt.plot(m_values, mean_group_aucs, color='black', marker='o', linewidth=2, label='Mean Group AUC')
plt.fill_between(
    m_values,
    mean_group_aucs - std_group_aucs,
    mean_group_aucs + std_group_aucs,
    color='grey', alpha=0.15
)

# Purple dash-dot line: mean class ES-AUC
plt.plot(
    m_values, mean_class_esaucs, color='#6A5ACD', marker='s', linewidth=2, linestyle='-.', label='Mean Class ES-AUC'
)
plt.fill_between(
    m_values,
    mean_class_esaucs - std_class_esaucs,
    mean_class_esaucs + std_class_esaucs,
    color='#6A5ACD', alpha=0.13
)

plt.xlabel('$m$', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Mean Group AUC and Macro ES-AUC vs $m$', fontsize=15)
plt.legend(fontsize=12, loc='lower left')
plt.grid(True, alpha=0.18)
plt.tight_layout()
plt.savefig('mean_group_auc_esauc_vs_m.png', dpi=300)
plt.show()
