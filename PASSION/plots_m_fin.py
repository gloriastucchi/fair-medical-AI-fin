import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt

# Pattern per i tuoi CSV
csv_files = sorted(glob.glob('best_model_fin_condition_m*_results.csv'))

fitz_columns = ['AUC (3)', 'AUC (4)', 'AUC (5)', 'AUC (6)']

m_values = []
mean_gaps = []
max_gaps = []

for file in csv_files:
    # Match "_m<numero_decimale>_" (con punto o virgola)
    match = re.search(r'_m([0-9]+[,.]?[0-9]*)_', os.path.basename(file))
    if not match:
        print(f"Warning: Could not extract m from filename {file}")
        continue
    m_str = match.group(1).replace(',', '.')
    m_value = float(m_str)
    m_values.append(m_value)
    df = pd.read_csv(file)
    
    per_class_mean_gaps = []
    per_class_max_gaps = []
    
    for idx, row in df.iterrows():
        aucs = row[fitz_columns].dropna().values.astype(float)
        auc_gaps = [abs(aucs[i] - aucs[j]) for i in range(len(aucs)) for j in range(i+1, len(aucs))]
        if len(auc_gaps) == 0:
            continue
        mean_gap = sum(auc_gaps) / len(auc_gaps)
        max_gap = max(auc_gaps)
        per_class_mean_gaps.append(mean_gap)
        per_class_max_gaps.append(max_gap)
    
    # Media su tutte le classi per questo m
    mean_gaps.append(sum(per_class_mean_gaps) / len(per_class_mean_gaps) if per_class_mean_gaps else float('nan'))
    max_gaps.append(sum(per_class_max_gaps) / len(per_class_max_gaps) if per_class_max_gaps else float('nan'))

# Ordina per m crescente per plot ordinato
sorted_indices = sorted(range(len(m_values)), key=lambda i: m_values[i])
m_values = [m_values[i] for i in sorted_indices]
mean_gaps = [mean_gaps[i] for i in sorted_indices]
max_gaps = [max_gaps[i] for i in sorted_indices]

# Plot
plt.figure(figsize=(7,5))
plt.plot(m_values, mean_gaps, marker='o', label='Mean AUC Gap')
plt.plot(m_values, max_gaps, marker='s', label='Max AUC Gap')
plt.xlabel('Blending Parameter $m$')
plt.ylabel('AUC Gap')
plt.title('Fitzpatrick Group AUC Gaps vs $m$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('fitz_auc_gap_vs_m.png', dpi=300)
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

# Mean AUC Gap: purple, dashed-dot, circles
plt.plot(
    m_values, mean_gaps,
    color='#6A5ACD',    # purple
    marker='o',
    linestyle='-.',
    linewidth=2,
    markersize=7,
    label='Mean AUC Gap'
)

# Max AUC Gap: black, solid, squares
plt.plot(
    m_values, max_gaps,
    color='black',
    marker='s',
    linestyle='-',
    linewidth=2,
    markersize=8,
    label='Max AUC Gap'
)

plt.xlabel('$m$', fontsize=14)
plt.ylabel('AUC Gap', fontsize=14)
plt.title('Group AUC Gaps (Mean and Max) across m values', fontsize=15)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fitz_auc_gap_vs_m_style.png', dpi=300)
plt.show()


# Salva anche la tabella CSV riassuntiva
summary_df = pd.DataFrame({'m': m_values, 'mean_auc_gap': mean_gaps, 'max_auc_gap': max_gaps})
summary_df.to_csv('fitz_auc_gap_summary.csv', index=False)
