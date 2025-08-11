import pandas as pd
import numpy as np
import sys

# Usage: python average_to_latex.py file1.csv file2.csv file3.csv

if len(sys.argv) != 4:
    print("Usage: python average_to_latex.py file1.csv file2.csv file3.csv")
    sys.exit(1)

files = sys.argv[1:]
dfs = [pd.read_csv(f) for f in files]

# Check all files have same rows and columns
for i, df in enumerate(dfs[1:], start=2):
    if not all(dfs[0]['Disease'] == df['Disease']):
        print(f"Error: Disease order does not match in file {i}.")
        sys.exit(2)

result = dfs[0].copy()
cols = [col for col in result.columns if col != "Disease"]
means = {}
stds = {}

for col in cols:
    vals = np.stack([df[col].values for df in dfs], axis=1)
    means[col] = np.mean(vals, axis=1)
    stds[col] = np.std(vals, axis=1)

# --- LaTeX table ---
header = (
    "\\begin{table}[h!]\n"
    "\\centering\n"
    "\\renewcommand{\\arraystretch}{1.4}\n"
    "\\resizebox{\\textwidth}{!}{\n"
    "\\begin{tabular}{lccccccccccc}\n"
    "\\toprule\n"
    "Disease & AUC (3) & AUC (4) & AUC (5) & AUC (6) & Overall AUC & Acc (3) & Acc (4) & Acc (5) & Acc (6) & Overall Acc & ES-AUC \\\\\n"
    "\\midrule\n"
)

rows = []
for i in range(len(result)):
    line = result.loc[i, "Disease"]
    for col in cols:
        m = means[col][i]
        s = stds[col][i]
        line += f" & {m:.4f} $\\pm$ {s:.4f}"
    line += " \\\\"
    rows.append(line)

footer = (
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "}\n"
    "\\caption{FIN Per-disease subgroup AUC, accuracy, and ES-AUC values (mean $\\pm$ std) on PASSION. Metrics are reported for Fitzpatrick groups 3--6 and overall.}\n"
    "\\label{tab:passion_final_results_std}\n"
    "\\end{table}\n"
)

table = header + "\n".join(rows) + "\n" + footer

# Save as .tex file
with open("FIN_average_results_table.tex", "w") as f:
    f.write(table)

print("\nLaTeX table saved as average_results_table.tex\n")
print(table)
