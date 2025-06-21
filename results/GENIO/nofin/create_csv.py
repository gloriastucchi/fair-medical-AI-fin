import os
import glob
import pandas as pd

# Cartelle
base_dir = "results/GENIO/nofin"
output_dir = "results/GENIO/nofin"
os.makedirs(output_dir, exist_ok=True)

# Pattern per i CSV da cercare
pattern = os.path.join(base_dir, "run_*/best_efficientnet_rnflt_race.csv")
files = glob.glob(pattern)

all_dfs = []

for file_path in files:
    run_name = os.path.basename(os.path.dirname(file_path))  # es: run_20250619_204945_rnflt_race
    try:
        df = pd.read_csv(file_path)
        df['run_name'] = run_name
        all_dfs.append(df)
    except Exception as e:
        print(f"Errore con {file_path}: {e}")

# Merge e aggiunta media
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Calcola la media per le colonne numeriche
    numeric_cols = merged_df.select_dtypes(include=['number']).columns
    mean_row = merged_df[numeric_cols].mean().to_dict()

    # Aggiungi i valori non numerici
    for col in merged_df.columns:
        if col not in mean_row:
            mean_row[col] = 'mean'

    # Aggiungi la riga al DataFrame
    merged_df = pd.concat([merged_df, pd.DataFrame([mean_row])], ignore_index=True)

    # Salva
    output_path = os.path.join(output_dir, "merged_best_nofin.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"✅ File salvato in: {output_path}")
else:
    print("⚠️ Nessun file trovato.")
