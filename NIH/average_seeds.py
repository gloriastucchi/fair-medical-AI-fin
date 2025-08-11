import pandas as pd

# Carica i CSV
df1 = pd.read_csv("/work3/s232437/fair-medical-AI-fin/NIH/NIH_eval_nofin_seed42.csv").set_index("Class")
df2 = pd.read_csv("/work3/s232437/fair-medical-AI-fin/NIH/NIH_eval_nofin_seed2025.csv").set_index("Class")
df3 = pd.read_csv("/work3/s232437/fair-medical-AI-fin/NIH/NIH_eval_nofin_seed221146.csv").set_index("Class")

# Concatena tutti i dataframe con chiavi
all_dfs = pd.concat([df1, df2, df3], axis=0, keys=['seed1', 'seed2', 'seed3'])

# Calcola media e std
df_mean = all_dfs.groupby(level=1).mean().round(4)
df_std = all_dfs.groupby(level=1).std().round(4)

# Rinomina colonne
df_std.columns = [col + "_std" for col in df_std.columns]

# Interleave: mettiamo accanto la colonna media e quella std per ogni metrica
interleaved_cols = []
for col in df_mean.columns:
    interleaved_cols.append(col)
    if col + "_std" in df_std.columns:
        interleaved_cols.append(col + "_std")

# Combina i due dataframe
df_combined = pd.concat([df_mean, df_std], axis=1)[interleaved_cols]

# Salva su CSV
df_combined.to_csv("averaged_results_with_std_interleaved_FINALNOFIN.csv")
