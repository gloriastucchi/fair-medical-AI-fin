import pandas as pd
df = pd.read_csv("passion_label.csv")
print(df["fitzpatrick"].value_counts())
print(df["fitzpatrick"].unique())
