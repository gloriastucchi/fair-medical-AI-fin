import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# === Percorsi ===
CSV_PATH = "/work3/s232437/fair-medical-AI-fin/PASSION/passion_label.csv"
IMG_DIR = "/work3/s232437/fair-medical-AI-fin/PASSION/images"

# === Carica CSV ===
df = pd.read_csv(CSV_PATH)
print(f"✅ Caricate {len(df)} righe dal CSV")

# === Mappa subject_id → lista di immagini ===
all_images = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
image_map = {}

for path in all_images:
    filename = os.path.basename(path)
    subject = filename.split("_")[0]  # es. AA00970040
    image_map.setdefault(subject, []).append(filename)

# === Colonna con lista immagini ===
df['images'] = df['subject_id'].map(image_map)
df['image_count'] = df['images'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['has_images'] = df['image_count'] > 0
print(f"\n🖼️ Soggetti con almeno una immagine: {df['has_images'].sum()}")

# === Statistiche ===
print("\n📊 Diagnosi:")
print(df['conditions_PASSION'].value_counts())

print("\n🎨 Fitzpatrick:")
print(df['fitzpatrick'].value_counts())

print("\n🚻 Genere:")
print(df['sex'].value_counts())

print("\n📈 Età:")
print(df['age'].describe())

df_exploded = df.explode("images")  # Una riga per immagine
print("\n🖼️ Immagini totali per diagnosi:")
print(df_exploded['conditions_PASSION'].value_counts())
