import os
import pandas as pd
from tqdm import tqdm
df = pd.read_csv("/zhome/4b/b/202548/NIH/Data_Entry_2017_v2020_.csv")
img_folder = "/work3/s232437/images_full/"
missing = []

for f in tqdm(df["Image Index"]):
    if not os.path.exists(os.path.join(img_folder, f)):
        missing.append(f)

print(f"Missing {len(missing)} images")
