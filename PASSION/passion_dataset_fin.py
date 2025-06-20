import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class PassionDataset(Dataset):
    def __init__(self, csv_path, img_dir, task='condition', transform=None, subject_list=None, identity_column=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.task = task
        self.transform = transform or self.get_default_transforms()
        self.identity_column = identity_column

        # Filtro opzionale per i soggetti (train/val/test)
        if subject_list:
            with open(subject_list, "r") as f:
                allowed_subjects = set(line.strip() for line in f)
            self.df = self.df[self.df["subject_id"].isin(allowed_subjects)]

        # Associa immagini ai subject_id
        self.df['images'] = self.df['subject_id'].apply(
            lambda sid: sorted([f for f in os.listdir(img_dir) if f.startswith(sid)])
        )
        self.df = self.df.explode('images').reset_index(drop=True)

        # Mappa delle label
        if task == 'condition':
            self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['conditions_PASSION'].unique()))}
        elif task == 'impetigo':
            self.label_map = None  # binario

        # Mappa dell'identità (es. fitzpatrick → interi)
        if self.identity_column:
            # Filtro per includere solo i gruppi validi (FST III–VI)
            if self.identity_column == "fitzpatrick":
                self.df = self.df[self.df[self.identity_column].isin([3, 4, 5, 6])]

            self.identity_map = {
                val: i for i, val in enumerate(sorted(self.df[self.identity_column].dropna().unique()))
            }

    def get_default_transforms(self):
        return T.Compose([
            T.Resize((256, 256)),             # Resize a 256x256
            T.CenterCrop(224),                # Taglio centrato
            T.RandomHorizontalFlip(p=0.5),    # Flip orizzontale (light augmentation)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['images'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.task == 'condition':
            label = self.label_map[row['conditions_PASSION']]
        elif self.task == 'impetigo':
            label = int(row['impetig'])

        if self.identity_column:
            identity = self.identity_map[row[self.identity_column]]
            return image, label, identity
        else:
            return image, label, row['subject_id']
