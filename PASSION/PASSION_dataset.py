import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class PassionDataset(Dataset):
    def __init__(self, csv_path, img_dir, task='condition', transform=None, subject_list=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.task = task
        self.transform = transform or self.get_default_transforms()

        # Apply subject filter
        if subject_list:
            with open(subject_list, "r") as f:
                allowed_subjects = set(line.strip() for line in f)
            self.df = self.df[self.df["subject_id"].isin(allowed_subjects)]

        # Map subject_id to all image paths
        self.df['images'] = self.df['subject_id'].apply(
            lambda sid: sorted([f for f in os.listdir(img_dir) if f.startswith(sid)])
        )
        self.df = self.df.explode('images').reset_index(drop=True)

        if task == 'condition':
            self.label_map = {label: idx for idx, label in enumerate(sorted(self.df['conditions_PASSION'].unique()))}
        elif task == 'impetigo':
            self.label_map = None  # binary

    def get_default_transforms(self):
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        return image, label, row['subject_id']

