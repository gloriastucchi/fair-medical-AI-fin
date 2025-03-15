import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_list, transform=None, subset_size=500):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        print(f"✅ Available columns in dataset: {self.data.columns.tolist()}")

        self.transform = transform

        # ✅ Define all 15 possible labels from the full dataset
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
        ]

        # Filter dataset based on train/test split
        with open(image_list, "r") as f:
            self.image_names = set(f.read().splitlines())

        self.data = self.data[self.data["Image Index"].isin(self.image_names)]

        # Apply subset sampling (only for training, not testing)
        if subset_size and len(self.data) > subset_size:
            self.data = self.data.sample(n=subset_size, random_state=42)

        # ✅ One-hot encode labels and ensure 15 columns
        label_columns = self.data["Finding Labels"].str.get_dummies(sep="|")

        # ✅ Ensure all 15 labels exist, even if some are missing from the subset
        for label in self.all_labels:
            if label not in label_columns.columns:
                label_columns[label] = 0  # Add missing labels as zeros

        # ✅ Reorder columns to match the correct order
        label_columns = label_columns[self.all_labels]
        self.labels = label_columns.values

        # ✅ Print the number of classes detected (should always be 15)
        print(f"✅ Number of classes in dataset: {self.labels.shape[1]} (Expected: 15)")

        # Calculate and print gender percentages
        if "Patient Gender" in self.data.columns:
            gender_counts = self.data["Patient Gender"].value_counts(normalize=True) * 100
            print("\n📊 Gender Distribution in Subset:")
            print(f"  🧑 Male:   {gender_counts.get('M', 0):.2f}%")
            print(f"  👩 Female: {gender_counts.get('F', 0):.2f}%\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Index"]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure it's RGB

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Convert "M" → 0 and "F" → 1
        if self.data.iloc[idx]["Patient Gender"] == "M":
            identity_group = 0
        elif self.data.iloc[idx]["Patient Gender"] == "F":
            identity_group = 1
        else:
            raise ValueError(f"❌ Unexpected gender value: {self.data.iloc[idx]['Patient Gender']}")

        # Convert to tensor
        identity_group = torch.tensor(identity_group, dtype=torch.long)

        return image, label, identity_group


# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match 3-channel shape
])


# Example usage
if __name__ == "__main__":
    dataset = ChestXrayDataset(
        csv_file="/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv",
        image_folder="/Users/gloriastucchi/Desktop/NIH/images/",
        image_list="/Users/gloriastucchi/Desktop/NIH/train_val_list.txt",
        transform=transform
    )
    print(f"Dataset size: {len(dataset)}")
