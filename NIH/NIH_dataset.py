import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_list, transform=None, subset_size=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

        # Define all 15 possible labels from the full dataset
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

        # One-hot encode labels and ensure 15 columns
        label_columns = self.data["Finding Labels"].str.get_dummies(sep="|")

        # Ensure all 15 labels exist, even if some are missing from the subset
        for label in self.all_labels:
            if label not in label_columns.columns:
                label_columns[label] = 0  # Add missing labels as zeros

        # Reorder columns to match the correct order
        label_columns = label_columns[self.all_labels]
        self.labels = label_columns.values

        # Calculate per-class analytics
        self.calculate_class_statistics()

        # Plot class distributions
        self.plot_class_distributions()

    def calculate_class_statistics(self):
        # Calculate the number of positive samples for each class
        positive_counts = self.labels.sum(axis=0)
        total_samples = len(self.data)

        print("\n📊 Per-Class Statistics:")
        for idx, label in enumerate(self.all_labels):
            count = positive_counts[idx]
            prevalence = (count / total_samples) * 100
            print(f"  🔹 {label}:")
            print(f"     - Positive Samples: {count}")
            print(f"     - Prevalence: {prevalence:.2f}%\n")

    def plot_class_distributions(self):
        # Overall class distribution
        positive_counts = self.labels.sum(axis=0)
        total_samples = len(self.data)
        prevalence = (positive_counts / total_samples) * 100

        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.all_labels, y=prevalence, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Conditions')
        plt.ylabel('Prevalence (%)')
        plt.title('Overall Class Distribution')
        plt.tight_layout()
        plt.show()

        # Class distribution by gender
        if "Patient Gender" in self.data.columns:
            gender_groups = self.data.groupby("Patient Gender")
            for gender, group in gender_groups:
                gender_labels = group["Finding Labels"].str.get_dummies(sep="|")
                gender_counts = gender_labels.sum(axis=0)
                gender_prevalence = (gender_counts / len(group)) * 100

                plt.figure(figsize=(12, 6))
                sns.barplot(x=self.all_labels, y=gender_prevalence, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Conditions')
                plt.ylabel('Prevalence (%)')
                plt.title(f'Class Distribution for {"Male" if gender == "M" else "Female"} Patients')
                plt.tight_layout()
                plt.show()

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

if __name__ == "__main__":
    dataset = ChestXrayDataset(
        csv_file="/Users/gloriastucchi/Desktop/NIH/Data_Entry_2017_v2020_.csv",
        image_folder="/Users/gloriastucchi/Desktop/NIH/images/",
        image_list="/Users/gloriastucchi/Desktop/NIH/train_val_list.txt",
        transform=transform
    )
    print(f"Dataset size: {len(dataset)}")
